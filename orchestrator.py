"""
orchestrator.py
===============
PHASE 7a-8 — Parameterized Benchmark Orchestrator.

A parameterized function that runs one benchmark run. A CLI wrapper calls
it in a loop across the default concurrency matrix. Every telemetry record
written includes a run_metadata block.

Usage (CLI):
  python orchestrator.py --engine vllm_local --quantization none --duration 30
  python orchestrator.py --engine custom_runtime --quantization q4 --duration 10 --concurrency 50

The CLI loops over DEFAULT_CONCURRENCY_LEVELS unless --concurrency is given.
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# The default benchmark matrix. The CLI wrapper calls run_benchmark in a
# loop over these. NOT passed into run_benchmark itself.
DEFAULT_CONCURRENCY_LEVELS = [1, 10, 50, 100]

GATEWAY_URL = "http://localhost:8000/v1/chat/completions"

# Default prompt used for benchmarking; short and deterministic.
_BENCH_PROMPT = "Briefly describe the second law of thermodynamics in one sentence."


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

async def run_benchmark(
    engine: str,
    quantization: str,
    concurrency: int,
    duration_seconds: int,
    run_id: str,
    injected_failure: Optional[str] = None,
) -> Path:
    """
    Runs a benchmark against the live gateway for `duration_seconds`.

    Parameters
    ----------
    engine : str
        Adapter name, e.g. "vllm_local", "custom_runtime".
    quantization : str
        Quantization label, e.g. "q4", "q8", "none". Written to run_metadata.
    concurrency : int
        Number of simultaneous requests permitted at once (semaphore cap).
    duration_seconds : int
        Wall-clock seconds to keep sending requests.
    run_id : str
        Unique ID for this run — caller provides it.
    injected_failure : str | None
        Failure type label from VALID_FAILURE_TYPES, or None for a clean run.

    Returns
    -------
    Path
        The path to telemetry_{run_id}.jsonl for this run.
    """
    run_metadata = {
        "run_id": run_id,
        "agent_enabled": False,      # always False in Phase 7
        "policy_version": None,      # null until Phase 12 sets a version
        "injected_failure": injected_failure,
        "engine": engine,
        "quantization": quantization,
        "concurrency": concurrency,
    }

    telemetry_path = Path(f"telemetry_{run_id}.jsonl")

    # Start power logger before the first request.
    power_proc = _start_power_logger(run_id)

    sem = asyncio.Semaphore(concurrency)
    deadline = time.monotonic() + duration_seconds
    tasks = []

    async with httpx.AsyncClient(timeout=120.0) as session:
        while time.monotonic() < deadline:
            async with sem:
                task = asyncio.create_task(
                    _send_single_request(session, engine, run_id, run_metadata, telemetry_path)
                )
                tasks.append(task)
            # Tiny yield so the event loop can start tasks.
            await asyncio.sleep(0)

        # Wait for all in-flight requests to complete.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # Stop power logger only after all requests have actually completed.
    _stop_power_logger(power_proc)

    return telemetry_path


async def _send_single_request(
    session: httpx.AsyncClient,
    engine: str,
    run_id: str,
    run_metadata: dict,
    telemetry_path: Path,
) -> None:
    """
    Sends one request to the gateway and writes one telemetry record.

    Records: timestamp, latency_ms, status_code, run_metadata block.

    Parameters
    ----------
    session : httpx.AsyncClient
        Shared HTTP client (connection-pooled).
    engine : str
        Target engine adapter name.
    run_id : str
        Unique run identifier (for log correlation).
    run_metadata : dict
        Metadata block attached to every telemetry record.
    telemetry_path : Path
        JSONL file to append this record to.
    """
    payload = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": _BENCH_PROMPT}],
        "engine": engine,
        "tier": "free",
        "max_tokens": 50,
    }

    t_start = time.monotonic()
    status_code = -1
    error_msg = None

    try:
        response = await session.post(GATEWAY_URL, json=payload)
        status_code = response.status_code
        # Consume the full SSE body (do not buffer tokens for latency accuracy).
        await response.aread()
    except httpx.RequestError as exc:
        error_msg = str(exc)
    finally:
        latency_ms = round((time.monotonic() - t_start) * 1000, 2)

    record = {
        "timestamp": time.time(),
        "latency_ms": latency_ms,
        "status_code": status_code,
        "run_metadata": run_metadata,
    }
    if error_msg:
        record["error"] = error_msg

    try:
        with telemetry_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        print(f"[orchestrator] telemetry write error: {exc}", file=sys.stderr)


def _start_power_logger(run_id: str) -> subprocess.Popen:
    """
    Starts `nvidia-smi --query-gpu=power.draw --format=csv -l 1`
    in the background, writing stdout to power_draw_{run_id}.log.

    Returns the Popen object so the caller can terminate it when the run ends.

    Parameters
    ----------
    run_id : str
        Used to name the output log file.
    """
    log_path = Path(f"power_draw_{run_id}.log")
    try:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv",
                "-l", "1",
            ],
            stdout=log_path.open("w", encoding="utf-8"),
            stderr=subprocess.DEVNULL,
        )
        return proc
    except FileNotFoundError:
        # nvidia-smi not available (CPU-only machine). Write a placeholder log.
        log_path.write_text("power.draw [W]\nN/A (nvidia-smi not found)\n")

        class _NullProc:
            """Dummy Popen replacement for CPU-only environments."""
            def terminate(self): pass
            def wait(self): pass

        return _NullProc()


def _stop_power_logger(proc: subprocess.Popen) -> None:
    """
    Terminates the nvidia-smi process cleanly.

    Parameters
    ----------
    proc : subprocess.Popen
        The Popen object returned by _start_power_logger.
    """
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        pass  # already exited or _NullProc


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the gateway across concurrency levels."
    )
    parser.add_argument("--engine", default="vllm_local", help="Adapter name.")
    parser.add_argument("--quantization", default="none", help="Quantization label.")
    parser.add_argument("--duration", type=int, default=30, help="Seconds per run.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Single concurrency level. Omit to loop over DEFAULT_CONCURRENCY_LEVELS.",
    )
    parser.add_argument(
        "--injected-failure",
        dest="injected_failure",
        default=None,
        help="Failure type label from VALID_FAILURE_TYPES, or omit for clean run.",
    )
    args = parser.parse_args()

    levels = [args.concurrency] if args.concurrency is not None else DEFAULT_CONCURRENCY_LEVELS

    async def _main():
        for level in levels:
            run_id = str(uuid.uuid4())
            print(
                f"[orchestrator] Starting run: engine={args.engine} "
                f"quant={args.quantization} concurrency={level} "
                f"duration={args.duration}s run_id={run_id}"
            )
            path = await run_benchmark(
                engine=args.engine,
                quantization=args.quantization,
                concurrency=level,
                duration_seconds=args.duration,
                run_id=run_id,
                injected_failure=args.injected_failure,
            )
            print(f"[orchestrator] Done. Telemetry: {path}")

    asyncio.run(_main())
