"""
gpu_monitor.py
==============
PHASE 7a-1 / PHASE 9 — GPU Cache Usage Poller.

Background asyncio.Task that polls the Prometheus metrics endpoint for
gpu_cache_usage_pct and stores allocated% in one shared float. Request
handlers read this variable — they never write it.

Why a single shared variable is safe here:
  asyncio is single-threaded. One writer (poll_gpu_metrics) + many readers
  (request handlers) never execute concurrently on the same thread, so there
  is no torn-write risk and no lock is needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import re
import time
from typing import Callable, Optional

import httpx

logger = logging.getLogger("gpu_monitor")

# ---------------------------------------------------------------------------
# SHARED VARIABLE AND STATE
# ---------------------------------------------------------------------------
_gpu_cache_usage_pct: float = 0.0
_is_failing: bool = False
_last_failure_log_time: float = 0.0

# Path to the telemetry file written by the vLLM engine.
_BATCH_EVENTS_PATH = Path("batch_events.jsonl")


def get_gpu_metrics_url() -> str:
    """
    Returns the target URL for GPU Prometheus metrics polling.
    Honors LANDAUER_GPU_METRICS_URL if set, defaulting to http://127.0.0.1:8002/metrics.
    """
    env_url = os.environ.get("LANDAUER_GPU_METRICS_URL", "").strip()
    return env_url if env_url else "http://127.0.0.1:8002/metrics"


def reset_poller_state() -> None:
    """
    Resets the internal failure tracking and cache usage state (primarily for unit tests).
    """
    global _gpu_cache_usage_pct, _is_failing, _last_failure_log_time
    _gpu_cache_usage_pct = 0.0
    _is_failing = False
    _last_failure_log_time = 0.0


async def poll_gpu_metrics_step(
    client: httpx.AsyncClient,
    url: str,
    clock: Callable[[], float] = time.monotonic,
) -> float:
    """
    Executes a single poll iteration against the GPU metrics endpoint and updates shared state.

    Inputs:
    - client: Async HTTP client used to execute GET requests.
    - url: Prometheus metrics endpoint URL.
    - clock: Monotonic time callable used for failure rate-limiting.

    Returns:
    - Float representing the current GPU KV-cache usage percentage.
    """
    global _gpu_cache_usage_pct, _is_failing, _last_failure_log_time

    try:
        response = await client.get(url, timeout=2.0)
        response.raise_for_status()
        match = re.search(r"^gpu_cache_usage_pct\s+([\d\.]+)", response.text, re.MULTILINE)
        if match:
            _gpu_cache_usage_pct = float(match.group(1))
        else:
            _gpu_cache_usage_pct = 0.0

        if _is_failing:
            _is_failing = False
            logger.info("GPU metrics poller recovered")

    except Exception as exc:
        _gpu_cache_usage_pct = 0.0
        now = clock()
        if not _is_failing:
            _is_failing = True
            _last_failure_log_time = now
            logger.warning("GPU metrics poll error: %s: %s", type(exc).__name__, exc)
        elif now - _last_failure_log_time >= 60.0:
            _last_failure_log_time = now
            logger.warning("GPU metrics poll still failing: %s: %s", type(exc).__name__, exc)

    return _gpu_cache_usage_pct


async def poll_gpu_metrics(
    interval_seconds: float = 0.1,
    url: Optional[str] = None,
    client: Optional[httpx.AsyncClient] = None,
    clock: Callable[[], float] = time.monotonic,
    max_polls: Optional[int] = None,
) -> None:
    """
    Runs in the background, polling GPU KV cache stats and updating the shared variable.

    Inputs:
    - interval_seconds: Sleep duration between polls in seconds.
    - url: Optional override for the Prometheus metrics URL (defaults to env or 8002).
    - client: Optional pre-configured httpx.AsyncClient (e.g. for testing).
    - clock: Injectable monotonic time callable.
    - max_polls: Optional maximum number of iterations before exiting (for testing).
    """
    target_url = url if url is not None else get_gpu_metrics_url()
    poll_count = 0

    if client is not None:
        while max_polls is None or poll_count < max_polls:
            await poll_gpu_metrics_step(client, target_url, clock=clock)
            poll_count += 1
            if max_polls is None or poll_count < max_polls:
                await asyncio.sleep(interval_seconds)
    else:
        async with httpx.AsyncClient() as managed_client:
            while max_polls is None or poll_count < max_polls:
                await poll_gpu_metrics_step(managed_client, target_url, clock=clock)
                poll_count += 1
                if max_polls is None or poll_count < max_polls:
                    await asyncio.sleep(interval_seconds)


def get_gpu_usage_pct() -> float:
    """
    The only way handlers should read the current GPU usage.

    Returns a float between 0.0 and 100.0.
    """
    return _gpu_cache_usage_pct
