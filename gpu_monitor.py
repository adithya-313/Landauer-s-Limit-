"""
gpu_monitor.py
==============
PHASE 7a-1 — GPU Cache Usage Poller.

Background asyncio.Task that tails batch_events.jsonl for kv_cache_stats
events and stores allocated% in one shared float. Request handlers read
this variable — they never write it.

Why a single shared variable is safe here:
  asyncio is single-threaded. One writer (poll_gpu_metrics) + many readers
  (request handlers) never execute concurrently on the same thread, so there
  is no torn-write risk and no lock is needed.
"""

import asyncio
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# SHARED VARIABLE
# ---------------------------------------------------------------------------
# The single shared variable. asyncio is single-threaded, so one writer +
# many readers is safe — no locks needed. Never write this from a request
# handler.
_gpu_cache_usage_pct: float = 0.0

# Path to the telemetry file written by the vLLM engine.
_BATCH_EVENTS_PATH = Path("batch_events.jsonl")


def _read_latest_kv_cache_pct() -> float:
    """
    Tails batch_events.jsonl for the most recent kv_cache_stats entry.

    GPU cache usage % = allocated_blocks / (allocated_blocks + free_blocks) * 100.

    Returns 0.0 if the file does not exist or contains no kv_cache_stats rows.
    Never raises — callers should not crash on a read error.
    """
    if not _BATCH_EVENTS_PATH.exists():
        return 0.0

    last_stat = None
    try:
        with _BATCH_EVENTS_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("event") == "kv_cache_stats":
                    last_stat = obj
    except OSError:
        return 0.0

    if last_stat is None:
        return 0.0

    allocated = last_stat.get("allocated_blocks", 0)
    free = last_stat.get("free_blocks", 0)
    total = allocated + free
    if total == 0:
        return 0.0
    return (allocated / total) * 100.0


async def poll_gpu_metrics(interval_seconds: float = 0.1) -> None:
    """
    Runs forever in the background. Wakes up every `interval_seconds`,
    reads the latest GPU cache stats, and updates the shared variable.

    A read error is logged to stderr and the poller sleeps before retrying —
    it never propagates an exception that would kill the background task.
    """
    global _gpu_cache_usage_pct
    while True:
        try:
            _gpu_cache_usage_pct = _read_latest_kv_cache_pct()
        except Exception as exc:  # pragma: no cover — defensive belt-and-braces
            print(f"[gpu_monitor] poll error: {exc}", file=sys.stderr)
        await asyncio.sleep(interval_seconds)


def get_gpu_usage_pct() -> float:
    """
    The only way handlers should read the current GPU usage.

    Returns a float between 0.0 and 100.0.
    """
    return _gpu_cache_usage_pct
