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


import httpx
import re

async def poll_gpu_metrics(interval_seconds: float = 0.1) -> None:
    """
    Runs forever in the background. Wakes up every `interval_seconds`,
    reads the latest GPU cache stats, and updates the shared variable.

    A read error is logged to stderr and the poller sleeps before retrying —
    it never propagates an exception that would kill the background task.
    """
    global _gpu_cache_usage_pct
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get("http://localhost:8001/metrics", timeout=2.0)
                response.raise_for_status()
                match = re.search(r'^gpu_cache_usage_pct\s+([\d\.]+)', response.text, re.MULTILINE)
                if match:
                    _gpu_cache_usage_pct = float(match.group(1))
                else:
                    _gpu_cache_usage_pct = 0.0
            except Exception as exc:  # pragma: no cover — defensive belt-and-braces
                _gpu_cache_usage_pct = 0.0
                print(f"[gpu_monitor] poll error: {exc}", file=sys.stderr)
            await asyncio.sleep(interval_seconds)


def get_gpu_usage_pct() -> float:
    """
    The only way handlers should read the current GPU usage.

    Returns a float between 0.0 and 100.0.
    """
    return _gpu_cache_usage_pct
