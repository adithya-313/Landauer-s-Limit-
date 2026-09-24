"""
admission_controller.py
=======================
PHASE 7a-5 — Admission Controller.

Runs before routing. Makes one of three decisions — accept, reject, or queue —
based on current GPU usage, tier, queue depth, circuit breaker state, and
KV fragmentation ratio. Uses the frozen decision table. Uses the semaphore
for premium queuing.

Decision order (must be preserved):
  1. circuit breaker open → reroute decision (accept with reroute logged)
  2. GPU > 80% and free tier → reject
  3. GPU > 92% and premium → queue via semaphore
  4. KV fragmentation above threshold → trigger_kv_eviction, then accept
  5. queue depth saturated and free tier → reject
  6. default → accept

Routing logic lives in the router, not here.
"""

import asyncio
from typing import Optional

from decision_table import DECISION_TABLE, KV_FRAGMENTATION_THRESHOLD
from gpu_monitor import get_gpu_usage_pct
from protective_actions import (
    reject_request,
    queue_request,
    reroute_request,
    trigger_kv_eviction,
)
from circuit_breaker import get_circuit_breaker

# ---------------------------------------------------------------------------
# SEMAPHORE — caps simultaneous premium requests in the queue.
# Never unbounded — an unbounded buffer is an OOM risk.
# Max 10 queued premium requests at a time.
# ---------------------------------------------------------------------------
PREMIUM_QUEUE_SEMAPHORE = asyncio.Semaphore(10)

# How long to wait for a semaphore slot before rejecting even premium tier.
_PREMIUM_QUEUE_TIMEOUT_SECONDS = 30.0

# Fallback engine when a circuit breaker is open (reroute destination).
_REROUTE_FALLBACK = "llamacpp_local"


async def admit_request(
    request_id: str,
    tier: str,
    engine_id: str,
    queue_depth: int,
    kv_fragmentation: float,
) -> dict:
    """
    The single admission decision function.

    Checks conditions in a fixed priority order (see module docstring) and
    returns one of three outcomes.

    Parameters
    ----------
    request_id : str
        Unique ID for this request (used in log correlation).
    tier : str
        "free" or "premium" — already validated by the Pydantic model upstream.
    engine_id : str
        The target engine adapter name (e.g. "vllm_local").
    queue_depth : int
        Current number of items in the request queue.
    kv_fragmentation : float
        Current KV cache fragmentation ratio (0.0 – 1.0).

    Returns
    -------
    dict
        {
          "decision": "accept" | "reject" | "queue",
          "reason": str,
          "response": dict | None,   # only set on "reject" (429 body)
          "engine_id": str,          # may differ from input if rerouted
        }
    """
    gpu_pct = get_gpu_usage_pct()
    cb = get_circuit_breaker(engine_id)

    # ------------------------------------------------------------------
    # 1. Circuit breaker open → reroute
    # ------------------------------------------------------------------
    if cb.is_open():
        reroute_request(
            from_engine=engine_id,
            to_engine=_REROUTE_FALLBACK,
            reason="Engine degraded — routing to fallback",
        )
        return {
            "decision": "accept",
            "reason": "Engine degraded — routing to fallback",
            "response": None,
            "engine_id": _REROUTE_FALLBACK,
        }

    # ------------------------------------------------------------------
    # 2. GPU > 80% and free tier → reject
    # ------------------------------------------------------------------
    if gpu_pct > 80.0 and tier == "free":
        reason = "High GPU load — shedding low-priority traffic"
        body = reject_request(reason=reason, tier=tier)
        return {
            "decision": "reject",
            "reason": reason,
            "response": body,
            "engine_id": engine_id,
        }

    # ------------------------------------------------------------------
    # 3. GPU > 92% and premium tier → queue via semaphore
    # ------------------------------------------------------------------
    if gpu_pct > 92.0 and tier == "premium":
        reason = "High GPU load — buffering premium request"
        queue_request(reason=reason, tier=tier)
        try:
            await asyncio.wait_for(
                PREMIUM_QUEUE_SEMAPHORE.acquire(),
                timeout=_PREMIUM_QUEUE_TIMEOUT_SECONDS,
            )
            # Semaphore acquired — caller must release after request completes.
        except asyncio.TimeoutError:
            # No slot available within the timeout; reject even premium.
            timeout_reason = "Premium queue timeout — no slot available after 30 s"
            body = reject_request(reason=timeout_reason, tier=tier)
            return {
                "decision": "reject",
                "reason": timeout_reason,
                "response": body,
                "engine_id": engine_id,
            }
        return {
            "decision": "queue",
            "reason": reason,
            "response": None,
            "engine_id": engine_id,
        }

    # ------------------------------------------------------------------
    # 4. KV fragmentation above threshold → trigger eviction, then accept
    # ------------------------------------------------------------------
    if kv_fragmentation > KV_FRAGMENTATION_THRESHOLD:
        trigger_kv_eviction(
            engine_id=engine_id,
            fragmentation_ratio=kv_fragmentation,
        )
        # Eviction triggered — request still proceeds.

    # ------------------------------------------------------------------
    # 5. Queue depth saturated and free tier → reject
    # ------------------------------------------------------------------
    # Use the RequestQueue.max_size as the saturation threshold.
    # queue_depth is passed in by the caller so we don't import _request_queue.
    from request_queue import RequestQueue
    _MAX_QUEUE_SIZE = RequestQueue.__init__.__defaults__[0] if RequestQueue.__init__.__defaults__ else 50  # type: ignore[attr-defined]
    if queue_depth >= _MAX_QUEUE_SIZE and tier == "free":
        reason = "Queue full — shedding free-tier request"
        body = reject_request(reason=reason, tier=tier)
        return {
            "decision": "reject",
            "reason": reason,
            "response": body,
            "engine_id": engine_id,
        }

    # ------------------------------------------------------------------
    # 6. Default → accept
    # ------------------------------------------------------------------
    return {
        "decision": "accept",
        "reason": "Within limits",
        "response": None,
        "engine_id": engine_id,
    }
