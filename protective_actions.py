"""
protective_actions.py
=====================
PHASE 7a-3 — Named Protective Actions.

Each function here does ONE thing and logs it to actions_taken.jsonl.
No action logic should appear in gateway.py or the router.
Phase 12's failure injection tests assert by function name — keep names stable.
"""

import json
import time
from pathlib import Path
from typing import Optional

ACTIONS_LOG = Path("actions_taken.jsonl")


# ---------------------------------------------------------------------------
# LOGGING HELPER
# ---------------------------------------------------------------------------

def _log_action(
    action_name: str,
    reason: str,
    engine_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Appends one line to actions_taken.jsonl.

    Parameters
    ----------
    action_name : str
        The exact function name that fired (used by Phase 12 to assert correctness).
    reason : str
        Why this action was taken (maps to the decision table reason field).
    engine_id : str | None
        Which adapter was involved, or None if not applicable.
    extra : dict | None
        Any additional context: failure_type, tier, fragmentation_ratio, etc.
    """
    record: dict = {
        "timestamp": time.time(),
        "action": action_name,
        "reason": reason,
    }
    if engine_id is not None:
        record["engine_id"] = engine_id
    if extra:
        record.update(extra)

    try:
        with ACTIONS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        # Logging must never crash the main path — swallow and emit to stderr.
        import sys
        print(f"[protective_actions] failed to write log: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# NAMED PROTECTIVE ACTIONS
# ---------------------------------------------------------------------------

def reject_request(reason: str, tier: str) -> dict:
    """
    Returns the 429 response body and logs the rejection.

    Callers: admission controller only.

    Parameters
    ----------
    reason : str
        Human-readable reason for rejection (from decision table).
    tier : str
        The tier of the rejected request ("free" or "premium").

    Returns
    -------
    dict
        The 429 JSON response body to return to the caller.
    """
    _log_action("reject_request", reason, extra={"tier": tier})
    return {
        "error": {
            "message": reason,
            "type": "rate_limited",
            "code": 429,
            "tier": tier,
        }
    }


def queue_request(reason: str, tier: str) -> None:
    """
    Logs that a request is being buffered.

    The actual semaphore acquire happens in the admission controller —
    this function only records the decision for observability.

    Parameters
    ----------
    reason : str
        Why the request is being queued.
    tier : str
        The tier of the queued request.
    """
    _log_action("queue_request", reason, extra={"tier": tier})


def reroute_request(from_engine: str, to_engine: str, reason: str) -> None:
    """
    Logs a reroute decision.

    Callers: circuit breaker handler and Colab failure handler.

    Parameters
    ----------
    from_engine : str
        The engine that was unavailable or degraded.
    to_engine : str
        The engine being routed to instead.
    reason : str
        Why the reroute is happening.
    """
    _log_action(
        "reroute_request",
        reason,
        engine_id=from_engine,
        extra={"from_engine": from_engine, "to_engine": to_engine},
    )


def trigger_kv_eviction(engine_id: str, fragmentation_ratio: float) -> None:
    """
    Logs a KV eviction trigger.

    Callers: the KV cache manager, via the admission controller check.

    Parameters
    ----------
    engine_id : str
        The engine whose KV cache is fragmented.
    fragmentation_ratio : float
        The current fragmentation ratio that crossed the threshold.
    """
    _log_action(
        "trigger_kv_eviction",
        "KV pool fragmented above safe threshold",
        engine_id=engine_id,
        extra={
            "fragmentation_ratio": fragmentation_ratio,
            "failure_type": "kv_pool_exhaustion",
        },
    )


def force_circuit_breaker_open(engine_id: str, reason: str) -> None:
    """
    Logs that the circuit breaker was forced open.

    Used by failure injection only — normal trips happen inside CircuitBreaker
    class via record_failure() which calls this function.

    Parameters
    ----------
    engine_id : str
        The engine whose breaker was tripped.
    reason : str
        The reason the breaker was opened.
    """
    _log_action(
        "force_circuit_breaker_open",
        reason,
        engine_id=engine_id,
        extra={"failure_type": "engine_crash"},
    )
