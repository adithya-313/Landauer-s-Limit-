"""
decision_table.py
=================
PHASE 7a-2 — Frozen Decision Table.

Every condition and its action lives here and only here.
Phase 12 uses this as the baseline to score the AI triage agent.
DO NOT scatter these rules into handler logic.
DO NOT modify DECISION_TABLE after this commit.
"""

# ---------------------------------------------------------------------------
# THE DECISION TABLE — frozen after Phase 7.
# Every condition and its action lives here and only here.
# Phase 12 uses this as the baseline to score the AI triage agent.
# DO NOT scatter these rules into handler logic.
# ---------------------------------------------------------------------------
DECISION_TABLE = [
    {
        "condition": "gpu_usage > 80 and tier == 'free'",
        "action": "reject",
        "reason": "High GPU load — shedding low-priority traffic",
        "failure_type": None,  # normal load shedding, not a fault
    },
    {
        "condition": "gpu_usage > 92 and tier == 'premium'",
        "action": "queue",
        "reason": "High GPU load — buffering premium request",
        "failure_type": None,
    },
    {
        "condition": "circuit_breaker == 'open'",
        "action": "reroute",
        "reason": "Engine degraded — routing to fallback",
        "failure_type": "engine_crash",
    },
    {
        "condition": "kv_fragmentation > threshold",
        "action": "trigger_kv_eviction",
        "reason": "KV pool fragmented above safe threshold",
        "failure_type": "kv_pool_exhaustion",
    },
    {
        "condition": "colab_session_dead",
        "action": "reroute",
        "reason": "Colab tunnel unreachable — routing to local",
        "failure_type": "colab_session_drop",
    },
    {
        "condition": "queue_depth_saturated and tier == 'free'",
        "action": "reject",
        "reason": "Queue full — shedding free-tier request",
        "failure_type": "queue_saturation",
    },
]

# KV fragmentation threshold — above this, trigger eviction.
KV_FRAGMENTATION_THRESHOLD = 0.15  # 15% fragmentation ratio

# These are the only valid failure_type label strings.
# Use exactly these — no variations — in actions_taken.jsonl and ground_truth.jsonl.
VALID_FAILURE_TYPES = [
    "forced_oom",
    "kv_pool_exhaustion",
    "colab_session_drop",
    "guardrail_timeout",
    "engine_crash",
    "queue_saturation",
]
