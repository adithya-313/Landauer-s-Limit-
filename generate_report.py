"""
generate_report.py
==================
Phase 8a — Comparison Dashboard Report Generator.

Reads all telemetry_*.jsonl files in the current directory, then writes a
Markdown report to stdout.  Run with:

    python generate_report.py > report.md

Dashboards implemented here:
  - Dashboard 1: Compute Engine & Quantization Matrix
  - Dashboard 4: SLA / Error Budget (premium tier only)
  - Appendix A1 + A2: KV fragmentation and batch-size time-series from docs/proof

Dashboards 2 (Cache) and 3 (Drift) come in later commits on this branch.

No external libraries required — stdlib only: json, os, glob, statistics,
datetime, pathlib.
"""

import glob
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# STEP 1 — Load telemetry
# ---------------------------------------------------------------------------

def load_telemetry(directory):
    """
    Finds every telemetry_*.jsonl file in `directory`, reads each line as a
    JSON record, and keeps only records where run_metadata.agent_enabled is
    False (the pure-infrastructure baseline; agent-assisted records come in a
    later phase).

    Returns a flat list of record dicts.
    Also prints a short summary so you can see what was loaded.
    """
    pattern = os.path.join(directory, "telemetry_*.jsonl")
    files = sorted(glob.glob(pattern))

    all_records = []
    skipped = 0

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                meta = record.get("run_metadata", {})
                # Keep only the agent_enabled=False baseline runs.
                if meta.get("agent_enabled") is False:
                    all_records.append(record)
                else:
                    skipped += 1

    print(
        f"[load_telemetry] {len(files)} files found. "
        f"{len(all_records)} records kept (agent_enabled=False). "
        f"{skipped} records skipped (agent_enabled=True or bad JSON).",
        file=sys.stderr,
        flush=True,
    )
    return all_records


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _p95(values):
    """
    Returns the 95th-percentile value from a list of numbers.
    Uses nearest-rank method: sorts the list and picks index int(N * 0.95).
    Returns None for an empty list.
    """
    if not values:
        return None
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * 0.95)
    # Clamp to last valid index in case of rounding.
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


def _mean(values):
    """Returns the arithmetic mean, or None for an empty list."""
    if not values:
        return None
    return statistics.mean(values)


# ---------------------------------------------------------------------------
# STEP 2 — Dashboard 1: Compute Engine & Quantization Matrix
# ---------------------------------------------------------------------------

def compute_dashboard_1(records):
    """
    Groups records by (engine, quantization, concurrency) and computes:
      - mean_ttft        : mean of client_ttft_seconds (200-status records only)
      - p95_ttft         : 95th percentile of client_ttft_seconds
      - mean_latency_ms  : mean of latency_ms
      - p95_latency_ms   : 95th percentile of latency_ms
      - est_throughput   : mean of (50 / (latency_ms / 1000))
                           50 = max_tokens hardcoded in orchestrator.py
      - error_rate       : fraction of records with status_code != 200
      - request_count    : total records in the group

    Returns a list of dicts sorted by engine, quantization, concurrency.
    """
    # Collect records into buckets keyed by (engine, quant, concurrency).
    buckets = {}
    for rec in records:
        meta = rec.get("run_metadata", {})
        key = (
            meta.get("engine", "unknown"),
            meta.get("quantization", "unknown"),
            meta.get("concurrency", 0),
        )
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(rec)

    rows = []
    for key in sorted(buckets.keys()):
        engine, quant, concurrency = key
        group = buckets[key]
        request_count = len(group)

        # Split into successful (200) and failed records.
        ok_records = [r for r in group if r.get("status_code") == 200]
        error_count = request_count - len(ok_records)
        error_rate = error_count / request_count if request_count > 0 else 0.0

        if ok_records:
            # client_ttft_seconds is only present on records that received at
            # least one byte — filter further in case a 200 arrived with no body.
            ttft_values = [
                r["client_ttft_seconds"]
                for r in ok_records
                if "client_ttft_seconds" in r
            ]
            latency_values = [r["latency_ms"] for r in ok_records]
            # Throughput estimate: max_tokens / wall-clock seconds.
            throughput_values = [
                50 / (r["latency_ms"] / 1000)
                for r in ok_records
                if r["latency_ms"] > 0
            ]
        else:
            ttft_values = []
            latency_values = []
            throughput_values = []

        rows.append({
            "engine": engine,
            "quantization": quant,
            "concurrency": concurrency,
            "request_count": request_count,
            "mean_ttft": _mean(ttft_values),
            "p95_ttft": _p95(ttft_values),
            "mean_latency_ms": _mean(latency_values),
            "p95_latency_ms": _p95(latency_values),
            "est_throughput_tps": _mean(throughput_values),
            "error_rate": error_rate,
        })

    return rows


# ---------------------------------------------------------------------------
# STEP 3 — Dashboard 4: SLA / Error Budget
# ---------------------------------------------------------------------------

def compute_dashboard_4(records):
    """
    Computes SLO attainment against the Phase 8 spec targets.

    SLO constants:
      PREMIUM_P95_TTFT_SLO = 1.0 s   (all benchmark traffic is premium)
      FREE_P95_TTFT_SLO    = 3.0 s   (cannot be evaluated — no free-tier data)

    Returns a dict with:
      - overall  : overall attainment across all 200-status records
      - by_group : per-(engine, quant, concurrency) breakdown
    """
    # SLO targets — Phase 8 spec.
    PREMIUM_P95_TTFT_SLO = 1.0   # seconds
    FREE_P95_TTFT_SLO = 3.0      # seconds  (kept for documentation only)

    ok_records = [r for r in records if r.get("status_code") == 200]
    all_ttft = [
        r["client_ttft_seconds"]
        for r in ok_records
        if "client_ttft_seconds" in r
    ]

    overall_p95 = _p95(all_ttft)
    slo_met = (overall_p95 is not None and overall_p95 <= PREMIUM_P95_TTFT_SLO)

    # Error budget: fraction of the SLO target consumed.
    # A p95 of 0.5s on a 1.0s target = 50% consumed; a p95 of 2.0s = 100% (capped).
    if overall_p95 is not None:
        error_budget_consumed = min(overall_p95 / PREMIUM_P95_TTFT_SLO, 1.0)
    else:
        error_budget_consumed = None

    overall = {
        "sample_count": len(all_ttft),
        "observed_p95_ttft": overall_p95,
        "slo_target": PREMIUM_P95_TTFT_SLO,
        "slo_met": slo_met,
        "error_budget_total": 1.0,
        "error_budget_consumed": error_budget_consumed,
        "error_budget_remaining": (1.0 - error_budget_consumed) if error_budget_consumed is not None else None,
        "free_tier_note": (
            f"Free-tier SLO (p95 TTFT < {FREE_P95_TTFT_SLO}s) cannot be evaluated "
            "— all benchmark traffic was sent as premium tier."
        ),
    }

    # Per-combination breakdown (reuse the same grouping as Dashboard 1).
    buckets = {}
    for rec in ok_records:
        meta = rec.get("run_metadata", {})
        key = (
            meta.get("engine", "unknown"),
            meta.get("quantization", "unknown"),
            meta.get("concurrency", 0),
        )
        if key not in buckets:
            buckets[key] = []
        if "client_ttft_seconds" in rec:
            buckets[key].append(rec["client_ttft_seconds"])

    by_group = []
    for key in sorted(buckets.keys()):
        engine, quant, concurrency = key
        values = buckets[key]
        p95 = _p95(values)
        by_group.append({
            "engine": engine,
            "quantization": quant,
            "concurrency": concurrency,
            "p95_ttft": p95,
            "slo_target": PREMIUM_P95_TTFT_SLO,
            "slo_met": (p95 is not None and p95 <= PREMIUM_P95_TTFT_SLO),
        })

    return {"overall": overall, "by_group": by_group}


# ---------------------------------------------------------------------------
# STEP 4 — Appendix
# ---------------------------------------------------------------------------

def compute_appendix(proof_dir):
    """
    Reads two proof files from docs/proof/ and builds time-series tables.

    A1: KV Fragmentation Over Time
        Source: stage4_block_reuse_and_fragmentation.jsonl
        Keeps records where event == "kv_cache_stats".

    A2: Batch Size Over Time
        Source: stage3_staggered_realtraffic_batch_events.jsonl
        Keeps records that have a "tokens_processed" field.

    Returns a dict with keys "a1" and "a2", each a list of row dicts.
    """
    # --- A1: KV fragmentation ---
    frag_path = Path(proof_dir) / "stage4_block_reuse_and_fragmentation.jsonl"
    a1_rows = []
    if frag_path.exists():
        with open(frag_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Only keep the kv_cache_stats events.
                if rec.get("event") == "kv_cache_stats":
                    a1_rows.append({
                        "timestamp": rec.get("timestamp"),
                        "allocated_blocks": rec.get("allocated_blocks"),
                        "free_blocks": rec.get("free_blocks"),
                        "fragmentation_ratio": rec.get("fragmentation_ratio"),
                    })
    else:
        print(f"[compute_appendix] WARNING: {frag_path} not found — A1 will be empty.", file=sys.stderr)

    # --- A2: Batch size over time ---
    batch_path = Path(proof_dir) / "stage3_staggered_realtraffic_batch_events.jsonl"
    a2_rows = []
    if batch_path.exists():
        with open(batch_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Batch-step records always have "tokens_processed".
                if "tokens_processed" in rec:
                    a2_rows.append({
                        "timestamp": rec.get("timestamp"),
                        "active_requests": len(rec.get("active_requests", [])),
                        "tokens_processed": rec.get("tokens_processed"),
                    })
    else:
        print(f"[compute_appendix] WARNING: {batch_path} not found — A2 will be empty.", file=sys.stderr)

    return {"a1": a1_rows, "a2": a2_rows}


# ---------------------------------------------------------------------------
# STEP 5 — Format Markdown
# ---------------------------------------------------------------------------

def _fmt(value, fmt_spec=".3f", missing="N/A"):
    """Format a float, returning `missing` if value is None."""
    if value is None:
        return missing
    return format(value, fmt_spec)


def format_markdown(dashboard_1, dashboard_4, appendix):
    """
    Assembles the full Markdown report string from the computed data structures.
    Returns the string (does not print it).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    # ---- Header ----
    lines += [
        "# Phase 8 — Comparison Dashboard Report",
        f"Generated: {now}",
        "",
        "> **Scope:** `agent_enabled=False` records only (baseline, no agent layer).",
        "> Dashboards 2 (Cache) and 3 (Drift) are implemented in a later commit.",
        "",
    ]

    # ---- Dashboard 1 ----
    lines += [
        "## Dashboard 1: Compute Engine & Quantization Matrix",
        "",
        "| Engine | Quant | Concurrency | Requests | Mean TTFT (s) | p95 TTFT (s) |"
        " Mean Latency (ms) | p95 Latency (ms) | Est. Throughput (tok/s) | Error Rate |",
        "|--------|-------|-------------|----------|---------------|--------------|"
        "-------------------|------------------|------------------------|------------|",
    ]
    for row in dashboard_1:
        lines.append(
            f"| {row['engine']} "
            f"| {row['quantization']} "
            f"| {row['concurrency']} "
            f"| {row['request_count']} "
            f"| {_fmt(row['mean_ttft'])} "
            f"| {_fmt(row['p95_ttft'])} "
            f"| {_fmt(row['mean_latency_ms'], '.1f')} "
            f"| {_fmt(row['p95_latency_ms'], '.1f')} "
            f"| {_fmt(row['est_throughput_tps'], '.1f')} "
            f"| {_fmt(row['error_rate'], '.1%')} |"
        )

    lines += [
        "",
        "> **Note:** ITL (inter-token latency) is not recorded in telemetry — the orchestrator"
        " only captures wall-clock TTFT (first byte) and total latency.",
        "> **Note:** Throughput is estimated as `max_tokens=50 / latency_seconds`."
        " This assumes every request generated exactly 50 tokens, which may not hold for"
        " requests that hit the EOS token early or errored.",
        "> **Note:** Cache hit rate per combination is not available in telemetry records"
        " (reported separately in Dashboard 2).",
        "",
    ]

    # ---- Dashboard 4 ----
    overall = dashboard_4["overall"]
    by_group = dashboard_4["by_group"]

    lines += [
        "## Dashboard 4: SLA / Error Budget",
        "",
        "### SLO Definitions",
        "",
        "| Tier    | Metric      | Target   |",
        "|---------|-------------|----------|",
        "| Premium | p95 TTFT    | <= 1.0 s |",
        "| Free    | p95 TTFT    | <= 3.0 s |",
        "",
        "### Overall Attainment (Premium Tier)",
        "",
    ]

    slo_icon = "MET" if overall["slo_met"] else "MISSED"
    lines += [
        f"- **Sample count (200-status with TTFT):** {overall['sample_count']}",
        f"- **Observed p95 TTFT:** {_fmt(overall['observed_p95_ttft'])} s",
        f"- **SLO target:** {overall['slo_target']} s",
        f"- **SLO status:** {slo_icon}",
        f"- **Error budget consumed:** {_fmt(overall['error_budget_consumed'], '.1%')}",
        f"- **Error budget remaining:** {_fmt(overall['error_budget_remaining'], '.1%')}",
        "",
    ]

    lines += [
        "### Per-Combination Attainment",
        "",
        "| Engine | Quant | Concurrency | p95 TTFT (s) | SLO Target (s) | Met? |",
        "|--------|-------|-------------|--------------|----------------|------|",
    ]
    for row in by_group:
        met = "YES" if row["slo_met"] else "NO"
        lines.append(
            f"| {row['engine']} "
            f"| {row['quantization']} "
            f"| {row['concurrency']} "
            f"| {_fmt(row['p95_ttft'])} "
            f"| {row['slo_target']} "
            f"| {met} |"
        )

    lines += [
        "",
        f"> **Note:** {overall['free_tier_note']}",
        "",
    ]

    # ---- Appendix ----
    lines += [
        "## Appendix",
        "",
        "### A1: KV Fragmentation Over Time (Phase 6c proof data)",
        "",
        "| Timestamp (unix) | Allocated Blocks | Free Blocks | Fragmentation Ratio |",
        "|------------------|------------------|-------------|---------------------|",
    ]
    for row in appendix["a1"]:
        lines.append(
            f"| {_fmt(row['timestamp'], '.2f')} "
            f"| {row['allocated_blocks']} "
            f"| {row['free_blocks']} "
            f"| {_fmt(row['fragmentation_ratio'], '.4f')} |"
        )

    lines += [
        "",
        "### A2: Batch Size Over Time (Phase 6b proof data)",
        "",
        "| Timestamp (unix) | Active Requests | Tokens Processed |",
        "|------------------|-----------------|------------------|",
    ]
    for row in appendix["a2"]:
        lines.append(
            f"| {_fmt(row['timestamp'], '.2f')} "
            f"| {row['active_requests']} "
            f"| {row['tokens_processed']} |"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """Entry point. Loads data, computes dashboards, prints Markdown to stdout."""
    # Force stdout to UTF-8 so redirection on Windows doesn't use cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load all agent_enabled=False telemetry records from the current directory.
    records = load_telemetry(".")

    # Compute each section.
    dashboard_1 = compute_dashboard_1(records)
    dashboard_4 = compute_dashboard_4(records)
    appendix = compute_appendix("docs/proof")

    # Format and print.
    report = format_markdown(dashboard_1, dashboard_4, appendix)
    print(report)


if __name__ == "__main__":
    main()
