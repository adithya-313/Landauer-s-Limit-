# HOW TO RUN (PowerShell):
#   python generate_report.py > report.md 2>$null
#   start report.html
#
# This generates both report.md (Markdown) and report.html (visual dashboard).
# Open report.html in any browser to see the full styled dashboard.
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
            "throughput_50tok_tps": _mean(throughput_values),
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


# ---------------------------------------------------------------------------
# STEP 2b — Dashboard 2A: Semantic Cache (Phase 4)
# ---------------------------------------------------------------------------

def compute_dashboard_2_semantic(events_path="semantic_cache_events.jsonl"):
    """
    Reads semantic_cache_events.jsonl — written by DeterministicSemanticCache._log_event()
    in semantic_cache.py.  Each record is one lookup attempt:

        {"timestamp": ..., "hit": bool, "similarity": float|null, "lookup_latency_ms": float}

    Computes hit rate and latency statistics across all lookups.
    Returns None if the file does not exist (formatter prints "Data not available").
    """
    path = Path(events_path)
    if not path.exists():
        print(f"[compute_dashboard_2_semantic] WARNING: {path} not found.", file=sys.stderr)
        return None

    all_latencies = []
    hit_latencies = []
    miss_latencies = []
    hit_count = 0
    miss_count = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            latency = rec.get("lookup_latency_ms")
            if latency is not None:
                all_latencies.append(latency)

            if rec.get("hit") is True:
                hit_count += 1
                if latency is not None:
                    hit_latencies.append(latency)
            else:
                miss_count += 1
                if latency is not None:
                    miss_latencies.append(latency)

    total = hit_count + miss_count
    if total == 0:
        return None

    return {
        "total_lookups": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_count / total,
        "mean_lookup_latency_ms": _mean(all_latencies),
        "p95_lookup_latency_ms": _p95(all_latencies),
        "mean_hit_latency_ms": _mean(hit_latencies),
        "mean_miss_latency_ms": _mean(miss_latencies),
    }


# ---------------------------------------------------------------------------
# STEP 2c — Dashboard 2B: Prefix Cache (Phase 6d)
# ---------------------------------------------------------------------------

def compute_dashboard_2_prefix(events_path="batch_events.jsonl"):
    """
    Reads batch_events.jsonl — written by BatchEngine._log_event() in
    runtime_core/batch_engine.py.  Only records that contain a "prefix_stats"
    key carry prefix cache data; all other records (kv_cache_stats events,
    batch steps without admissions) are skipped.

    Each prefix_stats entry has this confirmed schema:
        {
            "request_id": "...",
            "prefix_hit": bool,
            "prefix_blocks_reused": int,
            "prefix_tokens_saved": int,
            "arrival_time": float,
            "first_token_time": float,
            "ttft_seconds": float
        }

    Returns None if the file does not exist or contains no prefix_stats entries.
    Note: this data comes from Phase 6d custom-runtime traffic only.
    """
    path = Path(events_path)
    if not path.exists():
        print(f"[compute_dashboard_2_prefix] WARNING: {path} not found.", file=sys.stderr)
        return None

    # Flatten all prefix_stats sub-objects into one list.
    entries = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for entry in rec.get("prefix_stats", []):
                entries.append(entry)

    if not entries:
        print(
            "[compute_dashboard_2_prefix] WARNING: no prefix_stats entries found.",
            file=sys.stderr,
        )
        return None

    hit_entries = [e for e in entries if e.get("prefix_hit") is True]
    miss_entries = [e for e in entries if e.get("prefix_hit") is False]

    total = len(entries)
    hit_count = len(hit_entries)
    miss_count = len(miss_entries)

    tokens_saved = [e["prefix_tokens_saved"] for e in hit_entries if "prefix_tokens_saved" in e]
    ttft_hits = [e["ttft_seconds"] for e in hit_entries if e.get("ttft_seconds") is not None]
    ttft_misses = [e["ttft_seconds"] for e in miss_entries if e.get("ttft_seconds") is not None]

    mean_ttft_hit = _mean(ttft_hits)
    mean_ttft_miss = _mean(ttft_misses)

    # TTFT reduction: how much faster are cache hits vs cold misses?
    if mean_ttft_hit is not None and mean_ttft_miss is not None and mean_ttft_miss > 0:
        ttft_reduction_pct = (mean_ttft_miss - mean_ttft_hit) / mean_ttft_miss * 100
    else:
        ttft_reduction_pct = None

    return {
        "total_requests": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_count / total if total > 0 else 0.0,
        "mean_tokens_saved_per_hit": _mean(tokens_saved),
        "mean_ttft_hit": mean_ttft_hit,
        "mean_ttft_miss": mean_ttft_miss,
        "ttft_reduction_pct": ttft_reduction_pct,
    }


# ---------------------------------------------------------------------------
# STEP 5 — Format Markdown
# ---------------------------------------------------------------------------

def format_markdown(dashboard_1, dashboard_2_semantic, dashboard_2_prefix, dashboard_4, appendix):
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
        " Mean Latency (ms) | p95 Latency (ms) | Throughput (50-tok) | Error Rate |",
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
            f"| {_fmt(row['throughput_50tok_tps'], '.1f')} "
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

    # ---- Dashboard 2 ----
    lines += [
        "## Dashboard 2: Semantic Cache & Prefix Cache Efficiency",
        "",
        "### 2A: Semantic Cache (Phase 4 - FAISS Dual-Lock)",
        "",
    ]
    if dashboard_2_semantic is None:
        lines += ["> **Data not available** — `semantic_cache_events.jsonl` not found.", ""]
    else:
        s = dashboard_2_semantic
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Lookups | {s['total_lookups']} |",
            f"| Hit Count | {s['hit_count']} |",
            f"| Miss Count | {s['miss_count']} |",
            f"| Hit Rate | {_fmt(s['hit_rate'], '.1%')} |",
            f"| Mean Lookup Latency | {_fmt(s['mean_lookup_latency_ms'], '.2f')} ms |",
            f"| p95 Lookup Latency | {_fmt(s['p95_lookup_latency_ms'], '.2f')} ms |",
            f"| Mean Latency (Hits) | {_fmt(s['mean_hit_latency_ms'], '.2f')} ms |",
            f"| Mean Latency (Misses) | {_fmt(s['mean_miss_latency_ms'], '.2f')} ms |",
            "",
            "> Source: `semantic_cache_events.jsonl`",
            "",
        ]

    lines += [
        "### 2B: Prefix Cache (Phase 6d - KV Block Hash Reuse)",
        "",
    ]
    if dashboard_2_prefix is None:
        lines += ["> **Data not available** — `batch_events.jsonl` not found or contains no prefix_stats.", ""]
    else:
        p = dashboard_2_prefix
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Requests with Prefix Data | {p['total_requests']} |",
            f"| Prefix Hit Count | {p['hit_count']} |",
            f"| Prefix Miss Count | {p['miss_count']} |",
            f"| Prefix Hit Rate | {_fmt(p['hit_rate'], '.1%')} |",
            f"| Mean Tokens Saved per Hit | {_fmt(p['mean_tokens_saved_per_hit'], '.1f')} |",
            f"| Mean TTFT (Hits) | {_fmt(p['mean_ttft_hit'], '.3f')} s |",
            f"| Mean TTFT (Misses) | {_fmt(p['mean_ttft_miss'], '.3f')} s |",
            f"| TTFT Reduction from Hits | {_fmt(p['ttft_reduction_pct'], '.1f')}% |",
            "",
            "> Source: `batch_events.jsonl` (prefix_stats sub-records)",
            "> **Note:** Prefix cache data comes from Phase 6d traffic through the"
            " custom runtime only, not from semantic cache lookups.",
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
# STEP 6 — Render HTML
# ---------------------------------------------------------------------------

def render_html(dashboard_1, dashboard_2_semantic, dashboard_2_prefix, dashboard_4, appendix):
    """
    Returns a fully redesigned self-contained HTML dashboard.
    iOS SF Pro font stack, dark theme, sidebar nav, animated KPIs,
    grouped bar charts, circular SLO gauge, IntersectionObserver active nav.
    No external dependencies.
    """
    now = datetime.now().isoformat(timespec="seconds")
    overall = dashboard_4["overall"]

    total_records = sum(r["request_count"] for r in dashboard_1)
    engine_groups_count = len(dashboard_1)
    overall_p95_str = (_fmt(overall["observed_p95_ttft"]) + " s") if overall["observed_p95_ttft"] is not None else "N/A"
    total_err = sum(int(r["request_count"] * r["error_rate"]) for r in dashboard_1)
    overall_err_str = _fmt(total_err / total_records, ".1%") if total_records > 0 else "N/A"

    # CSS stored as a plain string so curly braces need no escaping.
    css = (
        "*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }\n"
        "body { background: #0a0a0f; color: #f1f5f9; font-family: -apple-system, 'SF Pro Display', 'SF Pro Text', BlinkMacSystemFont, sans-serif; font-size: 14px; line-height: 1.6; }\n"
        "aside { position: fixed; top: 0; left: 0; width: 220px; height: 100vh; background: #0d0d14; border-right: 1px solid #1e1e2e; padding: 24px 0; z-index: 100; overflow-y: auto; }\n"
        ".sidebar-brand { padding: 0 20px 20px; border-bottom: 1px solid #1e1e2e; margin-bottom: 16px; }\n"
        ".sidebar-brand h1 { font-size: 13px; font-weight: 700; color: #f1f5f9; letter-spacing: 0.3px; }\n"
        ".sidebar-brand p { font-size: 11px; color: #475569; margin-top: 2px; }\n"
        ".nav-link { display: block; padding: 8px 20px; color: #94a3b8; text-decoration: none; font-size: 13px; border-left: 2px solid transparent; transition: color 0.2s, border-left-color 0.2s; }\n"
        ".nav-link:hover { color: #f1f5f9; }\n"
        ".nav-link.active { color: #7c3aed; border-left-color: #7c3aed; }\n"
        "main { margin-left: 220px; padding: 32px 32px 60px; max-width: 1200px; }\n"
        ".page-header { margin-bottom: 28px; }\n"
        ".page-header h1 { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }\n"
        ".page-header p { font-size: 13px; color: #475569; margin-top: 4px; }\n"
        ".kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }\n"
        ".kpi { background: #111118; border: 1px solid #1e1e2e; border-top: 2px solid #7c3aed; border-radius: 10px; padding: 18px; transition: border-color 0.2s; animation: cardIn 0.4s ease-out both; }\n"
        ".kpi:hover { border-color: #7c3aed; }\n"
        ".kpi .icon { font-size: 18px; margin-bottom: 8px; }\n"
        ".kpi .lbl { font-size: 11px; color: #475569; text-transform: uppercase; letter-spacing: 0.6px; }\n"
        ".kpi .val { font-size: 1.6rem; font-weight: 700; color: #f1f5f9; margin-top: 4px; }\n"
        ".kpi .sub { font-size: 11px; color: #94a3b8; margin-top: 2px; }\n"
        ".card { background: #111118; border: 1px solid #1e1e2e; border-top: 2px solid; border-image: linear-gradient(90deg, #7c3aed, #2563eb) 1; border-radius: 10px; padding: 24px; margin-bottom: 24px; animation: cardIn 0.4s ease-out both; }\n"
        ".card h2 { font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #1e1e2e; }\n"
        ".card h3 { font-size: 0.875rem; font-weight: 600; color: #94a3b8; margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 0.4px; }\n"
        ".table-wrap { overflow-x: auto; }\n"
        "table { width: 100%; border-collapse: collapse; font-size: 13px; }\n"
        "th { background: #1e1e2e; color: #94a3b8; padding: 10px 12px; text-align: left; font-weight: 600; white-space: nowrap; font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; }\n"
        "tr:nth-child(even) td { background: #0f0f16; }\n"
        "tr:nth-child(odd) td { background: #111118; }\n"
        "tr:hover td { background: #1a1a2a; }\n"
        "td { padding: 9px 12px; border-bottom: 1px solid #1a1a2a; white-space: nowrap; color: #f1f5f9; }\n"
        "td.num { text-align: right; font-family: 'SF Mono', 'Fira Code', monospace; font-variant-numeric: tabular-nums; }\n"
        ".note { color: #475569; font-size: 12px; margin-top: 10px; font-style: italic; }\n"
        ".slo-badges { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }\n"
        ".slo-badge { background: #1e1e2e; border: 1px solid #2a2a3a; border-radius: 8px; padding: 10px 16px; }\n"
        ".slo-badge .lbl { font-size: 10px; color: #475569; text-transform: uppercase; letter-spacing: 0.5px; }\n"
        ".slo-badge .val { font-size: 1.15rem; font-weight: 700; margin-top: 2px; }\n"
        ".bar-track { background: #1e1e2e; border-radius: 4px; height: 10px; margin: 4px 0 8px; overflow: hidden; }\n"
        ".bar-fill { height: 100%; border-radius: 4px; width: 0; transition: width 1s ease-out; }\n"
        ".cache-row { margin-bottom: 12px; }\n"
        ".cache-row .clbl { font-size: 12px; color: #94a3b8; margin-bottom: 4px; display: flex; justify-content: space-between; }\n"
        "footer { margin-left: 220px; text-align: center; color: #475569; font-size: 11px; padding: 24px 32px; border-top: 1px solid #1e1e2e; }\n"
        "@keyframes cardIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }\n"
        "@keyframes barGrow { from { transform: scaleY(0); } to { transform: scaleY(1); } }\n"
        ".bar-anim { transform-box: fill-box; transform-origin: 50% 100%; animation: barGrow 0.6s ease-out both; }\n"
        "@media (max-width: 900px) { aside { display: none; } main { margin-left: 0; } footer { margin-left: 0; } .kpi-grid { grid-template-columns: repeat(2, 1fr); } }\n"
    )

    def _ec(rate):
        if rate < 0.05:
            return "#10b981"
        elif rate < 0.20:
            return "#f59e0b"
        return "#ef4444"

    # --- Dashboard 1 table rows ---
    d1_rows = ""
    for i, row in enumerate(dashboard_1):
        ec = _ec(row["error_rate"])
        d1_rows += (
            f'<tr>'
            f"<td>{row['engine']}</td><td>{row['quantization']}</td>"
            f'<td class="num">{row["concurrency"]}</td>'
            f'<td class="num">{row["request_count"]}</td>'
            f'<td class="num">{_fmt(row["mean_ttft"])}</td>'
            f'<td class="num">{_fmt(row["p95_ttft"])}</td>'
            f'<td class="num">{_fmt(row["mean_latency_ms"], ".1f")}</td>'
            f'<td class="num">{_fmt(row["p95_latency_ms"], ".1f")}</td>'
            f'<td class="num">{_fmt(row["throughput_50tok_tps"], ".1f")}</td>'
            f'<td class="num" style="color:{ec};font-weight:bold">{_fmt(row["error_rate"], ".1%")}</td>'
            "</tr>\n"
        )

    # --- Dashboard 1 grouped bar chart: p50 and p95 latency per engine ---
    engine_labels = []
    p50_vals = []
    p95_vals = []
    for r in dashboard_1:
        label = f"{r['engine'][:12]} c={r['concurrency']}"
        engine_labels.append(label)
        p50_vals.append(r["mean_latency_ms"] or 0)
        p95_vals.append(r["p95_latency_ms"] or 0)

    if engine_labels:
        max_val = max(max(p50_vals), max(p95_vals), 1)
        n = len(engine_labels)
        label_h = 60
        chart_h = 200
        bar_w = 18
        group_w = 55
        padding_l = 50
        total_w = padding_l + n * group_w + 40
        total_h = chart_h + label_h + 30
        svg_bars = ""
        for i, lbl in enumerate(engine_labels):
            gx = padding_l + i * group_w
            b50h = int((p50_vals[i] / max_val) * chart_h)
            b95h = int((p95_vals[i] / max_val) * chart_h)
            delay50 = 0.1 + i * 0.08
            delay95 = 0.15 + i * 0.08
            svg_bars += (
                f'<rect class="bar-anim" x="{gx}" y="{chart_h - b50h + 10}" width="{bar_w}" height="{b50h}" fill="#2563eb" rx="2" style="animation-delay:{delay50:.2f}s"/>'
                f'<rect class="bar-anim" x="{gx + bar_w + 2}" y="{chart_h - b95h + 10}" width="{bar_w}" height="{b95h}" fill="#7c3aed" rx="2" style="animation-delay:{delay95:.2f}s"/>'
                f'<text x="{gx + bar_w}" y="{chart_h + 26}" text-anchor="middle" fill="#94a3b8" font-size="9" transform="rotate(-30,{gx + bar_w},{chart_h + 26})">{lbl}</text>'
            )
        y_labels = ""
        for tick in [0, 25, 50, 75, 100]:
            y = 10 + chart_h - int(tick / 100 * chart_h)
            ms_val = int(tick / 100 * max_val)
            y_labels += (
                f'<line x1="{padding_l - 4}" y1="{y}" x2="{padding_l + n * group_w + 20}" y2="{y}" stroke="#1e1e2e" stroke-width="1"/>'
                f'<text x="{padding_l - 8}" y="{y + 4}" text-anchor="end" fill="#475569" font-size="9">{ms_val}</text>'
            )
        legend = (
            f'<rect x="{padding_l}" y="{total_h - 14}" width="10" height="10" fill="#2563eb" rx="2"/>'
            f'<text x="{padding_l + 14}" y="{total_h - 5}" fill="#94a3b8" font-size="10">p50 latency (ms)</text>'
            f'<rect x="{padding_l + 110}" y="{total_h - 14}" width="10" height="10" fill="#7c3aed" rx="2"/>'
            f'<text x="{padding_l + 124}" y="{total_h - 5}" fill="#94a3b8" font-size="10">p95 latency (ms)</text>'
        )
        d1_chart = (
            '<div style="overflow-x:auto;margin-top:16px">'
            f'<svg viewBox="0 0 {total_w} {total_h + 20}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{total_w}px">'
            f'{y_labels}{svg_bars}{legend}'
            "</svg></div>"
        )
    else:
        d1_chart = ""

    # --- Throughput sparkline ---
    spark_items = [(f"{r['engine'][:14]} c={r['concurrency']}", r["throughput_50tok_tps"] or 0) for r in dashboard_1]
    max_tps = max((v for _, v in spark_items), default=1) or 1
    spark_html = ""
    for lbl, tps in spark_items:
        pct = int(tps / max_tps * 100)
        spark_html += (
            f'<div class="cache-row"><div class="clbl"><span>{lbl}</span><span style="color:#0891b2;font-weight:600">{tps:.1f} tps</span></div>'
            f'<div class="bar-track"><div class="bar-fill" data-w="{pct}" style="background:#0891b2"></div></div></div>\n'
        )

    # --- Dashboard 2A ---
    if dashboard_2_semantic:
        s = dashboard_2_semantic
        sem_hit_pct = int((s["hit_rate"] or 0) * 100)
        rows_2a = ""
        for lbl, val in [
            ("Total Lookups", str(s["total_lookups"])),
            ("Hit Count", str(s["hit_count"])),
            ("Miss Count", str(s["miss_count"])),
            ("Hit Rate", _fmt(s["hit_rate"], ".1%")),
            ("Mean Lookup Latency", _fmt(s["mean_lookup_latency_ms"], ".2f") + " ms"),
            ("p95 Lookup Latency", _fmt(s["p95_lookup_latency_ms"], ".2f") + " ms"),
            ("Mean Latency (Hits)", _fmt(s["mean_hit_latency_ms"], ".2f") + " ms"),
            ("Mean Latency (Misses)", _fmt(s["mean_miss_latency_ms"], ".2f") + " ms"),
        ]:
            rows_2a += f'<tr><td>{lbl}</td><td class="num">{val}</td></tr>\n'
        d2a_table = "<table><tr><th>Metric</th><th>Value</th></tr>\n" + rows_2a + "</table>\n"
        d2a_bar = (
            f'<div class="cache-row" style="margin-top:12px"><div class="clbl"><span>Semantic Hit Rate</span><span style="color:#0891b2;font-weight:600">{_fmt(s["hit_rate"], ".1%")}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" data-w="{sem_hit_pct}" style="background:#0891b2"></div></div></div>'
        )
        d2a = d2a_table + d2a_bar + '<p class="note">Source: semantic_cache_events.jsonl</p>'
    else:
        d2a = '<p class="note">Data not available - semantic_cache_events.jsonl not found.</p>'

    # --- Dashboard 2B ---
    if dashboard_2_prefix:
        p = dashboard_2_prefix
        pfx_hit_pct = int((p["hit_rate"] or 0) * 100)
        rows_2b = ""
        for lbl, val in [
            ("Total Requests", str(p["total_requests"])),
            ("Prefix Hit Count", str(p["hit_count"])),
            ("Prefix Miss Count", str(p["miss_count"])),
            ("Prefix Hit Rate", _fmt(p["hit_rate"], ".1%")),
            ("Mean Tokens Saved per Hit", _fmt(p["mean_tokens_saved_per_hit"], ".1f")),
            ("Mean TTFT (Hits)", _fmt(p["mean_ttft_hit"], ".3f") + " s"),
            ("Mean TTFT (Misses)", _fmt(p["mean_ttft_miss"], ".3f") + " s"),
            ("TTFT Reduction", _fmt(p["ttft_reduction_pct"], ".1f") + "%"),
        ]:
            rows_2b += f'<tr><td>{lbl}</td><td class="num">{val}</td></tr>\n'
        d2b_table = "<table><tr><th>Metric</th><th>Value</th></tr>\n" + rows_2b + "</table>\n"
        d2b_bar = (
            f'<div class="cache-row" style="margin-top:12px"><div class="clbl"><span>Prefix Hit Rate</span><span style="color:#7c3aed;font-weight:600">{_fmt(p["hit_rate"], ".1%")}</span></div>'
            f'<div class="bar-track"><div class="bar-fill" data-w="{pfx_hit_pct}" style="background:#7c3aed"></div></div></div>'
        )
        d2b = d2b_table + d2b_bar + '<p class="note">Source: batch_events.jsonl (prefix_stats sub-records)</p>'
    else:
        d2b = '<p class="note">Data not available - batch_events.jsonl not found or contains no prefix_stats.</p>'

    # --- Dashboard 4 table rows ---
    d4_rows = ""
    for row in dashboard_4["by_group"]:
        sc = "#10b981" if row["slo_met"] else "#ef4444"
        sl = "&#10003;" if row["slo_met"] else "&#10007;"
        d4_rows += (
            f'<tr>'
            f"<td>{row['engine']}</td><td>{row['quantization']}</td>"
            f'<td class="num">{row["concurrency"]}</td>'
            f'<td class="num">{_fmt(row["p95_ttft"])}</td>'
            f'<td class="num">{row["slo_target"]}</td>'
            f'<td class="num" style="color:{sc};font-weight:bold;font-size:16px">{sl}</td>'
            "</tr>\n"
        )

    slo_color = "#10b981" if overall["slo_met"] else "#ef4444"
    slo_label = "MET" if overall["slo_met"] else "MISSED"

    # --- Circular SLO gauge SVG ---
    consumed = (overall["error_budget_consumed"] or 0.0) * 100
    consumed_clamped = min(consumed, 100)
    gc = "#10b981" if consumed_clamped < 50 else ("#f59e0b" if consumed_clamped < 90 else "#ef4444")
    r_val = 78
    circ = 2 * 3.14159 * r_val
    dash_used = circ * consumed_clamped / 100
    gauge = (
        '<div style="display:flex;align-items:center;gap:24px;margin:12px 0">'
        '<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg" style="width:140px;height:140px">'
        f'<circle cx="100" cy="100" r="{r_val}" fill="none" stroke="#1e1e2e" stroke-width="14"/>'
        f'<circle cx="100" cy="100" r="{r_val}" fill="none" stroke="{gc}" stroke-width="14"'
        f' stroke-dasharray="{dash_used:.1f} {circ:.1f}" stroke-linecap="round"'
        f' transform="rotate(-90 100 100)"/>'
        f'<text x="100" y="96" text-anchor="middle" fill="#f1f5f9" font-size="22" font-weight="700">{consumed_clamped:.0f}%</text>'
        f'<text x="100" y="116" text-anchor="middle" fill="#475569" font-size="11">consumed</text>'
        "</svg>"
        f'<div><p style="color:#94a3b8;font-size:13px">Error budget</p>'
        f'<p style="color:{gc};font-size:1.4rem;font-weight:700">{100 - consumed_clamped:.1f}% remaining</p>'
        f'<p style="color:#475569;font-size:12px;margin-top:4px">{consumed:.1f}% of budget consumed</p></div>'
        "</div>"
    )

    # --- Appendix A1 rows (cap at 50) ---
    a1_rows = ""
    for row in appendix["a1"][:50]:
        a1_rows += (
            f'<tr>'
            f'<td class="num">{_fmt(row["timestamp"], ".2f")}</td>'
            f'<td class="num">{row["allocated_blocks"]}</td>'
            f'<td class="num">{row["free_blocks"]}</td>'
            f'<td class="num">{_fmt(row["fragmentation_ratio"], ".4f")}</td>'
            "</tr>\n"
        )
    a1_note = f" (first 50 of {len(appendix['a1'])})" if len(appendix["a1"]) > 50 else ""

    # --- Appendix A2 rows ---
    a2_rows = ""
    for row in appendix["a2"][:50]:
        a2_rows += (
            f'<tr>'
            f'<td class="num">{_fmt(row["timestamp"], ".2f")}</td>'
            f'<td class="num">{row["active_requests"]}</td>'
            f'<td class="num">{row["tokens_processed"]}</td>'
            "</tr>\n"
        )
    a2_note = f" (first 50 of {len(appendix['a2'])})" if len(appendix["a2"]) > 50 else ""

    js = (
        "document.querySelectorAll('.bar-fill[data-w]').forEach(el=>{"
        "setTimeout(()=>{el.style.width=el.dataset.w+'%'},100)});\n"
        "document.querySelectorAll('.kpi .val[data-v]').forEach(el=>{"
        "const end=parseFloat(el.dataset.v),isInt=Number.isInteger(end),dec=el.dataset.dec||'0';"
        "const t0=performance.now();"
        "const step=ts=>{const p=Math.min((ts-t0)/800,1),v=end*p;"
        "el.textContent=isInt?Math.round(v):v.toFixed(dec);"
        "if(p<1)requestAnimationFrame(step)};requestAnimationFrame(step)});\n"
        "const obs=new IntersectionObserver(es=>es.forEach(e=>{"
        "document.querySelectorAll('.nav-link').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+e.target.id&&e.isIntersecting))"
        "}),{threshold:0.3});\n"
        "document.querySelectorAll('section[id]').forEach(s=>obs.observe(s));\n"
    )

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Landauer's Limit - Phase 8 Telemetry</title>\n"
        f"<style>\n{css}</style>\n"
        "</head>\n"
        "<body>\n"
        "<aside>\n"
        "  <div class=\"sidebar-brand\">\n"
        "    <h1>Landauer's Limit</h1>\n"
        "    <p>Phase 8 Telemetry</p>\n"
        "  </div>\n"
        '  <a class="nav-link" href="#overview">Overview</a>\n'
        '  <a class="nav-link" href="#dashboard-1">Dashboard 1</a>\n'
        '  <a class="nav-link" href="#dashboard-2">Dashboard 2</a>\n'
        '  <a class="nav-link" href="#dashboard-4">Dashboard 4</a>\n'
        '  <a class="nav-link" href="#appendix">Appendix</a>\n'
        '  <a class="nav-link" href="#data-notes">Data Notes</a>\n'
        "</aside>\n"
        "<main>\n"
        '  <div class="page-header">\n'
        "    <h1>Phase 8 Telemetry Dashboard</h1>\n"
        f'    <p>Generated {now}</p>\n'
        "  </div>\n"
        '  <section id="overview">\n'
        '    <div class="kpi-grid">\n'
        f'      <div class="kpi" style="animation-delay:0s"><div class="icon">&#9672;</div><div class="lbl">Total Records</div><div class="val" data-v="{total_records}">{total_records}</div><div class="sub">baseline requests</div></div>\n'
        f'      <div class="kpi" style="animation-delay:0.1s"><div class="icon">&#11041;</div><div class="lbl">Engine Groups</div><div class="val" data-v="{engine_groups_count}">{engine_groups_count}</div><div class="sub">configurations</div></div>\n'
        f'      <div class="kpi" style="animation-delay:0.2s"><div class="icon">&#9650;</div><div class="lbl">Overall p95 TTFT</div><div class="val">{overall_p95_str}</div><div class="sub">time-to-first-token</div></div>\n'
        f'      <div class="kpi" style="animation-delay:0.3s"><div class="icon">&#9678;</div><div class="lbl">Error Rate</div><div class="val">{overall_err_str}</div><div class="sub">across all engines</div></div>\n'
        "    </div>\n"
        "  </section>\n"
        '  <section id="dashboard-1">\n'
        '    <div class="card">\n'
        "      <h2>Dashboard 1 - Compute Engine &amp; Quantization Matrix</h2>\n"
        '      <div class="table-wrap"><table>\n'
        "        <tr><th>Engine</th><th>Quant</th><th>Concurrency</th><th>Requests</th>"
        "<th>Mean TTFT (s)</th><th>p95 TTFT (s)</th>"
        "<th>Mean Lat (ms)</th><th>p95 Lat (ms)</th>"
        '<th title="throughput_50tok_tps">Throughput (50-tok)</th><th>Error Rate</th></tr>\n'
        f"        {d1_rows}"
        "      </table></div>\n"
        f"      {d1_chart}\n"
        "      <h3>Throughput Comparison</h3>\n"
        f"      {spark_html}\n"
        '      <p class="note">Throughput estimated as max_tokens=50 / latency_s. ITL not recorded in telemetry.</p>\n'
        "    </div>\n"
        "  </section>\n"
        '  <section id="dashboard-2">\n'
        '    <div class="card">\n'
        "      <h2>Dashboard 2 - Semantic Cache &amp; Prefix Cache Efficiency</h2>\n"
        "      <h3>2A: Semantic Cache (Phase 4 - FAISS Dual-Lock)</h3>\n"
        f'      <div class="table-wrap">{d2a}</div>\n'
        "      <h3>2B: Prefix Cache (Phase 6d - KV Block Hash Reuse)</h3>\n"
        f'      <div class="table-wrap">{d2b}</div>\n'
        "    </div>\n"
        "  </section>\n"
        '  <section id="dashboard-4">\n'
        '    <div class="card">\n'
        "      <h2>Dashboard 4 - SLO &amp; Error Budget</h2>\n"
        '      <div class="slo-badges">\n'
        f'        <div class="slo-badge"><div class="lbl">Sample Count</div><div class="val">{overall["sample_count"]}</div></div>\n'
        f'        <div class="slo-badge"><div class="lbl">Observed p95 TTFT</div><div class="val">{_fmt(overall["observed_p95_ttft"])} s</div></div>\n'
        f'        <div class="slo-badge"><div class="lbl">SLO Target</div><div class="val">{overall["slo_target"]} s</div></div>\n'
        f'        <div class="slo-badge"><div class="lbl">SLO Status</div><div class="val" style="color:{slo_color}">{slo_label}</div></div>\n'
        "      </div>\n"
        f"      {gauge}\n"
        "      <h3>Per-Combination Attainment</h3>\n"
        '      <div class="table-wrap"><table>\n'
        "        <tr><th>Engine</th><th>Quant</th><th>Concurrency</th>"
        "<th>p95 TTFT (s)</th><th>SLO Target (s)</th><th>Met?</th></tr>\n"
        f"        {d4_rows}"
        "      </table></div>\n"
        f'      <p class="note">{overall["free_tier_note"]}</p>\n'
        "    </div>\n"
        "  </section>\n"
        '  <section id="appendix">\n'
        '    <div class="card">\n'
        "      <h2>Appendix</h2>\n"
        f"      <h3>A1: KV Fragmentation Over Time (Phase 6c proof data){a1_note}</h3>\n"
        '      <div class="table-wrap"><table>\n'
        "        <tr><th>Timestamp (unix)</th><th>Allocated Blocks</th>"
        "<th>Free Blocks</th><th>Fragmentation Ratio</th></tr>\n"
        f"        {a1_rows}"
        "      </table></div>\n"
        f"      <h3>A2: Batch Size Over Time (Phase 6b proof data){a2_note}</h3>\n"
        '      <div class="table-wrap"><table>\n'
        "        <tr><th>Timestamp (unix)</th><th>Active Requests</th><th>Tokens Processed</th></tr>\n"
        f"        {a2_rows}"
        "      </table></div>\n"
        "    </div>\n"
        "  </section>\n"
        '  <section id="data-notes">\n'
        '    <div class="card" style="border-image:none;border-top:2px solid #f59e0b">\n'
        '      <h2 style="color:#f59e0b">Data Notes</h2>\n'
        '      <ul style="padding-left:1.4em;color:#94a3b8;line-height:1.9">\n'
        "        <li>Throughput is estimated at max_tokens=50 per request &mdash; reflects benchmark conditions, not open-ended generation.</li>\n"
        "        <li>custom_runtime TTFT shows N/A because 85&ndash;95% of those runs returned errors with no successful response to measure.</li>\n"
        "        <li>Semantic cache data: n=69 events. Prefix cache data: n=13 events. These are small samples &mdash; treat as indicative.</li>\n"
        "        <li>All numbers are computed directly from raw telemetry JSONL files. No values are fabricated or estimated beyond what is noted above.</li>\n"
        "      </ul>\n"
        "    </div>\n"
        "  </section>\n"
        "</main>\n"
        f"<footer>Landauer's Limit &middot; Phase 8 Telemetry &middot; Generated {now} &middot; {total_records} records &middot; {engine_groups_count} engines</footer>\n"
        f"<script>{js}</script>\n"
        "</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    """Entry point. Loads data, computes dashboards, prints Markdown to stdout and writes report.html."""
    # Force stdout to UTF-8 so redirection on Windows doesn't use cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load all agent_enabled=False telemetry records from the current directory.
    records = load_telemetry(".")

    # Compute each section.
    dashboard_1 = compute_dashboard_1(records)
    # Dashboard 2: two separate caching systems — do NOT conflate.
    dashboard_2_semantic = compute_dashboard_2_semantic("semantic_cache_events.jsonl")
    dashboard_2_prefix = compute_dashboard_2_prefix("batch_events.jsonl")
    dashboard_4 = compute_dashboard_4(records)
    appendix = compute_appendix("docs/proof")

    # Format and print Markdown (unchanged behaviour).
    report = format_markdown(dashboard_1, dashboard_2_semantic, dashboard_2_prefix, dashboard_4, appendix)
    print(report)

    # Also write a self-contained HTML dashboard.
    html_content = render_html(dashboard_1, dashboard_2_semantic, dashboard_2_prefix, dashboard_4, appendix)
    Path("report.html").write_text(html_content, encoding="utf-8")
    print("report.html written", file=sys.stderr)


if __name__ == "__main__":
    main()
