"""Tests for generate_report.py error counting and Dashboard 3 loading."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_report as gr


def _rec(status=200, ttft=0.1, error=None, latency=1000.0):
    rec = {
        "timestamp": 0.0,
        "latency_ms": latency,
        "status_code": status,
        "run_metadata": {
            "agent_enabled": False,
            "engine": "e",
            "quantization": "q",
            "concurrency": 1,
        },
    }
    if ttft is not None:
        rec["client_ttft_seconds"] = ttft
    if error is not None:
        rec["error"] = error
    return rec


def test_status_200_with_error_field_counts_as_error():
    assert gr._is_error(_rec(error="ReadTimeout")) is True
    assert gr._is_error(_rec(status=500)) is True
    assert gr._is_error(_rec()) is False
    assert gr._is_error(_rec(error="")) is False


def test_dashboard_1_counts_streaming_timeout_and_skips_its_ttft():
    records = [
        _rec(ttft=0.1),
        _rec(ttft=0.2),
        _rec(ttft=0.3),
        _rec(ttft=99.0, error="ReadTimeout"),
    ]
    row = gr.compute_dashboard_1(records)[0]
    assert row["request_count"] == 4
    assert row["error_rate"] == 0.25
    assert row["p95_ttft"] == 0.3


def test_dashboard_4_ignores_timed_out_streaming_records():
    records = [_rec(ttft=0.1), _rec(ttft=0.2), _rec(ttft=99.0, error="ReadTimeout")]
    d4 = gr.compute_dashboard_4(records)
    assert d4["overall"]["sample_count"] == 2
    assert d4["overall"]["observed_p95_ttft"] == 0.2


def test_dashboard_3_stats_and_lowest_prompts(tmp_path):
    sims = [0.95, 0.90, 0.70, 0.50, 0.99]
    path = tmp_path / "drift.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(sims):
            fh.write(json.dumps({
                "prompt_id": f"p{i}", "prompt": f"prompt {i}", "engine": "eng",
                "quantization": "q4", "reference_output": "r", "candidate_output": "c",
                "cosine_similarity": s, "timestamp": 0.0,
            }) + "\n")
    d3 = gr.compute_dashboard_3(str(path))
    row = d3["rows"][0]
    assert row["prompt_count"] == 5
    assert row["min_cosine"] == 0.50
    assert row["below_threshold"] == 2
    assert [low["prompt_id"] for low in row["lowest"]] == ["p3", "p2", "p1"]


def test_dashboard_3_missing_file_returns_none(tmp_path):
    assert gr.compute_dashboard_3(str(tmp_path / "nope.jsonl")) is None


def test_dashboard_3_custom_runtime_marked_invalid(tmp_path):
    # Verify custom_runtime is labeled as invalid due to engine corruption
    path = tmp_path / "drift.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "prompt_id": "p0", "prompt": "prompt 0", "engine": "custom_runtime",
            "quantization": "int4", "reference_output": "r", "candidate_output": "c",
            "cosine_similarity": 0.5, "timestamp": 0.0,
        }) + "\n")
    d3 = gr.compute_dashboard_3(str(path))
    assert d3 is not None
    row = d3["rows"][0]
    assert row["engine"] == "custom_runtime"
    assert "INVALID" in row["display_engine"]


def test_format_markdown_custom_runtime_not_scored(tmp_path):
    # Verify custom_runtime shows 'not scored' in markdown table output
    path = tmp_path / "drift.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "prompt_id": "p0", "prompt": "prompt 0", "engine": "custom_runtime",
            "quantization": "int4", "reference_output": "r", "candidate_output": "c",
            "cosine_similarity": 0.5, "timestamp": 0.0,
        }) + "\n")
    d3 = gr.compute_dashboard_3(str(path))
    d1 = []
    d4 = gr.compute_dashboard_4([])
    app = {"a1": [], "a2": []}
    md = gr.format_markdown(d1, None, None, d4, app, d3)
    assert "not scored" in md


def test_reproduction_deterministic_timestamp():
    # Verify deterministic timestamp is rendered in format_markdown
    d1 = []
    d4 = gr.compute_dashboard_4([])
    app = {"a1": [], "a2": []}
    fixed_ts = "2026-10-01 07:46 UTC"
    md = gr.format_markdown(d1, None, None, d4, app, None, generated_at=fixed_ts)
    assert f"Generated: {fixed_ts}" in md

