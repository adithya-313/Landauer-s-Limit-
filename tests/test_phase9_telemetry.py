from __future__ import annotations

from datetime import datetime, timezone
import fnmatch
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from telemetry import (
    JsonlEventWriter,
    StreamTimer,
    TelemetryEvent,
    default_events_path,
)


class ScriptedClock:
    """Scriptable clock returning a predefined sequence of timestamps."""

    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = list(timestamps)
        self._index = 0

    def __call__(self) -> float:
        if self._index < len(self._timestamps):
            val = self._timestamps[self._index]
            self._index += 1
            return val
        return self._timestamps[-1] if self._timestamps else 0.0


def test_stream_timer_multi_token() -> None:
    """
    Verifies timer calculations with multiple streamed tokens:
    start=0, tokens at 0.5, 0.6, 0.7, 0.8, finish ->
    ttft 500.0 ms, itl 100.0 ms, token_count 4, tokens_per_sec 5.0.
    """
    # 1 call for start (0.0), 4 calls for mark_token (0.5, 0.6, 0.7, 0.8), 1 call for finish (1.0)
    clock = ScriptedClock([0.0, 0.5, 0.6, 0.7, 0.8, 1.0])
    timer = StreamTimer(clock=clock)

    timer.start()
    for _ in range(4):
        timer.mark_token()
    timer.finish()

    assert timer.ttft_ms == pytest.approx(500.0)
    assert timer.itl_ms == pytest.approx(100.0)
    assert timer.token_count == 4
    assert timer.tokens_per_sec == pytest.approx(5.0)


def test_stream_timer_one_token() -> None:
    """Verifies that a single token yields itl_ms=None and computed tokens_per_sec."""
    clock = ScriptedClock([0.0, 0.25, 0.5])
    timer = StreamTimer(clock=clock)

    timer.start()
    timer.mark_token()
    timer.finish()

    assert timer.token_count == 1
    assert timer.ttft_ms == pytest.approx(250.0)
    assert timer.itl_ms is None
    assert timer.tokens_per_sec == pytest.approx(4.0)


def test_stream_timer_zero_tokens() -> None:
    """Verifies that zero tokens yield None for ttft, itl, and tokens_per_sec."""
    clock = ScriptedClock([0.0, 1.0])
    timer = StreamTimer(clock=clock)

    timer.start()
    timer.finish()

    assert timer.token_count == 0
    assert timer.ttft_ms is None
    assert timer.itl_ms is None
    assert timer.tokens_per_sec is None


def _make_dummy_event(run_id: str = "run_1") -> TelemetryEvent:
    return TelemetryEvent(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc),
        engine="vllm_local",
        concurrency=1,
        ttft_ms=120.0,
        itl_ms=25.0,
        tokens_per_sec=40.0,
        cache_hit=False,
        agent_enabled=False,
        tenant_id=None,
        power_draw_w=None,
        request_tier="premium",
        outcome="ok",
        token_count=10,
    )


def test_jsonl_event_writer_appends_exact_keys(tmp_path: Path) -> None:
    """
    Verifies that writing two events produces 2 valid JSON lines whose keys
    match the 11 schema fields + 3 extras exactly.
    """
    expected_keys = {
        "run_id",
        "timestamp",
        "engine",
        "concurrency",
        "ttft_ms",
        "itl_ms",
        "tokens_per_sec",
        "cache_hit",
        "agent_enabled",
        "tenant_id",
        "power_draw_w",
        "request_tier",
        "outcome",
        "token_count",
    }
    assert len(expected_keys) == 14

    log_path = tmp_path / "sub_dir" / "events.jsonl"
    writer = JsonlEventWriter(log_path)

    ev1 = _make_dummy_event("run_a")
    ev2 = _make_dummy_event("run_b")

    assert writer.append(ev1) is True
    assert writer.append(ev2) is True

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    for line in lines:
        data = json.loads(line)
        assert set(data.keys()) == expected_keys


def test_jsonl_event_writer_os_error_handling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verifies that an unwritable path logs WARNING and returns False without raising."""
    log_path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(log_path)
    event = _make_dummy_event()

    def _failing_open(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Permission denied (simulated)")

    monkeypatch.setattr("builtins.open", _failing_open)

    with caplog.at_level(logging.WARNING, logger="telemetry"):
        success = writer.append(event)

    assert success is False
    assert any("Failed to append telemetry event" in rec.message for rec in caplog.records)


def test_default_events_path_not_matching_report_glob() -> None:
    """Verifies that the default events file does NOT match the 'telemetry_*.jsonl' report glob."""
    resolved_path = default_events_path()
    assert not fnmatch.fnmatch(resolved_path.name, "telemetry_*.jsonl")
    assert resolved_path.name == "gateway_events.jsonl"
