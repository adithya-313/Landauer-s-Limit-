from __future__ import annotations

import asyncio
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


def test_stream_timer_elapsed_ms() -> None:
    """Verifies that elapsed_ms calculates time since start in milliseconds."""
    clock = ScriptedClock([10.0, 10.35])
    timer = StreamTimer(clock=clock)
    timer.start()
    assert timer.elapsed_ms() == pytest.approx(350.0)


def test_stream_timer_elapsed_ms_before_start_raises() -> None:
    """Verifies that calling elapsed_ms before start raises RuntimeError."""
    timer = StreamTimer()
    with pytest.raises(RuntimeError, match="has not been called"):
        timer.elapsed_ms()


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


def test_metrics_endpoint() -> None:
    """
    Verifies that the /metrics endpoint exposes Prometheus metrics with the expected values
    and does not expose batch_engine's metric in the default registry.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from telemetry import metrics_router, observe_event

    app = FastAPI()
    app.include_router(metrics_router)
    client = TestClient(app)

    # Observe 3 events:
    # 2 ok premium with ttft 0.4s (400.0 ms)
    # 1 error free with ttft 2.0s (2000.0 ms)
    now = datetime.now(timezone.utc)
    ev_ok_1 = TelemetryEvent(
        run_id="run_1",
        timestamp=now,
        engine="vllm_local",
        concurrency=1,
        ttft_ms=400.0,
        itl_ms=20.0,
        tokens_per_sec=50.0,
        cache_hit=False,
        agent_enabled=False,
        tenant_id=None,
        power_draw_w=None,
        request_tier="premium",
        outcome="ok",
        token_count=5,
    )
    ev_ok_2 = TelemetryEvent(
        run_id="run_2",
        timestamp=now,
        engine="vllm_local",
        concurrency=1,
        ttft_ms=400.0,
        itl_ms=20.0,
        tokens_per_sec=50.0,
        cache_hit=False,
        agent_enabled=False,
        tenant_id=None,
        power_draw_w=None,
        request_tier="premium",
        outcome="ok",
        token_count=5,
    )
    ev_err = TelemetryEvent(
        run_id="run_3",
        timestamp=now,
        engine="vllm_local",
        concurrency=1,
        ttft_ms=2000.0,
        itl_ms=None,
        tokens_per_sec=None,
        cache_hit=False,
        agent_enabled=False,
        tenant_id=None,
        power_draw_w=None,
        request_tier="free",
        outcome="error",
        token_count=0,
    )

    observe_event(ev_ok_1)
    observe_event(ev_ok_2)
    observe_event(ev_err)

    response = client.get("/metrics")
    assert response.status_code == 200
    metrics_text = response.text

    # Assert exact expected lines
    assert 'landauer_requests_total{cache="miss",engine="vllm_local",outcome="ok",request_tier="premium"} 2.0' in metrics_text
    assert 'landauer_ttft_seconds_bucket{engine="vllm_local",le="1.0",request_tier="premium"} 2.0' in metrics_text
    assert 'landauer_ttft_seconds_bucket{engine="vllm_local",le="1.0",request_tier="free"} 0.0' in metrics_text
    assert 'landauer_ttft_seconds_bucket{engine="vllm_local",le="3.0",request_tier="free"} 1.0' in metrics_text

    # Assert batch_engine's metric from the default registry does NOT appear
    assert "gpu_cache_usage_pct" not in metrics_text


def test_build_telemetry_event_cache_hit_fields() -> None:
    """Verifies that build_telemetry_event formats cache hit events correctly."""
    from telemetry import build_telemetry_event

    event = build_telemetry_event(
        run_id="run_cache_hit_test",
        engine="vllm_local",
        concurrency=4,
        request_tier="premium",
        agent_enabled=True,
        cache_hit=True,
        ttft_ms=15.5,
        outcome="ok",
    )
    assert event.run_id == "run_cache_hit_test"
    assert event.engine == "vllm_local"
    assert event.concurrency == 4
    assert event.request_tier == "premium"
    assert event.agent_enabled is True
    assert event.cache_hit is True
    assert event.ttft_ms == 15.5
    assert event.itl_ms is None
    assert event.tokens_per_sec is None
    assert event.token_count == 0
    assert event.outcome == "ok"
    assert event.tenant_id is None
    assert event.power_draw_w is None
    assert event.timestamp.tzinfo == timezone.utc


def test_build_telemetry_event_cache_miss_fields_with_timer() -> None:
    """Verifies that build_telemetry_event extracts metrics from a StreamTimer for cache misses."""
    from telemetry import build_telemetry_event

    clock = ScriptedClock([0.0, 0.2, 0.4, 0.6])
    timer = StreamTimer(clock=clock)
    timer.start()
    timer.mark_token()
    timer.mark_token()
    timer.mark_token()

    event = build_telemetry_event(
        run_id="run_stream_test",
        engine="llamacpp_local",
        concurrency=2,
        request_tier="free",
        agent_enabled=False,
        cache_hit=False,
        timer=timer,
        outcome="ok",
    )
    assert event.run_id == "run_stream_test"
    assert event.engine == "llamacpp_local"
    assert event.concurrency == 2
    assert event.request_tier == "free"
    assert event.agent_enabled is False
    assert event.cache_hit is False
    assert event.token_count == 3
    assert event.ttft_ms == pytest.approx(200.0)
    assert event.itl_ms == pytest.approx(200.0)
    assert event.tokens_per_sec == pytest.approx(5.0)
    assert event.outcome == "ok"


def test_build_telemetry_event_error_outcome() -> None:
    """Verifies that error outcome is correctly stored on the TelemetryEvent."""
    from telemetry import build_telemetry_event

    event = build_telemetry_event(
        run_id="run_error_test",
        engine="colab_cloud",
        concurrency=1,
        request_tier="best_effort",
        agent_enabled=False,
        cache_hit=False,
        outcome="error",
    )
    assert event.outcome == "error"
    assert event.engine == "colab_cloud"


def test_record_event_background_dual_write_independence() -> None:
    """Verifies JSONL append completes within 1s even if Postgres write blocks forever."""
    async def _test() -> None:
        import time
        from telemetry import JsonlEventWriter, record_event_background, build_telemetry_event

        class FakeJsonlWriter(JsonlEventWriter):
            def __init__(self) -> None:
                self.appended: list[Any] = []

            async def append_async(self, event: Any) -> bool:
                self.appended.append(event)
                return True

        class BlockingPgWriter:
            def __init__(self) -> None:
                self.hang_event = asyncio.Event()

            async def write(self, event: Any) -> str:
                await self.hang_event.wait()
                return "ok"

        jsonl_writer = FakeJsonlWriter()
        pg_writer = BlockingPgWriter()
        event = build_telemetry_event(
            run_id="run_dual_test",
            engine="vllm_local",
            concurrency=1,
            request_tier="premium",
            agent_enabled=False,
            cache_hit=True,
            ttft_ms=5.0,
        )

        t_start = time.monotonic()
        record_event_background(jsonl_writer, event, pg_writer=pg_writer)

        # Allow the event loop to schedule the tasks
        for _ in range(50):
            if len(jsonl_writer.appended) > 0:
                break
            await asyncio.sleep(0.01)

        elapsed = time.monotonic() - t_start
        assert elapsed < 1.0
        assert len(jsonl_writer.appended) == 1
        assert jsonl_writer.appended[0] is event

        # Clean up blocked task
        pg_writer.hang_event.set()
        await asyncio.sleep(0.02)

    asyncio.run(_test())


def test_record_event_background_pg_exception_handled() -> None:
    """Verifies that an exception in Postgres write does not prevent JSONL write or escape."""
    async def _test() -> None:
        from telemetry import (
            JsonlEventWriter,
            _background_tasks,
            build_telemetry_event,
            record_event_background,
        )

        class FakeJsonlWriter(JsonlEventWriter):
            def __init__(self) -> None:
                self.appended: list[Any] = []

            async def append_async(self, event: Any) -> bool:
                self.appended.append(event)
                return True

        class CrashingPgWriter:
            async def write(self, event: Any) -> str:
                raise RuntimeError("Simulated Postgres writer crash")

        jsonl_writer = FakeJsonlWriter()
        pg_writer = CrashingPgWriter()
        event = build_telemetry_event(
            run_id="run_crash_test",
            engine="vllm_local",
            concurrency=1,
            request_tier="free",
            agent_enabled=False,
            cache_hit=False,
            ttft_ms=10.0,
        )

        record_event_background(jsonl_writer, event, pg_writer=pg_writer)

        # Allow background tasks to run and settle
        await asyncio.sleep(0.05)

        assert len(jsonl_writer.appended) == 1
        assert jsonl_writer.appended[0] is event

        # Verify no tasks in _background_tasks hold unhandled exceptions
        for task in list(_background_tasks):
            if task.done():
                assert task.exception() is None

    asyncio.run(_test())



