from __future__ import annotations

import asyncio
import logging

import httpx
import pytest

from gpu_monitor import (
    get_gpu_metrics_url,
    get_gpu_usage_pct,
    poll_gpu_metrics,
    reset_poller_state,
)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.time: float = start

    def __call__(self) -> float:
        return self.time

    def advance(self, delta: float) -> None:
        self.time += delta


def test_100_consecutive_failures_throttled(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that 100 consecutive poll failures within 59 seconds emit exactly 1 WARNING."""
    async def _test() -> None:
        reset_poller_state()
        caplog.set_level(logging.WARNING, logger="gpu_monitor")

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        transport = httpx.MockTransport(failing_handler)
        clock = FakeClock(0.0)

        async with httpx.AsyncClient(transport=transport) as client:
            # Execute 100 polls while advancing clock within 59 seconds total
            for _ in range(100):
                await poll_gpu_metrics(
                    interval_seconds=0,
                    client=client,
                    clock=clock,
                    max_polls=1,
                )
                clock.advance(0.59)  # 100 * 0.59 = 59.0 seconds

        warnings = [
            r for r in caplog.records
            if r.name == "gpu_monitor" and r.levelname == "WARNING"
        ]
        assert len(warnings) == 1
        assert get_gpu_usage_pct() == 0.0

    asyncio.run(_test())


def test_failures_at_t0_and_t61(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that failures at t=0 and t=61 emit exactly 2 WARNINGs."""
    async def _test() -> None:
        reset_poller_state()
        caplog.set_level(logging.WARNING, logger="gpu_monitor")

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused", request=request)

        transport = httpx.MockTransport(failing_handler)
        clock = FakeClock(0.0)

        async with httpx.AsyncClient(transport=transport) as client:
            # Poll at t = 0.0 -> First failure
            await poll_gpu_metrics(interval_seconds=0, client=client, clock=clock, max_polls=1)

            # Advance to t = 61.0 (> 60s threshold) -> Throttled failure log
            clock.advance(61.0)
            await poll_gpu_metrics(interval_seconds=0, client=client, clock=clock, max_polls=1)

        warnings = [
            r for r in caplog.records
            if r.name == "gpu_monitor" and r.levelname == "WARNING"
        ]
        assert len(warnings) == 2

    asyncio.run(_test())


def test_fail_then_succeed_logs_recovered(caplog: pytest.LogCaptureFixture) -> None:
    """Verifies that recovery from failure logs 1 INFO containing 'recovered'."""
    async def _test() -> None:
        reset_poller_state()
        caplog.set_level(logging.INFO, logger="gpu_monitor")

        calls = 0

        def alternating_handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("Down", request=request)
            return httpx.Response(
                200,
                text="# HELP gpu_cache_usage_pct\ngpu_cache_usage_pct 42.5\n",
            )

        transport = httpx.MockTransport(alternating_handler)
        clock = FakeClock(0.0)

        async with httpx.AsyncClient(transport=transport) as client:
            # Poll 1: Fails
            await poll_gpu_metrics(interval_seconds=0, client=client, clock=clock, max_polls=1)
            # Poll 2: Succeeds
            await poll_gpu_metrics(interval_seconds=0, client=client, clock=clock, max_polls=1)

        info_records = [
            r for r in caplog.records
            if r.name == "gpu_monitor" and r.levelname == "INFO"
        ]
        assert len(info_records) == 1
        assert "recovered" in info_records[0].message
        assert get_gpu_usage_pct() == 42.5

    asyncio.run(_test())


def test_env_var_url_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that LANDAUER_GPU_METRICS_URL environment variable is respected."""
    async def _test() -> None:
        reset_poller_state()
        custom_url = "http://custom-metrics-host:9999/v1/metrics"
        monkeypatch.setenv("LANDAUER_GPU_METRICS_URL", custom_url)

        requested_urls: list[str] = []

        def capture_handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(200, text="gpu_cache_usage_pct 10.0\n")

        transport = httpx.MockTransport(capture_handler)
        async with httpx.AsyncClient(transport=transport) as client:
            await poll_gpu_metrics(interval_seconds=0, client=client, max_polls=1)

        assert len(requested_urls) == 1
        assert requested_urls[0] == custom_url
        assert get_gpu_metrics_url() == custom_url

    asyncio.run(_test())
