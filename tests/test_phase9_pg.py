from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import time
from typing import Any, Optional

import asyncpg

from pg_writer import PgEventWriter
from telemetry import TelemetryEvent, metrics_registry


class FakeClock:
    def __init__(self, start_time: float = 100.0) -> None:
        self.time: float = start_time

    def __call__(self) -> float:
        return self.time

    def advance(self, delta_s: float) -> None:
        self.time += delta_s


class FakePool:
    def __init__(self) -> None:
        self.executed_queries: list[tuple[str, tuple[Any, ...]]] = []
        self.closed: bool = False
        self.execute_hang_event: Optional[asyncio.Event] = None
        self.execute_exception: Optional[Exception] = None

    async def execute(self, query: str, *args: Any, timeout: Optional[float] = None) -> str:
        if self.execute_hang_event is not None:
            await self.execute_hang_event.wait()
        if self.execute_exception is not None:
            raise self.execute_exception
        self.executed_queries.append((query, args))
        return "INSERT 0 1"

    async def close(self) -> None:
        self.closed = True


class FakePoolFactory:
    def __init__(
        self,
        pool: Optional[FakePool] = None,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self.pool: FakePool = pool if pool is not None else FakePool()
        self.call_count: int = 0
        self.raise_exc: Optional[Exception] = raise_exc
        self.last_dsn: Optional[str] = None
        self.last_kwargs: dict[str, Any] = {}

    async def __call__(self, dsn: Optional[str] = None, **kwargs: Any) -> FakePool:
        self.call_count += 1
        self.last_dsn = dsn
        self.last_kwargs = kwargs
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.pool


def _make_dummy_event() -> TelemetryEvent:
    return TelemetryEvent(
        run_id="run_pg_001",
        timestamp=datetime.now(timezone.utc),
        engine="vllm_local",
        concurrency=2,
        ttft_ms=12.5,
        itl_ms=4.2,
        tokens_per_sec=25.0,
        cache_hit=False,
        agent_enabled=True,
        tenant_id="tenant_pg",
        power_draw_w=180.5,
        request_tier="premium",
        outcome="ok",
        token_count=10,
    )


def test_pg_writer_disabled() -> None:
    """Verifies that disabled mode returns 'disabled' without calling pool factory."""
    async def _test() -> None:
        clock = FakeClock()
        factory = FakePoolFactory()
        event = _make_dummy_event()

        # Case 1: None DSN
        writer_none = PgEventWriter(dsn=None, pool_factory=factory, clock=clock)
        await writer_none.start()
        assert factory.call_count == 0
        res_none = await writer_none.write(event)
        assert res_none == "disabled"
        assert factory.call_count == 0

        # Case 2: Empty string DSN
        writer_empty = PgEventWriter(dsn="", pool_factory=factory, clock=clock)
        await writer_empty.start()
        res_empty = await writer_empty.write(event)
        assert res_empty == "disabled"
        assert factory.call_count == 0

    asyncio.run(_test())


def test_pg_writer_ok_path() -> None:
    """Verifies that normal write executes INSERT with 11 parameters in schema order."""
    async def _test() -> None:
        clock = FakeClock()
        fake_pool = FakePool()
        factory = FakePoolFactory(pool=fake_pool)
        event = _make_dummy_event()

        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
        )
        await writer.start()
        assert factory.call_count == 1
        assert factory.last_kwargs.get("min_size") == 1
        assert factory.last_kwargs.get("max_size") == 4

        res = await writer.write(event)
        assert res == "ok"
        assert len(fake_pool.executed_queries) == 1

        query, args = fake_pool.executed_queries[0]
        assert "INSERT INTO telemetry_events" in query
        assert len(args) == 11
        assert args == (
            event.run_id,
            event.timestamp,
            event.engine,
            event.concurrency,
            event.ttft_ms,
            event.itl_ms,
            event.tokens_per_sec,
            event.cache_hit,
            event.agent_enabled,
            event.tenant_id,
            event.power_draw_w,
        )

        await writer.close()
        assert fake_pool.closed is True
        assert writer._pool is None

    asyncio.run(_test())


def test_pg_writer_execute_timeout() -> None:
    """Verifies that execute timeout returns 'error' within 1 second."""
    async def _test() -> None:
        clock = FakeClock()
        fake_pool = FakePool()
        fake_pool.execute_hang_event = asyncio.Event()  # never set -> hangs
        factory = FakePoolFactory(pool=fake_pool)
        event = _make_dummy_event()

        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
            write_timeout_s=0.05,
        )
        await writer.start()

        t_start = time.monotonic()
        res = await writer.write(event)
        elapsed = time.monotonic() - t_start

        assert res == "error"
        assert elapsed < 1.0

    asyncio.run(_test())


def test_pg_writer_connection_error_and_reconnect() -> None:
    """Verifies connection-class error handling, pool closure, backoff, and recovery."""
    async def _test() -> None:
        clock = FakeClock()
        fake_pool = FakePool()
        factory = FakePoolFactory(pool=fake_pool)
        event = _make_dummy_event()

        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
            reconnect_interval_s=5.0,
        )
        await writer.start()
        assert factory.call_count == 1

        # Simulate connection-class error
        fake_pool.execute_exception = asyncpg.PostgresConnectionError("connection severed")
        res1 = await writer.write(event)
        assert res1 == "error"
        assert writer._pool is None
        assert fake_pool.closed is True

        # Next write before reconnect interval has elapsed -> factory must NOT be called
        clock.advance(2.0)  # elapsed 2.0s < 5.0s
        res2 = await writer.write(event)
        assert res2 == "error"
        assert factory.call_count == 1

        # Advance clock past reconnect interval -> factory must be called and succeed
        clock.advance(4.0)  # total 6.0s > 5.0s
        new_pool = FakePool()
        factory.pool = new_pool
        res3 = await writer.write(event)
        assert res3 == "ok"
        assert factory.call_count == 2
        assert writer._pool is new_pool
        assert len(new_pool.executed_queries) == 1

    asyncio.run(_test())


def test_pg_writer_start_factory_raising() -> None:
    """Verifies that failure in start() logs WARNING and leaves pool None without raising."""
    async def _test() -> None:
        clock = FakeClock()
        factory = FakePoolFactory(raise_exc=ConnectionError("Postgres unreachable"))
        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
        )

        # Must not raise
        await writer.start()
        assert writer._pool is None

    asyncio.run(_test())


def test_pg_writer_max_pending_dropped() -> None:
    """Verifies that exceeding max_pending immediately drops subsequent write requests."""
    async def _test() -> None:
        clock = FakeClock()
        fake_pool = FakePool()
        hang_event = asyncio.Event()
        fake_pool.execute_hang_event = hang_event
        factory = FakePoolFactory(pool=fake_pool)
        event = _make_dummy_event()

        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
            max_pending=1,
        )
        await writer.start()

        # First write will block inside execute
        task1 = asyncio.create_task(writer.write(event))
        await asyncio.sleep(0.01)

        # Second write should be dropped immediately
        res_dropped = await writer.write(event)
        assert res_dropped == "dropped"

        # Unblock first write and await completion
        hang_event.set()
        res1 = await task1
        assert res1 == "ok"

    asyncio.run(_test())


def test_pg_writer_prometheus_counters() -> None:
    """Verifies that landauer_pg_writes_total counters track outcomes accurately."""
    async def _test() -> None:
        clock = FakeClock()
        fake_pool = FakePool()
        factory = FakePoolFactory(pool=fake_pool)
        event = _make_dummy_event()

        writer = PgEventWriter(
            dsn="postgresql://user:pass@localhost:5432/db",
            pool_factory=factory,
            clock=clock,
        )
        await writer.start()

        val_ok_before = metrics_registry.get_sample_value(
            "landauer_pg_writes_total", {"result": "ok"}
        ) or 0.0

        await writer.write(event)

        val_ok_after = metrics_registry.get_sample_value(
            "landauer_pg_writes_total", {"result": "ok"}
        ) or 0.0
        assert val_ok_after == val_ok_before + 1.0

    asyncio.run(_test())

