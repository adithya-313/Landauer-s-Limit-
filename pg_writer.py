from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any, Callable, Optional

import asyncpg

from telemetry import TelemetryEvent, landauer_pg_writes_total

logger = logging.getLogger("pg_writer")

_INSERT_TELEMETRY_SQL = (
    "INSERT INTO telemetry_events ("
    "run_id, timestamp, engine, concurrency, ttft_ms, itl_ms, "
    "tokens_per_sec, cache_hit, agent_enabled, tenant_id, power_draw_w"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)"
)


def _sanitize_error(message: str, dsn: Optional[str]) -> str:
    """
    Strips raw DSN strings and embedded passwords from error messages.

    Inputs:
    - message: Raw error string.
    - dsn: Active DSN string to redact if present.

    Returns:
    - Sanitized error message without credentials.
    """
    if dsn is not None and dsn in message:
        message = message.replace(dsn, "[REDACTED_DSN]")
    return re.sub(r"://([^:@\s]+):([^@\s]+)@", r"://\1:***@", message)


class PgEventWriter:
    """
    Asynchronous event writer that buffers and writes telemetry events to PostgreSQL.
    """

    def __init__(
        self,
        dsn: Optional[str] = None,
        pool_factory: Any = asyncpg.create_pool,
        clock: Callable[[], float] = time.monotonic,
        max_pending: int = 100,
        write_timeout_s: float = 2.0,
        reconnect_interval_s: float = 5.0,
    ) -> None:
        """
        Initializes the PostgreSQL telemetry event writer.

        Inputs:
        - dsn: PostgreSQL connection URI (e.g. postgresql://user:pass@host/db).
          If None or empty, operates in disabled mode.
        - pool_factory: Async factory callable to create connection pools (default asyncpg.create_pool).
        - clock: Monotonic time source callable.
        - max_pending: Maximum concurrent in-flight writes before dropping events.
        - write_timeout_s: Maximum seconds to wait for an execute call before timing out.
        - reconnect_interval_s: Minimum seconds between connection pool re-creation attempts.
        """
        self._enabled: bool = bool(dsn and dsn.strip())
        self._dsn: Optional[str] = dsn.strip() if self._enabled and dsn is not None else None
        self._pool_factory: Any = pool_factory
        self._clock: Callable[[], float] = clock
        self._max_pending: int = max_pending
        self._write_timeout_s: float = write_timeout_s
        self._reconnect_interval_s: float = reconnect_interval_s

        self._pool: Optional[Any] = None
        self._pending: int = 0
        self._last_connect_attempt: float = 0.0

    async def start(self) -> None:
        """
        Initializes the PostgreSQL connection pool if writer is enabled.

        Side Effects:
        - Creates a connection pool and sets self._pool.
        - On failure, logs a WARNING without credentials and leaves self._pool as None.
        """
        if not self._enabled:
            return

        self._last_connect_attempt = self._clock()
        try:
            self._pool = await self._pool_factory(self._dsn, min_size=1, max_size=4)
        except Exception as exc:
            logger.warning(
                "Failed to create Postgres connection pool: %s: %s",
                type(exc).__name__,
                _sanitize_error(str(exc), self._dsn),
            )
            self._pool = None

    async def write(self, event: TelemetryEvent) -> str:
        """
        Writes a single telemetry event to PostgreSQL.

        Inputs:
        - event: TelemetryEvent instance with request metrics.

        Returns:
        - Status string: 'ok', 'error', 'dropped', or 'disabled'.

        Side Effects:
        - Increments landauer_pg_writes_total counter in the Prometheus registry.
        - Never raises exceptions to the caller.
        """
        if not self._enabled:
            result = "disabled"
            landauer_pg_writes_total.labels(result=result).inc()
            return result

        if self._pending >= self._max_pending:
            logger.warning(
                "Postgres event writer queue full (%d pending), dropping event",
                self._pending,
            )
            result = "dropped"
            landauer_pg_writes_total.labels(result=result).inc()
            return result

        self._pending += 1
        result = "error"
        try:
            if self._pool is None:
                now = self._clock()
                if now - self._last_connect_attempt >= self._reconnect_interval_s:
                    self._last_connect_attempt = now
                    try:
                        self._pool = await self._pool_factory(self._dsn, min_size=1, max_size=4)
                    except Exception as exc:
                        logger.warning(
                            "Postgres reconnect failed: %s: %s",
                            type(exc).__name__,
                            _sanitize_error(str(exc), self._dsn),
                        )
                        self._pool = None

                if self._pool is None:
                    result = "error"
                    return result

            args = (
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
            await asyncio.wait_for(
                self._pool.execute(_INSERT_TELEMETRY_SQL, *args),
                timeout=self._write_timeout_s,
            )
            result = "ok"
        except asyncio.TimeoutError:
            logger.warning(
                "Postgres write timed out after %.2f seconds",
                self._write_timeout_s,
            )
            result = "error"
        except (
            asyncpg.PostgresConnectionError,
            ConnectionError,
            OSError,
            asyncpg.InterfaceError,
        ) as exc:
            logger.warning(
                "Postgres connection error during write: %s: %s",
                type(exc).__name__,
                _sanitize_error(str(exc), self._dsn),
            )
            self._last_connect_attempt = self._clock()
            if self._pool is not None:
                pool_to_close = self._pool
                self._pool = None
                try:
                    close_coro = pool_to_close.close()
                    if inspect.isawaitable(close_coro):
                        await close_coro
                except Exception as close_exc:
                    logger.warning(
                        "Error closing dropped Postgres pool: %s: %s",
                        type(close_exc).__name__,
                        _sanitize_error(str(close_exc), self._dsn),
                    )
            result = "error"
        except asyncpg.PostgresError as exc:
            logger.warning(
                "Postgres error during write: %s: %s",
                type(exc).__name__,
                _sanitize_error(str(exc), self._dsn),
            )
            result = "error"
        except Exception as exc:
            logger.warning(
                "Unexpected error during Postgres write: %s: %s",
                type(exc).__name__,
                _sanitize_error(str(exc), self._dsn),
            )
            result = "error"
        finally:
            self._pending -= 1
            landauer_pg_writes_total.labels(result=result).inc()

        return result

    async def close(self) -> None:
        """
        Closes the active connection pool if present.
        """
        if self._pool is not None:
            pool_to_close = self._pool
            self._pool = None
            try:
                close_coro = pool_to_close.close()
                if inspect.isawaitable(close_coro):
                    await close_coro
            except Exception as exc:
                logger.warning(
                    "Error closing Postgres pool: %s: %s",
                    type(exc).__name__,
                    _sanitize_error(str(exc), self._dsn),
                )
