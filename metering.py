"""Usage metering, Prometheus metrics, and passive quota refusal logging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Literal, Optional

from prometheus_client import Counter

from telemetry import metrics_registry

logger = logging.getLogger("metering")

# ---------------------------------------------------------------------------
# PROMETHEUS METRIC COLLECTORS
# ---------------------------------------------------------------------------

landauer_tenant_tokens_total = Counter(
    "landauer_tenant_tokens_total",
    "Total token consumption partitioned by tenant and token kind.",
    ["tenant_id", "kind"],
    registry=metrics_registry,
)

landauer_tenant_energy_cost_total = Counter(
    "landauer_tenant_energy_cost_total",
    "Total electricity cost accumulated per tenant and currency.",
    ["tenant_id", "currency"],
    registry=metrics_registry,
)

landauer_quota_refusals_total = Counter(
    "landauer_quota_refusals_total",
    "Total requests rejected due to quota exhaustion per tenant and refusal reason.",
    ["tenant_id", "reason"],
    registry=metrics_registry,
)


@dataclass
class UsageEvent:
    """Record of token and energy consumption for a single request execution."""

    run_id: str
    tenant_id: str
    engine: str
    prompt_tokens: int
    output_tokens: int
    cache_hit: bool
    energy_cost: Optional[float] = None
    currency: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Ensures created_at defaults to current UTC timestamp."""
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


def default_quota_events_path() -> Path:
    """Returns the filesystem path for the quota events JSON Lines audit log.

    Honors LANDAUER_QUOTA_EVENTS_PATH if set, otherwise defaults to
    <repo root>/telemetry/quota_events.jsonl.
    """
    env_path = os.getenv("LANDAUER_QUOTA_EVENTS_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent
    return repo_root / "telemetry" / "quota_events.jsonl"


class QuotaEventWriter:
    """Thread-safe append-only writer for passive quota refusal audit logs.

    CRITICAL: This writer is a passive audit log and MUST NOT call protective_actions.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        """Initializes the quota event writer with an explicit or default file path."""
        self.path: Path = Path(path) if path is not None else default_quota_events_path()
        self._lock = threading.Lock()

    def record_refusal(
        self,
        tenant_id: str,
        reason: Literal["quota_tokens", "quota_spend"],
        requested_tokens: int,
        available_tokens: float,
        spent_today: float,
        spend_cap: Optional[float],
        tokens_per_minute: Optional[int],
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Records a quota refusal event to the audit log and increments refusal metrics.

        Inputs:
            tenant_id: Tenant identifier.
            reason: Refusal classification ('quota_tokens' or 'quota_spend').
            requested_tokens: Requested total tokens (prompt + max_tokens).
            available_tokens: Token bucket balance at the time of evaluation.
            spent_today: Cumulative daily spend at the time of evaluation.
            spend_cap: Configured daily spend cap (or None).
            tokens_per_minute: Configured token bucket capacity (or None).
            timestamp: Event timestamp (defaults to current UTC time).

        Returns:
            bool: True if written successfully to disk, False on filesystem error.

        Side Effects:
            Increments landauer_quota_refusals_total Prometheus counter.
            Appends one JSON line to quota_events.jsonl under lock.
            Does NOT trigger any protective actions or engine throttling.
        """
        # Increment Prometheus counter immediately
        landauer_quota_refusals_total.labels(tenant_id=tenant_id, reason=reason).inc()

        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        record = {
            "timestamp": ts.isoformat(),
            "tenant_id": tenant_id,
            "reason": reason,
            "requested_tokens": requested_tokens,
            "available_tokens": available_tokens,
            "spent_today": spent_today,
            "spend_cap": spend_cap,
            "tokens_per_minute": tokens_per_minute,
        }
        line = json.dumps(record) + "\n"

        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line)
                return True
            except OSError as exc:
                logger.warning("Failed to append quota refusal event to %s: %s", self.path, exc)
                return False


class UsageMeter:
    """Maintains in-memory tenant usage totals, updates Prometheus metrics, and persists to Postgres."""

    def __init__(self, pg_writer: Optional[Any] = None) -> None:
        """Initializes the usage meter with an optional asynchronous PostgreSQL writer.

        Inputs:
            pg_writer: Optional PgEventWriter instance for database persistence.
        """
        self._pg_writer = pg_writer
        self._lock = threading.Lock()
        self._totals: dict[str, dict[str, Any]] = {}

    def _ensure_tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        """Ensures an in-memory statistics dictionary exists for the tenant."""
        if tenant_id not in self._totals:
            self._totals[tenant_id] = {
                "requests": 0,
                "cache_hits": 0,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "energy_cost": {},
            }
        return self._totals[tenant_id]

    def record(self, event: UsageEvent) -> None:
        """Records a request usage event across memory, Prometheus, and PostgreSQL.

        Inputs:
            event: Validated UsageEvent instance.

        Key Logic:
            1. Updates in-memory tenant accumulators under lock.
            2. Increments Prometheus token counters (kind="prompt" and kind="output").
            3. If energy_cost > 0 and currency is present, increments energy cost counter.
            4. Schedules non-blocking PostgreSQL write task if pg_writer is present and active.

        Side Effects:
            Never raises exceptions to the caller; failures in background persistence are logged.
        """
        # 1. Update in-memory statistics
        with self._lock:
            stats = self._ensure_tenant_stats(event.tenant_id)
            stats["requests"] += 1
            if event.cache_hit:
                stats["cache_hits"] += 1
            stats["prompt_tokens"] += event.prompt_tokens
            stats["output_tokens"] += event.output_tokens
            if event.energy_cost is not None and event.currency is not None and event.energy_cost > 0.0:
                curr_dict = stats["energy_cost"]
                curr_dict[event.currency] = curr_dict.get(event.currency, 0.0) + event.energy_cost

        # 2. Update Prometheus metrics
        if event.prompt_tokens > 0:
            landauer_tenant_tokens_total.labels(
                tenant_id=event.tenant_id, kind="prompt"
            ).inc(event.prompt_tokens)

        if event.output_tokens > 0:
            landauer_tenant_tokens_total.labels(
                tenant_id=event.tenant_id, kind="output"
            ).inc(event.output_tokens)

        if event.energy_cost is not None and event.currency is not None and event.energy_cost > 0.0:
            landauer_tenant_energy_cost_total.labels(
                tenant_id=event.tenant_id, currency=event.currency
            ).inc(event.energy_cost)

        # 3. Schedule non-blocking asynchronous write to PostgreSQL if writer configured
        if self._pg_writer is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._pg_writer.write_usage(event))
            except RuntimeError:
                # No running event loop (e.g. executing in a synchronous test or worker thread)
                pass

    def get_totals(self, tenant_id: str) -> dict[str, Any]:
        """Returns a snapshot of accumulated usage statistics for the given tenant."""
        with self._lock:
            stats = self._ensure_tenant_stats(tenant_id)
            return {
                "requests": stats["requests"],
                "cache_hits": stats["cache_hits"],
                "prompt_tokens": stats["prompt_tokens"],
                "output_tokens": stats["output_tokens"],
                "energy_cost": dict(stats["energy_cost"]),
            }
