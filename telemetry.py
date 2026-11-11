from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Literal, Optional

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("telemetry")


class StreamTimer:
    """
    Measures request latency and streaming throughput piece by piece.

    Notes:
    - N counts streamed pieces (≈ tokens as returned by the engine generator),
      not tokenizer tokens.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """
        Initializes the timer with a monotonic time source.

        Inputs:
        - clock: Callable returning time in fractional seconds.
        """
        self._clock: Callable[[], float] = clock
        self._t_start: Optional[float] = None
        self._t_first: Optional[float] = None
        self._t_last: Optional[float] = None
        self._t_finish: Optional[float] = None
        self._token_count: int = 0

    def start(self) -> None:
        """Records the request arrival timestamp."""
        self._t_start = self._clock()

    def mark_token(self) -> None:
        """
        Records the arrival of a single streamed chunk/piece.

        Side Effects:
        - Sets the first token timestamp if not already recorded.
        - Updates the most recent token timestamp.
        - Increments token_count.
        """
        now = self._clock()
        if self._t_first is None:
            self._t_first = now
        self._t_last = now
        self._token_count += 1

    def finish(self) -> None:
        """Records the completion of the streaming response."""
        self._t_finish = self._clock()

    @property
    def token_count(self) -> int:
        """Number of streamed pieces received."""
        return self._token_count

    @property
    def ttft_ms(self) -> Optional[float]:
        """
        Time to first token in milliseconds.
        Returns None if no streamed piece was received or timer not started.
        """
        if self._t_start is None or self._t_first is None:
            return None
        return (self._t_first - self._t_start) * 1000.0

    @property
    def itl_ms(self) -> Optional[float]:
        """
        Inter-token latency in milliseconds.
        Calculated as (t_last - t_first) / (N - 1) * 1000.
        Returns None if fewer than 2 pieces were received.
        """
        if self._token_count > 1 and self._t_first is not None and self._t_last is not None:
            return ((self._t_last - self._t_first) / (self._token_count - 1)) * 1000.0
        return None

    @property
    def tokens_per_sec(self) -> Optional[float]:
        """
        Streaming throughput in pieces per second.
        Calculated as N / (t_last - t_start).
        Returns None if N == 0 or duration <= 0.
        """
        if self._token_count > 0 and self._t_start is not None and self._t_last is not None:
            duration = self._t_last - self._t_start
            if duration > 0:
                return self._token_count / duration
        return None


@dataclass
class TelemetryEvent:
    """
    Standard event record capturing performance and execution metrics for a request.
    """

    run_id: str
    timestamp: datetime
    engine: str
    concurrency: int
    ttft_ms: Optional[float]
    itl_ms: Optional[float]
    tokens_per_sec: Optional[float]
    cache_hit: bool
    agent_enabled: bool
    tenant_id: Optional[str]
    power_draw_w: Optional[float]
    request_tier: str
    outcome: Literal["ok", "error"]
    token_count: int

    def to_json_dict(self) -> dict[str, object]:
        """
        Serializes the event to a JSON-compatible dictionary.
        Formats timestamp as an ISO-8601 string.
        """
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "engine": self.engine,
            "concurrency": self.concurrency,
            "ttft_ms": self.ttft_ms,
            "itl_ms": self.itl_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "cache_hit": self.cache_hit,
            "agent_enabled": self.agent_enabled,
            "tenant_id": self.tenant_id,
            "power_draw_w": self.power_draw_w,
            "request_tier": self.request_tier,
            "outcome": self.outcome,
            "token_count": self.token_count,
        }


def default_events_path() -> Path:
    """
    Resolves the destination path for JSONL telemetry events.
    Honors LANDAUER_EVENTS_PATH or defaults to <repo_root>/telemetry/gateway_events.jsonl.
    """
    env_path = os.getenv("LANDAUER_EVENTS_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent
    return repo_root / "telemetry" / "gateway_events.jsonl"


class JsonlEventWriter:
    """
    Appends TelemetryEvent instances to a JSON Lines file safely across threads.
    """

    def __init__(self, path: str | Path) -> None:
        self.path: Path = Path(path)
        self._lock = threading.Lock()

    def append(self, event: TelemetryEvent) -> bool:
        """
        Synchronously appends an event to the JSONL file.

        Inputs:
        - event: TelemetryEvent to write.

        Returns:
        - True if written successfully, False on OSError.
        """
        payload = json.dumps(event.to_json_dict()) + "\n"
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(payload)
                return True
            except OSError as exc:
                logger.warning("Failed to append telemetry event to %s: %s", self.path, exc)
                return False

    async def append_async(self, event: TelemetryEvent) -> bool:
        """
        Asynchronously appends an event by offloading file I/O to a worker thread.
        """
        return await asyncio.to_thread(self.append, event)


# ---------------------------------------------------------------------------
# PROMETHEUS METRICS REGISTRY & COLLECTORS
# ---------------------------------------------------------------------------
metrics_registry = CollectorRegistry()

landauer_requests_total = Counter(
    "landauer_requests_total",
    "Total count of requests processed by engine, tier, outcome, and cache status.",
    ["engine", "request_tier", "outcome", "cache"],
    registry=metrics_registry,
)

landauer_ttft_seconds = Histogram(
    "landauer_ttft_seconds",
    "Time to first token in seconds by engine and request tier.",
    ["engine", "request_tier"],
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=metrics_registry,
)

landauer_stream_tokens_total = Counter(
    "landauer_stream_tokens_total",
    "Total number of streamed pieces emitted by engine.",
    ["engine"],
    registry=metrics_registry,
)

landauer_inflight_requests = Gauge(
    "landauer_inflight_requests",
    "Number of requests currently executing in the gateway.",
    registry=metrics_registry,
)


def _poll_gpu_cache_usage() -> float:
    """
    Lazily fetches the latest polled GPU cache usage percentage from gpu_monitor.
    Falls back to 0.0 on import failure or read error.
    """
    try:
        from gpu_monitor import get_gpu_usage_pct

        return float(get_gpu_usage_pct())
    except Exception as exc:
        logger.warning("Could not read GPU usage pct: %s", exc)
        return 0.0


landauer_gpu_cache_usage_percent = Gauge(
    "landauer_gpu_cache_usage_percent",
    "Current GPU KV-cache usage percentage.",
    registry=metrics_registry,
)
landauer_gpu_cache_usage_percent.set_function(_poll_gpu_cache_usage)


def observe_event(event: TelemetryEvent) -> None:
    """
    Updates Prometheus counters, histograms, and token counts from a completed request event.

    Inputs:
    - event: TelemetryEvent recording request metrics.
    """
    cache_label = "hit" if event.cache_hit else "miss"
    landauer_requests_total.labels(
        engine=event.engine,
        request_tier=event.request_tier,
        outcome=event.outcome,
        cache=cache_label,
    ).inc()

    if event.ttft_ms is not None:
        landauer_ttft_seconds.labels(
            engine=event.engine,
            request_tier=event.request_tier,
        ).observe(event.ttft_ms / 1000.0)

    if event.token_count > 0:
        landauer_stream_tokens_total.labels(
            engine=event.engine,
        ).inc(event.token_count)


metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics")
def metrics_endpoint() -> Response:
    """
    Exposes Prometheus metrics collected in the private registry.
    """
    return Response(
        content=generate_latest(metrics_registry),
        media_type=CONTENT_TYPE_LATEST,
    )


def inc_inflight() -> int:
    """
    Increments the in-flight gauge and returns the current concurrency count.
    """
    landauer_inflight_requests.inc()
    return int(landauer_inflight_requests._value.get())


def dec_inflight() -> None:
    """
    Decrements the in-flight request gauge.
    """
    landauer_inflight_requests.dec()


_background_tasks: set[asyncio.Task[Any]] = set()


def record_event_background(writer: JsonlEventWriter, event: TelemetryEvent) -> None:
    """
    Schedules an asynchronous append of the telemetry event without blocking.
    Holds a strong reference in a module-level set until done to avoid garbage collection.
    """
    task = asyncio.create_task(writer.append_async(event))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


def build_telemetry_event(
    *,
    run_id: str,
    engine: str,
    concurrency: int,
    request_tier: str,
    agent_enabled: bool,
    cache_hit: bool,
    timer: Optional[StreamTimer] = None,
    ttft_ms: Optional[float] = None,
    itl_ms: Optional[float] = None,
    tokens_per_sec: Optional[float] = None,
    token_count: int = 0,
    outcome: Literal["ok", "error"] = "ok",
    tenant_id: Optional[str] = None,
    power_draw_w: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> TelemetryEvent:
    """
    Constructs a validated TelemetryEvent from request parameters and timers.

    Inputs:
    - run_id: Benchmark or experiment run identifier.
    - engine: Requested engine identifier.
    - concurrency: In-flight request count measured at request start.
    - request_tier: Request classification tier (e.g. 'premium', 'free').
    - agent_enabled: Whether client indicated agent execution via header.
    - cache_hit: True if response was served directly from cache.
    - timer: Optional StreamTimer from which latency and throughput are extracted.
    - ttft_ms: Explicit time-to-first-token in ms (overrides timer if given).
    - itl_ms: Explicit inter-token latency in ms (overrides timer if given).
    - tokens_per_sec: Explicit throughput in tokens/sec (overrides timer if given).
    - token_count: Explicit token count (defaults to timer count if timer provided).
    - outcome: Either 'ok' or 'error'.
    - tenant_id: Optional tenant identifier.
    - power_draw_w: Optional power draw in Watts.
    - timestamp: Optional timezone-aware UTC datetime.

    Returns:
    - TelemetryEvent instance with formatted metrics.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    if timer is not None:
        if ttft_ms is None:
            ttft_ms = timer.ttft_ms
        if itl_ms is None:
            itl_ms = timer.itl_ms
        if tokens_per_sec is None:
            tokens_per_sec = timer.tokens_per_sec
        if token_count == 0:
            token_count = timer.token_count

    return TelemetryEvent(
        run_id=run_id,
        timestamp=timestamp,
        engine=engine,
        concurrency=concurrency,
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        tokens_per_sec=tokens_per_sec,
        cache_hit=cache_hit,
        agent_enabled=agent_enabled,
        tenant_id=tenant_id,
        power_draw_w=power_draw_w,
        request_tier=request_tier,
        outcome=outcome,
        token_count=token_count,
    )


