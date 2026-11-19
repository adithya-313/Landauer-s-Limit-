"""Unit tests for usage metering, quota event log, and per-tenant Prometheus metrics."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pytest

from metering import (
    QuotaEventWriter,
    UsageEvent,
    UsageMeter,
    landauer_quota_refusals_total,
    landauer_tenant_energy_cost_total,
    landauer_tenant_tokens_total,
)
from pg_writer import PgEventWriter


def test_usage_meter_in_memory_totals_and_prometheus_metrics() -> None:
    """Verify UsageMeter updates in-memory accumulators and increments Prometheus counters."""
    meter = UsageMeter(pg_writer=None)

    event1 = UsageEvent(
        run_id="run_1",
        tenant_id="tenant_alpha",
        engine="mock-engine",
        prompt_tokens=25,
        output_tokens=75,
        cache_hit=False,
        energy_cost=0.015,
        currency="USD",
    )
    meter.record(event1)

    totals1 = meter.get_totals("tenant_alpha")
    assert totals1["requests"] == 1
    assert totals1["cache_hits"] == 0
    assert totals1["prompt_tokens"] == 25
    assert totals1["output_tokens"] == 75
    assert totals1["energy_cost"]["USD"] == pytest.approx(0.015)

    # Cache hit event
    event2 = UsageEvent(
        run_id="run_2",
        tenant_id="tenant_alpha",
        engine="mock-engine",
        prompt_tokens=25,
        output_tokens=0,
        cache_hit=True,
        energy_cost=None,
        currency=None,
    )
    meter.record(event2)

    totals2 = meter.get_totals("tenant_alpha")
    assert totals2["requests"] == 2
    assert totals2["cache_hits"] == 1
    assert totals2["prompt_tokens"] == 50
    assert totals2["output_tokens"] == 75
    assert totals2["energy_cost"]["USD"] == pytest.approx(0.015)

    # Verify Prometheus metrics exist in metrics_registry
    prompt_val = landauer_tenant_tokens_total.labels(
        tenant_id="tenant_alpha", kind="prompt"
    )._value.get()
    assert prompt_val >= 50.0

    output_val = landauer_tenant_tokens_total.labels(
        tenant_id="tenant_alpha", kind="output"
    )._value.get()
    assert output_val >= 75.0

    cost_val = landauer_tenant_energy_cost_total.labels(
        tenant_id="tenant_alpha", currency="USD"
    )._value.get()
    assert cost_val >= 0.015


def test_quota_event_writer_logs_valid_jsonl_without_protective_actions() -> None:
    """Verify QuotaEventWriter writes valid JSON line and never touches actions_taken.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        quota_events_file = tmp_path / "quota_events.jsonl"
        actions_file = tmp_path / "actions_taken.jsonl"

        writer = QuotaEventWriter(path=quota_events_file)

        now = datetime(2026, 11, 19, 15, 30, 0, tzinfo=timezone.utc)
        success = writer.record_refusal(
            tenant_id="tenant_beta",
            reason="quota_tokens",
            requested_tokens=250,
            available_tokens=100.0,
            spent_today=1.20,
            spend_cap=5.00,
            tokens_per_minute=500,
            timestamp=now,
        )
        assert success is True
        assert quota_events_file.is_file()

        # Read and parse line
        content = quota_events_file.read_text(encoding="utf-8").strip()
        data = json.loads(content)

        assert data["timestamp"] == now.isoformat()
        assert data["tenant_id"] == "tenant_beta"
        assert data["reason"] == "quota_tokens"
        assert data["requested_tokens"] == 250
        assert data["available_tokens"] == 100.0
        assert data["spent_today"] == 1.20
        assert data["spend_cap"] == 5.00
        assert data["tokens_per_minute"] == 500

        # CRITICAL check: actions_taken.jsonl must NOT exist / be modified
        assert not actions_file.exists()

        # Check Prometheus refusal metric incremented
        refusal_val = landauer_quota_refusals_total.labels(
            tenant_id="tenant_beta", reason="quota_tokens"
        )._value.get()
        assert refusal_val >= 1.0


@pytest.mark.asyncio
async def test_pg_writer_write_usage_with_unset_dsn_does_not_raise() -> None:
    """Verify PgEventWriter.write_usage operates as a safe no-op when DSN is unset."""
    writer = PgEventWriter(dsn=None)
    await writer.start()

    event = UsageEvent(
        run_id="run_test",
        tenant_id="tenant_gamma",
        engine="mock-engine",
        prompt_tokens=10,
        output_tokens=20,
        cache_hit=False,
        energy_cost=0.005,
        currency="USD",
    )

    # Should not raise any error
    await writer.write_usage(event)
    await writer.close()
