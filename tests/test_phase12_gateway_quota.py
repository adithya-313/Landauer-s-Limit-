"""Integration tests for gateway quota enforcement, spend caps, and usage metering."""

from __future__ import annotations

import hashlib
import os
from typing import Any
import pytest
from fastapi.testclient import TestClient

import gateway
from telemetry import TelemetryEvent
from tenants import Tenant


@pytest.fixture
def client() -> TestClient:
    """Plain TestClient instance without entering indefinite lifespan tasks."""
    return TestClient(gateway.app)


@pytest.fixture
def test_tenants() -> dict[str, tuple[str, Tenant]]:
    """Configures deterministic test tenants in the gateway's tenant store."""
    keys = {
        "rate": ("key_rate_123", Tenant(
            tenant_id="t_rate_limited",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_rate_123").hexdigest(),
            tokens_per_minute=100,
            cache_mode="isolated",
        )),
        "spend": ("key_spend_123", Tenant(
            tenant_id="t_spend_limited",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_spend_123").hexdigest(),
            spend_cap_per_day=2.00,
            cache_mode="isolated",
        )),
        "shared1": ("key_shared_1", Tenant(
            tenant_id="t_shared_1",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_shared_1").hexdigest(),
            tokens_per_minute=10000,
            cache_mode="shared",
        )),
        "shared2": ("key_shared_2", Tenant(
            tenant_id="t_shared_2",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_shared_2").hexdigest(),
            tokens_per_minute=10000,
            cache_mode="shared",
        )),
        "iso1": ("key_iso_1", Tenant(
            tenant_id="t_iso_1",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_iso_1").hexdigest(),
            tokens_per_minute=10000,
            cache_mode="isolated",
        )),
        "iso2": ("key_iso_2", Tenant(
            tenant_id="t_iso_2",
            tier="premium",
            key_sha256=hashlib.sha256(b"key_iso_2").hexdigest(),
            tokens_per_minute=10000,
            cache_mode="isolated",
        )),
    }

    dev_key = os.environ.get("LANDAUER_API_KEY", "")
    dev_tenant = Tenant(
        tenant_id="local_dev",
        tier="premium",
        key_sha256=hashlib.sha256(dev_key.encode("utf-8")).hexdigest(),
        tokens_per_minute=10000,
    )

    all_tenants = [dev_tenant] + [t[1] for t in keys.values()]
    gateway._tenant_store.replace_tenants(all_tenants)
    return keys


@pytest.fixture
def mock_enqueue(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks queue enqueue to deliver simulated streaming tokens directly into response_queue."""
    async def fake_enqueue(
        engine: str,
        prompt: str,
        tier: str,
        max_tokens: int,
        response_queue: Any,
        messages: Any = None,
    ) -> str:
        # Deliver 3 tokens into the response queue
        await response_queue.put({"type": "token", "content": "token1 "})
        await response_queue.put({"type": "token", "content": "token2 "})
        await response_queue.put({"type": "token", "content": "token3"})
        await response_queue.put({"type": "done"})
        return "req_mock_123"

    monkeypatch.setattr(gateway._request_queue, "enqueue", fake_enqueue)


def test_gateway_token_quota_rejection(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    mock_enqueue: None,
) -> None:
    """Verify 429 with type='quota_exceeded' and reason='quota_tokens' when exceeding tokens_per_minute."""
    raw_key, tenant = test_tenants["rate"]

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "How do I build a search engine in Python?"}],
        "max_tokens": 150,
    }
    headers = {"X-API-Key": raw_key}

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 429
    body = response.json()
    assert "error" in body
    assert body["error"]["type"] == "quota_exceeded"
    assert body["error"]["code"] == 429
    assert body["error"]["reason"] == "quota_tokens"
    assert tenant.tenant_id in body["error"]["message"]


def test_gateway_spend_cap_rejection(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
) -> None:
    """Verify 429 with type='quota_exceeded' and reason='quota_spend' when daily spend cap reached."""
    raw_key, tenant = test_tenants["spend"]

    # Pre-load spend exceeding the 2.00 cap
    gateway._quota.spend_ledger.add_spend(tenant.tenant_id, 2.50)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "max_tokens": 10,
    }
    headers = {"X-API-Key": raw_key}

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 429
    body = response.json()
    assert "error" in body
    assert body["error"]["type"] == "quota_exceeded"
    assert body["error"]["code"] == 429
    assert body["error"]["reason"] == "quota_spend"
    assert "daily spend cap" in body["error"]["message"]


def test_gateway_queue_full_returns_queue_full_not_quota_exceeded(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Phase 7 queue-full failure returns type='queue_full' and does not conflict with quotas."""
    raw_key, tenant = test_tenants["rate"]

    # Force queue to simulate overflow
    async def broken_enqueue(*args: object, **kwargs: object) -> str:
        raise RuntimeError("Queue is full, cannot accept more requests.")

    monkeypatch.setattr(gateway._request_queue, "enqueue", broken_enqueue)

    bucket = gateway._quota._get_or_create_bucket(tenant)
    initial_balance = bucket.balance

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 10,
    }
    headers = {"X-API-Key": raw_key}

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 429
    body = response.json()
    # MUST have type == 'queue_full', NOT 'quota_exceeded'
    assert body.get("type") == "queue_full"
    assert "Queue is full" in body.get("error", "")

    # Reservation should have been refunded on queue failure
    assert bucket.balance == initial_balance


def test_gateway_output_tokens_matches_telemetry_event(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    mock_enqueue: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify UsageMeter output_tokens matches token_count in telemetry event."""
    raw_key, tenant = test_tenants["iso1"]

    captured_events: list[TelemetryEvent] = []
    real_record = gateway.record_event_background

    def fake_record(writer: Any, event: TelemetryEvent, pg_writer: Any = None) -> None:
        captured_events.append(event)
        real_record(writer, event, pg_writer)

    monkeypatch.setattr(gateway, "record_event_background", fake_record)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Count from 1 to 3"}],
        "max_tokens": 50,
    }
    headers = {"X-API-Key": raw_key}

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200

    # Check in-memory meter
    meter_totals = gateway._meter.get_totals(tenant.tenant_id)
    assert meter_totals["output_tokens"] == 3  # Mock delivers 3 tokens

    assert len(captured_events) == 1
    telemetry_event = captured_events[0]
    assert telemetry_event.tenant_id == tenant.tenant_id
    assert telemetry_event.token_count == 3
    assert telemetry_event.token_count == meter_totals["output_tokens"]


def test_gateway_cache_hit_free_tokens_and_metered(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    mock_enqueue: None,
) -> None:
    """Verify cache HIT does not consume bucket balance but records UsageEvent."""
    raw_key, tenant = test_tenants["iso1"]

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is Python cache?"}],
        "max_tokens": 50,
    }
    headers = {"X-API-Key": raw_key}

    # Request 1: Cache MISS -> generates and caches
    r1 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r1.status_code == 200
    assert r1.headers.get("X-Cache") == "MISS"

    bucket = gateway._quota._get_or_create_bucket(tenant)
    balance_after_r1 = bucket.balance

    # Request 2: Cache HIT -> returns cached response
    r2 = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert r2.status_code == 200
    assert r2.headers.get("X-Cache") == "HIT"

    # Bucket balance must not be debited by the cache hit
    assert bucket.balance == balance_after_r1

    # Meter must record cache hit
    totals = gateway._meter.get_totals(tenant.tenant_id)
    assert totals["cache_hits"] >= 1


def test_gateway_tenant_cache_isolation_and_sharing(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    mock_enqueue: None,
) -> None:
    """Verify isolated tenants miss each other's cache while shared tenants hit it."""
    key_iso1, t_iso1 = test_tenants["iso1"]
    key_iso2, t_iso2 = test_tenants["iso2"]

    key_sh1, t_sh1 = test_tenants["shared1"]
    key_sh2, t_sh2 = test_tenants["shared2"]

    # Isolated test: uses entity 'python'
    iso_prompt = [{"role": "user", "content": "How do I install python on ubuntu?"}]
    payload_iso = {"model": "test-model", "messages": iso_prompt}

    r_iso1 = client.post("/v1/chat/completions", json=payload_iso, headers={"X-API-Key": key_iso1})
    assert r_iso1.status_code == 200
    assert r_iso1.headers.get("X-Cache") == "MISS"

    # Tenant 2 asks the same prompt -> MUST MISS due to isolated cache
    r_iso2 = client.post("/v1/chat/completions", json=payload_iso, headers={"X-API-Key": key_iso2})
    assert r_iso2.status_code == 200
    assert r_iso2.headers.get("X-Cache") == "MISS"

    # Shared test: uses entity 'docker'
    shared_prompt = [{"role": "user", "content": "How to deploy docker container in production?"}]
    payload_sh = {"model": "test-model", "messages": shared_prompt}

    r_sh1 = client.post("/v1/chat/completions", json=payload_sh, headers={"X-API-Key": key_sh1})
    assert r_sh1.status_code == 200
    assert r_sh1.headers.get("X-Cache") == "MISS"

    # Shared Tenant 2 asks the same prompt -> MUST HIT shared cache
    r_sh2 = client.post("/v1/chat/completions", json=payload_sh, headers={"X-API-Key": key_sh2})
    assert r_sh2.status_code == 200
    assert r_sh2.headers.get("X-Cache") == "HIT"


def test_gateway_bearer_and_x_api_key_meter_to_same_tenant(
    client: TestClient,
    test_tenants: dict[str, tuple[str, Tenant]],
    mock_enqueue: None,
) -> None:
    """Verify Bearer and X-API-Key headers identify and meter to the same tenant."""
    raw_key, tenant = test_tenants["iso1"]

    payload1 = {"model": "test-model", "messages": [{"role": "user", "content": "Query via X-Key"}]}
    r1 = client.post("/v1/chat/completions", json=payload1, headers={"X-API-Key": raw_key})
    assert r1.status_code == 200

    payload2 = {"model": "test-model", "messages": [{"role": "user", "content": "Query via Bearer"}]}
    r2 = client.post("/v1/chat/completions", json=payload2, headers={"Authorization": f"Bearer {raw_key}"})
    assert r2.status_code == 200

    totals = gateway._meter.get_totals(tenant.tenant_id)
    assert totals["requests"] >= 2
