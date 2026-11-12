from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
import re
from typing import Any

from fastapi import Request
import pytest
import yaml

from tenants import Tenant, TenantStore, default_tenants_path, load_tenants


def test_tenants_valid_load(tmp_path: Path) -> None:
    """Verifies that a well-formed tenants YAML file loads successfully."""
    config_path = tmp_path / "valid_tenants.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "tenant_a",
                "tier": "free",
                "key_sha256": "a" * 64,
            },
            {
                "tenant_id": "tenant_b",
                "tier": "premium",
                "key_sha256": "b" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    store = load_tenants(config_path)
    assert len(store.tenants) == 2
    assert store.tenants[0].tenant_id == "tenant_a"
    assert store.tenants[0].tier == "free"
    assert store.tenants[1].tenant_id == "tenant_b"
    assert store.tenants[1].tier == "premium"


def test_tenants_missing_file_raises(tmp_path: Path) -> None:
    """Verifies that attempting to load a nonexistent file raises ValueError."""
    nonexistent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(ValueError, match="does not exist"):
        load_tenants(nonexistent)


def test_tenants_duplicate_id_raises(tmp_path: Path) -> None:
    """Verifies that duplicate tenant_id values raise ValueError."""
    config_path = tmp_path / "dup_id.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "same_id",
                "tier": "free",
                "key_sha256": "a" * 64,
            },
            {
                "tenant_id": "same_id",
                "tier": "premium",
                "key_sha256": "b" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate tenant_id 'same_id'"):
        load_tenants(config_path)


def test_tenants_duplicate_hash_raises(tmp_path: Path) -> None:
    """Verifies that duplicate key_sha256 hashes raise ValueError."""
    config_path = tmp_path / "dup_hash.yaml"
    content = {
        "tenants": [
            {
                "tenant_id": "tenant_one",
                "tier": "free",
                "key_sha256": "c" * 64,
            },
            {
                "tenant_id": "tenant_two",
                "tier": "premium",
                "key_sha256": "c" * 64,
            },
        ]
    }
    config_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key_sha256"):
        load_tenants(config_path)


def test_tenants_bad_hash_length_and_uppercase_raises(tmp_path: Path) -> None:
    """Verifies that invalid hash lengths and uppercase hex characters raise ValueError."""
    # Bad length
    bad_len_path = tmp_path / "bad_len.yaml"
    bad_len_path.write_text(
        yaml.safe_dump({"tenants": [{"tenant_id": "t1", "tier": "free", "key_sha256": "1234"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        load_tenants(bad_len_path)

    # Uppercase hex chars
    upper_path = tmp_path / "upper.yaml"
    upper_path.write_text(
        yaml.safe_dump(
            {"tenants": [{"tenant_id": "t2", "tier": "free", "key_sha256": ("A" * 64)}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid"):
        load_tenants(upper_path)


def test_tenant_store_lookup_success_and_failure() -> None:
    """Verifies lookup behavior with valid keys, wrong keys, empty strings, and None."""
    raw_secret = "my_secret_token_12345"
    secret_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()

    tenant_record = Tenant(
        tenant_id="dev_user",
        tier="premium",
        key_sha256=secret_hash,
    )
    store = TenantStore([tenant_record])

    # Valid key -> returns Tenant
    found = store.lookup(raw_secret)
    assert found is not None
    assert found.tenant_id == "dev_user"
    assert found.tier == "premium"

    # Wrong key -> returns None
    assert store.lookup("completely_incorrect_key") is None

    # Empty string and None -> returns None
    assert store.lookup("") is None
    assert store.lookup(None) is None


def test_committed_tenants_yaml_contains_no_plaintext_keys() -> None:
    """
    Verifies that the committed tenants.yaml loads cleanly and contains
    no string longer than 64 characters or non-hash key values.
    """
    tenants_path = default_tenants_path()
    assert tenants_path.is_file(), f"Expected {tenants_path} to exist"

    raw_text = tenants_path.read_text(encoding="utf-8")
    store = load_tenants(tenants_path)
    assert len(store.tenants) >= 1

    # Check each token and string in the file to guarantee no plaintext key leaked
    hex_64_re = re.compile(r"^[a-f0-9]{64}$")
    for tenant in store.tenants:
        assert hex_64_re.match(tenant.key_sha256), f"Invalid hash: {tenant.key_sha256}"

    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # If line has a scalar string value longer than 64 chars, it is suspicious
        parts = line.split(":", 1)
        if len(parts) == 2:
            val = parts[1].strip().strip('"').strip("'")
            if len(val) > 64:
                pytest.fail(f"Found scalar value longer than 64 characters in tenants.yaml: {val}")


def _create_test_auth_client(store: TenantStore) -> Any:
    """Creates a minimal FastAPI app with both middlewares wired in identical order to gateway."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from payload_guard import check_payload_size
    from tenants import make_api_key_middleware

    app = FastAPI()

    # Wire in the identical order as gateway.py:
    # 1. check_payload_size
    # 2. make_api_key_middleware
    # Starlette executes the later-registered middleware FIRST (outermost layer).
    app.middleware("http")(check_payload_size)
    app.middleware("http")(make_api_key_middleware(store))

    @app.get("/health")
    def health_stub() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics_stub() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/engines")
    def admin_engines_stub() -> dict[str, str]:
        return {"engines": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_stub(request: Request) -> dict[str, Any]:
        return {
            "tenant_id": getattr(request.state, "tenant_id", None),
            "tier": getattr(request.state, "tier", None),
        }

    return TestClient(app)


def test_auth_middleware_order_no_key_and_no_content_length_returns_401() -> None:
    """
    Verifies middleware order: when both API key and Content-Length are missing,
    auth middleware executes first and returns 401 Unauthorized (not 411).
    """
    secret = "valid_key_123"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="dev", tier="premium", key_sha256=sha)])
    client = _create_test_auth_client(store)

    # Streaming generator omits Content-Length header
    response = client.post("/v1/chat/completions", content=(b"data" for _ in range(1)))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401


def test_auth_middleware_valid_key_and_no_content_length_returns_411() -> None:
    """Verifies that with a valid API key but missing Content-Length, payload check returns 411."""
    secret = "valid_key_123"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="dev", tier="premium", key_sha256=sha)])
    client = _create_test_auth_client(store)

    response = client.post(
        "/v1/chat/completions",
        content=(b"data" for _ in range(1)),
        headers={"X-API-Key": secret},
    )
    assert response.status_code == 411
    assert response.json()["error"]["code"] == 411


def test_auth_middleware_valid_key_and_oversized_payload_returns_413() -> None:
    """Verifies that with a valid API key but oversized Content-Length, payload check returns 413."""
    secret = "valid_key_123"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="dev", tier="premium", key_sha256=sha)])
    client = _create_test_auth_client(store)

    response = client.post(
        "/v1/chat/completions",
        headers={"X-API-Key": secret, "Content-Length": "70000"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == 413


def test_auth_middleware_missing_key_vs_wrong_key_byte_identical() -> None:
    """
    Verifies that missing API key and invalid API key produce byte-identical
    response bodies, status codes, and identical headers (excluding Date / Content-Length).
    """
    secret = "valid_key_123"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="dev", tier="premium", key_sha256=sha)])
    client = _create_test_auth_client(store)

    resp_missing = client.post("/v1/chat/completions")
    resp_wrong = client.post("/v1/chat/completions", headers={"X-API-Key": "wrong_key"})

    assert resp_missing.status_code == 401
    assert resp_wrong.status_code == 401
    assert resp_missing.content == resp_wrong.content

    # Compare header sets excluding date and content-length
    headers_missing = {
        k.lower(): v
        for k, v in resp_missing.headers.items()
        if k.lower() not in ("date", "content-length")
    }
    headers_wrong = {
        k.lower(): v
        for k, v in resp_wrong.headers.items()
        if k.lower() not in ("date", "content-length")
    }
    assert headers_missing == headers_wrong


def test_auth_middleware_exempt_routes_accessible_without_key() -> None:
    """Verifies that GET /health and GET /metrics are exempt from API key validation."""
    store = TenantStore([])
    client = _create_test_auth_client(store)

    resp_health = client.get("/health")
    assert resp_health.status_code == 200

    resp_metrics = client.get("/metrics")
    assert resp_metrics.status_code == 200


def test_auth_middleware_admin_route_requires_key() -> None:
    """Verifies that admin routes require a valid key and return 401 without it."""
    secret = "admin_key_abc"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="admin_user", tier="premium", key_sha256=sha)])
    client = _create_test_auth_client(store)

    resp_unauth = client.get("/admin/engines")
    assert resp_unauth.status_code == 401

    resp_auth = client.get("/admin/engines", headers={"X-API-Key": secret})
    assert resp_auth.status_code == 200


def test_auth_middleware_populates_request_state_tenant_and_tier() -> None:
    """Verifies that a valid key populates request.state with the tenant's id and tier."""
    secret = "tenant_key_xyz"
    sha = hashlib.sha256(secret.encode("utf-8")).hexdigest()
    store = TenantStore([Tenant(tenant_id="custom_tenant", tier="free", key_sha256=sha)])
    client = _create_test_auth_client(store)

    response = client.post(
        "/v1/chat/completions",
        json={"prompt": "test"},
        headers={"X-API-Key": secret},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["tenant_id"] == "custom_tenant"
    assert data["tier"] == "free"


def test_telemetry_event_preserves_tenant_id() -> None:
    """Verifies that build_telemetry_event preserves and serializes tenant_id in to_json_dict."""
    from telemetry import build_telemetry_event

    event = build_telemetry_event(
        run_id="test_run",
        engine="vllm_local",
        concurrency=1,
        request_tier="premium",
        agent_enabled=False,
        cache_hit=True,
        ttft_ms=10.0,
        tenant_id="tenant_alpha",
    )
    assert event.tenant_id == "tenant_alpha"
    serialized = event.to_json_dict()
    assert serialized["tenant_id"] == "tenant_alpha"


def test_get_tenant_context_success() -> None:
    """Verifies that get_tenant_context returns (tenant_id, tier) when request state is populated."""
    from types import SimpleNamespace
    from tenants import get_tenant_context

    dummy_request = SimpleNamespace(state=SimpleNamespace(tenant_id="tenant_123", tier="premium"))
    tenant_id, tier = get_tenant_context(dummy_request)
    assert tenant_id == "tenant_123"
    assert tier == "premium"


def test_get_tenant_context_missing_raises() -> None:
    """Verifies that get_tenant_context raises RuntimeError when request.state or attributes are missing."""
    from types import SimpleNamespace
    from tenants import get_tenant_context

    expected_msg = "tenant context missing — auth middleware not applied"

    # Case 1: No state attribute
    with pytest.raises(RuntimeError, match=expected_msg):
        get_tenant_context(SimpleNamespace())

    # Case 2: state has only tenant_id
    with pytest.raises(RuntimeError, match=expected_msg):
        get_tenant_context(SimpleNamespace(state=SimpleNamespace(tenant_id="tenant_123")))

    # Case 3: state has only tier
    with pytest.raises(RuntimeError, match=expected_msg):
        get_tenant_context(SimpleNamespace(state=SimpleNamespace(tier="free")))

    # Case 4: state.tenant_id is None
    with pytest.raises(RuntimeError, match=expected_msg):
        get_tenant_context(SimpleNamespace(state=SimpleNamespace(tenant_id=None, tier="free")))

    # Case 5: state.tier is None
    with pytest.raises(RuntimeError, match=expected_msg):
        get_tenant_context(SimpleNamespace(state=SimpleNamespace(tenant_id="tenant_123", tier=None)))


def test_tenant_store_replace_tenants() -> None:
    """Verifies replace_tenants changes list identity and lookup uses new tenants."""
    raw_k1 = "secret_key_1"
    h1 = hashlib.sha256(raw_k1.encode("utf-8")).hexdigest()
    raw_k2 = "secret_key_2"
    h2 = hashlib.sha256(raw_k2.encode("utf-8")).hexdigest()

    tenant_old = Tenant(tenant_id="old_t", tier="free", key_sha256=h1)
    tenant_new = Tenant(tenant_id="new_t", tier="premium", key_sha256=h2)
    store = TenantStore([tenant_old])
    old_list = store.tenants

    assert store.lookup(raw_k1) == tenant_old
    assert store.lookup(raw_k2) is None

    store.replace_tenants([tenant_new])
    assert store.tenants is not old_list
    assert store.lookup(raw_k1) is None
    assert store.lookup(raw_k2) == tenant_new


def test_load_tenants_from_pg_success() -> None:
    """Verifies loading tenants from a fake Postgres connection returning rows."""
    async def _test() -> None:
        fake_rows = [
            {
                "tenant_id": "tenant_pg_1",
                "tier": "free",
                "key_sha256": "1" * 64,
            },
            {
                "tenant_id": "tenant_pg_2",
                "tier": "premium",
                "key_sha256": "2" * 64,
            },
        ]

        class FakeConn:
            async def fetch(self, query: str) -> list[dict[str, Any]]:
                assert "SELECT tenant_id, tier, key_sha256 FROM tenants" in query
                return fake_rows

        from tenants import load_tenants_from_pg

        tenants = await load_tenants_from_pg(FakeConn())
        assert len(tenants) == 2
        assert tenants[0].tenant_id == "tenant_pg_1"
        assert tenants[0].tier == "free"
        assert tenants[1].tenant_id == "tenant_pg_2"
        assert tenants[1].tier == "premium"

    asyncio.run(_test())


def test_load_tenants_from_pg_zero_rows_raises() -> None:
    """Verifies ValueError is raised when Postgres returns zero rows."""
    async def _test() -> None:
        class EmptyConn:
            async def fetch(self, query: str) -> list[dict[str, Any]]:
                return []

        from tenants import load_tenants_from_pg

        with pytest.raises(ValueError, match="zero rows"):
            await load_tenants_from_pg(EmptyConn())

    asyncio.run(_test())


def test_load_tenants_from_pg_bad_row_raises() -> None:
    """Verifies ValueError is raised when a row contains an invalid (e.g. uppercase) hash."""
    async def _test() -> None:
        bad_rows = [
            {
                "tenant_id": "bad_tenant",
                "tier": "free",
                "key_sha256": "A" * 64,
            }
        ]

        class BadConn:
            async def fetch(self, query: str) -> list[dict[str, Any]]:
                return bad_rows

        from tenants import load_tenants_from_pg

        with pytest.raises(ValueError, match="invalid"):
            await load_tenants_from_pg(BadConn())

    asyncio.run(_test())



