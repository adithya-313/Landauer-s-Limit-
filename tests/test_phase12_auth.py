"""
tests/test_phase12_auth.py
==========================
Unit and integration tests for Phase 12 RBAC authentication:
- Admin role enforcement on /admin/* endpoints (HTTP 403 for tenants).
- Admin keys permitted on /admin/* but rejected with HTTP 401 on /v1/chat/completions.
- Path boundary testing (/administrator vs /admin and /admin/*).
- AdminStore constant-time lookup and load_admins validation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from payload_guard import check_payload_size
import pytest
import yaml

from tenants import (
    Admin,
    AdminStore,
    Tenant,
    TenantStore,
    load_admins,
    make_api_key_middleware,
)

_EXPECTED_FORBIDDEN_BODY: dict[str, Any] = {
    "error": {
        "message": "Admin role required",
        "type": "forbidden",
        "code": 403,
    }
}


def _create_test_app(
    tenant_store: TenantStore,
    admin_store: AdminStore,
) -> TestClient:
    """
    Creates a minimal FastAPI application with payload and auth middlewares
    registered in identical order to gateway.py for authentication testing.
    """
    app = FastAPI()

    # Middleware execution order in Starlette: later registered runs first
    app.middleware("http")(check_payload_size)
    app.middleware("http")(make_api_key_middleware(tenant_store, admin_store))

    @app.get("/health")
    def health_stub() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/admin/engines")
    def admin_engines_get() -> dict[str, str]:
        return {"action": "list"}

    @app.post("/admin/engines")
    def admin_engines_post() -> dict[str, str]:
        return {"action": "create"}

    @app.delete("/admin/engines/{engine_id}")
    def admin_engines_delete(engine_id: str) -> dict[str, str]:
        return {"action": "delete", "engine_id": engine_id}

    @app.post("/v1/chat/completions")
    async def chat_stub(request: Request) -> dict[str, Any]:
        return {
            "tenant_id": getattr(request.state, "tenant_id", None),
            "tier": getattr(request.state, "tier", None),
            "role": getattr(request.state, "role", None),
        }

    @app.get("/administrator")
    def administrator_stub(request: Request) -> dict[str, Any]:
        return {
            "path": "administrator",
            "role": getattr(request.state, "role", None),
        }

    return TestClient(app)


@pytest.fixture
def test_setup() -> tuple[TestClient, str, str]:
    """
    Sets up isolated tenant and admin stores with known keys.
    Returns (client, tenant_key, admin_key).
    """
    tenant_key = "tenant_secret_key_12345"
    admin_key = "admin_secret_key_67890"

    tenant_hash = hashlib.sha256(tenant_key.encode("utf-8")).hexdigest()
    admin_hash = hashlib.sha256(admin_key.encode("utf-8")).hexdigest()

    tenant_store = TenantStore([Tenant(tenant_id="test_tenant", tier="premium", key_sha256=tenant_hash)])
    admin_store = AdminStore([Admin(admin_id="test_admin", key_sha256=admin_hash)])

    client = _create_test_app(tenant_store, admin_store)
    return client, tenant_key, admin_key


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/engines"),
        ("POST", "/admin/engines"),
        ("DELETE", "/admin/engines/x"),
    ],
)
def test_tenant_key_returns_403_on_admin_routes(
    test_setup: tuple[TestClient, str, str],
    method: str,
    path: str,
) -> None:
    """Verifies that a valid tenant key returns HTTP 403 Forbidden on all /admin routes."""
    client, tenant_key, _ = test_setup

    response = client.request(method, path, headers={"X-API-Key": tenant_key})
    assert response.status_code == 403
    assert response.json() == _EXPECTED_FORBIDDEN_BODY


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/engines"),
        ("POST", "/admin/engines"),
        ("DELETE", "/admin/engines/x"),
    ],
)
def test_admin_key_returns_200_on_admin_routes(
    test_setup: tuple[TestClient, str, str],
    method: str,
    path: str,
) -> None:
    """Verifies that a valid admin key returns HTTP 200 OK on all /admin routes."""
    client, _, admin_key = test_setup

    response = client.request(method, path, headers={"X-API-Key": admin_key})
    assert response.status_code == 200


def test_admin_key_on_chat_completions_returns_401_identical_to_missing_key(
    test_setup: tuple[TestClient, str, str],
) -> None:
    """
    Verifies that presenting an admin key on non-admin routes returns HTTP 401
    with byte-identical response body to a request with a missing API key.
    """
    client, _, admin_key = test_setup

    resp_admin = client.post("/v1/chat/completions", headers={"X-API-Key": admin_key})
    resp_missing = client.post("/v1/chat/completions")

    assert resp_admin.status_code == 401
    assert resp_missing.status_code == 401
    assert resp_admin.content == resp_missing.content


def test_no_key_on_admin_route_returns_401(
    test_setup: tuple[TestClient, str, str],
) -> None:
    """Verifies that accessing an admin route with no credentials returns HTTP 401, not 403."""
    client, _, _ = test_setup

    response = client.get("/admin/engines")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == 401


def test_administrator_path_not_treated_as_admin_path(
    test_setup: tuple[TestClient, str, str],
) -> None:
    """
    Verifies that paths like /administrator are not treated as /admin routes,
    and a tenant key can reach them without receiving 403.
    """
    client, tenant_key, _ = test_setup

    response = client.get("/administrator", headers={"X-API-Key": tenant_key})
    assert response.status_code == 200
    assert response.json()["path"] == "administrator"
    assert response.json()["role"] == "tenant"


def test_load_admins_missing_admins_key_returns_empty_store(tmp_path: Path) -> None:
    """Verifies that a configuration without an 'admins' key yields an empty AdminStore."""
    config_file = tmp_path / "tenants_no_admin.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "tenants": [
                    {
                        "tenant_id": "tenant_1",
                        "tier": "free",
                        "key_sha256": "1" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    store = load_admins(config_file)
    assert len(store.admins) == 0


def test_load_admins_present_but_not_list_raises(tmp_path: Path) -> None:
    """Verifies that a non-list 'admins' field raises ValueError."""
    config_file = tmp_path / "bad_admins.yaml"
    config_file.write_text(
        yaml.safe_dump({"admins": "invalid_not_a_list"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="'admins' field must be a list"):
        load_admins(config_file)


def test_load_admins_duplicate_admin_id_raises(tmp_path: Path) -> None:
    """Verifies that duplicate admin_id entries raise ValueError."""
    config_file = tmp_path / "dup_admin_id.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "admins": [
                    {"admin_id": "adm_1", "key_sha256": "a" * 64},
                    {"admin_id": "adm_1", "key_sha256": "b" * 64},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate admin_id 'adm_1'"):
        load_admins(config_file)


def test_load_admins_duplicate_key_sha256_raises(tmp_path: Path) -> None:
    """Verifies that duplicate admin key hashes raise ValueError."""
    config_file = tmp_path / "dup_admin_hash.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "admins": [
                    {"admin_id": "adm_1", "key_sha256": "c" * 64},
                    {"admin_id": "adm_2", "key_sha256": "c" * 64},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key_sha256"):
        load_admins(config_file)


def test_load_admins_admin_hash_equal_to_tenant_hash_raises(tmp_path: Path) -> None:
    """Verifies that an admin key matching an existing tenant key raises ValueError."""
    colliding_hash = "d" * 64
    config_file = tmp_path / "collision.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "tenants": [
                    {
                        "tenant_id": "tenant_shared",
                        "tier": "free",
                        "key_sha256": colliding_hash,
                    }
                ],
                "admins": [
                    {
                        "admin_id": "admin_shared",
                        "key_sha256": colliding_hash,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps with an existing tenant key"):
        load_admins(config_file)
