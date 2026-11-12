from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from fastapi import Request, Response
from pydantic import BaseModel, Field, ValidationError
import yaml

# Note: In Phase 9, any valid tenant key may access all routes including /admin/*;
# role-based access control (e.g. admin vs tenant) is deferred to Phase 12.


class Tenant(BaseModel):
    """
    Data model representing an authenticated tenant and their assigned tier.
    """

    tenant_id: str = Field(..., pattern=r"^[a-z0-9_]{1,64}$")
    tier: Literal["free", "premium"]
    key_sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")


class TenantStore:
    """
    In-memory store of authenticated tenants providing constant-time lookup.
    """

    def __init__(self, tenants: list[Tenant]) -> None:
        self.tenants: list[Tenant] = list(tenants)

    def lookup(self, raw_key: Optional[str]) -> Optional[Tenant]:
        """
        Looks up a tenant by their plaintext API key using constant-time comparisons.

        Inputs:
        - raw_key: Plaintext API key provided in the request header (or None/empty).

        Returns:
        - Tenant matching the key, or None if raw_key is invalid or empty.

        Security Note:
        - We compare against every stored key hash using hmac.compare_digest
          without early return to ensure execution time does not depend on the
          matched tenant's position in the list, preventing timing side-channel attacks.
        """
        if not raw_key:
            return None

        incoming_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        matched: Optional[Tenant] = None

        # Compare against every stored tenant hash without early exit to maintain constant time
        for tenant in self.tenants:
            if hmac.compare_digest(incoming_hash, tenant.key_sha256):
                matched = tenant

        return matched


def get_tenant_context(request: Any) -> tuple[str, str]:
    """
    Extracts the authenticated tenant ID and tier from the request state.

    Inputs:
    - request: The incoming HTTP request or request-like object.

    Returns:
    - tuple[str, str] of (tenant_id, tier).

    Errors:
    - Raises RuntimeError("tenant context missing — auth middleware not applied")
      if request.state is absent or either tenant_id or tier is absent or None.
    """
    if not hasattr(request, "state"):
        raise RuntimeError("tenant context missing — auth middleware not applied")
    state = request.state
    if not hasattr(state, "tenant_id") or not hasattr(state, "tier"):
        raise RuntimeError("tenant context missing — auth middleware not applied")

    tenant_id = state.tenant_id
    tier = state.tier
    if tenant_id is None or tier is None:
        raise RuntimeError("tenant context missing — auth middleware not applied")

    return str(tenant_id), str(tier)


def default_tenants_path() -> Path:
    """
    Returns the filesystem path to the tenants configuration file.
    Honors LANDAUER_TENANTS_PATH if set, otherwise defaults to <repo root>/tenants.yaml.
    """
    env_path = os.getenv("LANDAUER_TENANTS_PATH")
    if env_path is not None and env_path.strip():
        return Path(env_path)
    repo_root = Path(__file__).resolve().parent
    return repo_root / "tenants.yaml"


def load_tenants(path: str | Path) -> TenantStore:
    """
    Loads and validates tenant configurations from a YAML file.

    Inputs:
    - path: Filesystem path to the tenants YAML file.

    Returns:
    - TenantStore containing the validated Tenant models.

    Errors:
    - Raises ValueError naming the file and problem if the file is missing,
      unparseable, structurally invalid, contains duplicate IDs, or duplicate hashes.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Tenant configuration file '{p}' does not exist")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise ValueError(f"Failed to parse tenant configuration file '{p}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Tenant configuration file '{p}' must contain a top-level mapping")

    raw_tenants = data.get("tenants")
    if not isinstance(raw_tenants, list):
        raise ValueError(f"Tenant configuration file '{p}' must contain a 'tenants' list")

    tenants: list[Tenant] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for idx, entry in enumerate(raw_tenants):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Tenant configuration file '{p}' entry at index {idx} must be a dictionary"
            )

        try:
            tenant = Tenant(**entry)
        except ValidationError as exc:
            raise ValueError(
                f"Tenant configuration file '{p}' entry at index {idx} is invalid: {exc}"
            ) from exc

        if tenant.tenant_id in seen_ids:
            raise ValueError(
                f"Tenant configuration file '{p}' contains duplicate tenant_id '{tenant.tenant_id}'"
            )
        seen_ids.add(tenant.tenant_id)

        if tenant.key_sha256 in seen_hashes:
            raise ValueError(
                f"Tenant configuration file '{p}' contains duplicate key_sha256 '{tenant.key_sha256}'"
            )
        seen_hashes.add(tenant.key_sha256)

        tenants.append(tenant)

    return TenantStore(tenants)


_AUTH_ERROR_BYTES = json.dumps(
    {
        "error": {
            "message": "Invalid or missing API key",
            "type": "unauthorized",
            "code": 401,
        }
    }
).encode("utf-8")

EXEMPT_ROUTES = {("GET", "/health"), ("GET", "/metrics")}


def make_api_key_middleware(
    store: TenantStore,
) -> Callable[[Request, Callable[[Request], Any]], Any]:
    """
    Factory creating an ASGI HTTP middleware that validates X-API-Key against the TenantStore.

    Inputs:
    - store: Validated TenantStore used for constant-time key hash lookup.

    Returns:
    - Middleware coroutine function taking (request, call_next).
    """

    async def api_key_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        # Exempt specific paths that must remain publicly accessible without credentials
        if (request.method, request.url.path) in EXEMPT_ROUTES:
            return await call_next(request)

        raw_key = request.headers.get("x-api-key")
        tenant = store.lookup(raw_key)

        if tenant is None:
            return Response(
                content=_AUTH_ERROR_BYTES,
                status_code=401,
                media_type="application/json",
            )

        request.state.tenant_id = tenant.tenant_id
        request.state.tier = tenant.tier

        return await call_next(request)

    return api_key_middleware

