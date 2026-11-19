from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import asyncpg
from fastapi import Request, Response
from pydantic import BaseModel, Field, ValidationError, field_validator
import yaml

# Phase 12: Role-based access control (RBAC). Admin routes (/admin and /admin/*)
# require an authenticated admin key loaded into AdminStore. Tenant keys cannot
# access admin routes (yielding HTTP 403 Forbidden). Non-admin routes require a
# tenant key (admin keys yield HTTP 401 Unauthorized).


class Admin(BaseModel):
    """
    Data model representing an authenticated administrator.
    """

    admin_id: str = Field(..., pattern=r"^[a-z0-9_]{1,64}$")
    key_sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")


class Tenant(BaseModel):
    """
    Data model representing an authenticated tenant and their assigned tier.
    """

    tenant_id: str = Field(..., pattern=r"^[a-z0-9_]{1,64}$")
    tier: Literal["free", "premium"]
    key_sha256: str = Field(..., pattern=r"^[a-f0-9]{64}$")
    cache_mode: Literal["isolated", "shared", "off"] = "isolated"
    tokens_per_minute: Optional[int] = None
    spend_cap_per_day: Optional[float] = None

    @field_validator("tokens_per_minute")
    @classmethod
    def validate_tokens_per_minute(cls, v: Optional[int]) -> Optional[int]:
        """Ensures tokens_per_minute is strictly positive if provided."""
        if v is not None and v <= 0:
            raise ValueError(f"tokens_per_minute must be > 0, got {v}")
        return v

    @field_validator("spend_cap_per_day")
    @classmethod
    def validate_spend_cap_per_day(cls, v: Optional[float]) -> Optional[float]:
        """Ensures spend_cap_per_day is strictly positive if provided."""
        if v is not None and v <= 0.0:
            raise ValueError(f"spend_cap_per_day must be > 0, got {v}")
        return v


def _constant_time_lookup(raw_key: Optional[str], items: list[Any]) -> Optional[Any]:
    """
    Looks up an item by comparing raw_key's SHA-256 hash against key_sha256.

    Inputs:
    - raw_key: Plaintext API key provided in the request (or None/empty).
    - items: List of objects exposing a 'key_sha256' attribute.

    Returns:
    - Matching item, or None if raw_key is empty or no match is found.

    Security Note:
    - Hashes raw_key once, then compares against every stored key hash using
      hmac.compare_digest without early return to ensure execution time does
      not depend on match position, preventing timing side-channel attacks.
    """
    if not raw_key:
        return None

    incoming_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    matched: Optional[Any] = None

    for item in items:
        if hmac.compare_digest(incoming_hash, item.key_sha256):
            matched = item

    return matched


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
        """
        return _constant_time_lookup(raw_key, self.tenants)

    def get(self, tenant_id: str) -> Optional[Tenant]:
        """
        Looks up a tenant by their tenant_id.

        Inputs:
        - tenant_id: Unique string identifier of the tenant.

        Returns:
        - Matching Tenant instance, or None if tenant_id is not found.
        """
        for t in self.tenants:
            if t.tenant_id == tenant_id:
                return t
        return None

    def replace_tenants(self, new_tenants: list[Tenant]) -> None:
        """
        Replaces the stored tenants atomically using copy-then-assign.

        Inputs:
        - new_tenants: List of validated Tenant instances.

        Side Effects:
        - Replaces self.tenants with a new list copy without mutating the previous list in-place.
        """
        self.tenants = list(new_tenants)


class AdminStore:
    """
    In-memory store of authenticated administrators providing constant-time lookup.
    """

    def __init__(self, admins: list[Admin]) -> None:
        self.admins: list[Admin] = list(admins)

    def lookup(self, raw_key: Optional[str]) -> Optional[Admin]:
        """
        Looks up an admin by their plaintext API key using constant-time comparisons.

        Inputs:
        - raw_key: Plaintext API key provided in the request header (or None/empty).

        Returns:
        - Admin matching the key, or None if raw_key is invalid or empty.
        """
        return _constant_time_lookup(raw_key, self.admins)


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


def get_cache_mode(request: Any) -> str:
    """
    Extracts the authenticated tenant cache mode from the request state.

    Inputs:
    - request: The incoming HTTP request or request-like object.

    Returns:
    - str cache mode ('isolated', 'shared', or 'off').

    Errors:
    - Raises RuntimeError("cache mode missing — auth middleware not applied")
      if request.state is absent or cache_mode is absent or None.
    """
    if not hasattr(request, "state"):
        raise RuntimeError("cache mode missing — auth middleware not applied")
    state = request.state
    if not hasattr(state, "cache_mode"):
        raise RuntimeError("cache mode missing — auth middleware not applied")

    cache_mode = state.cache_mode
    if cache_mode is None:
        raise RuntimeError("cache mode missing — auth middleware not applied")

    return str(cache_mode)


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


async def load_tenants_from_pg(pool_or_dsn: Any) -> list[Tenant]:
    """
    Loads and validates tenant configurations from PostgreSQL.

    Inputs:
    - pool_or_dsn: asyncpg Pool, Connection, or connection DSN string.

    Returns:
    - list[Tenant] containing validated Tenant instances.

    Errors:
    - Raises ValueError if zero rows are returned, if duplicate tenant_id
      or key_sha256 entries exist, or if row validation fails.
    """
    query = "SELECT tenant_id, tier, key_sha256 FROM tenants"
    if isinstance(pool_or_dsn, str):
        conn = await asyncpg.connect(pool_or_dsn)
        try:
            rows = await conn.fetch(query)
        finally:
            await conn.close()
    elif hasattr(pool_or_dsn, "fetch"):
        rows = await pool_or_dsn.fetch(query)
    elif hasattr(pool_or_dsn, "acquire"):
        async with pool_or_dsn.acquire() as conn:
            rows = await conn.fetch(query)
    else:
        raise ValueError(f"Invalid pool_or_dsn object: {type(pool_or_dsn)}")

    if not rows:
        raise ValueError("No tenant records found in database (zero rows)")

    tenants: list[Tenant] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for idx, row in enumerate(rows):
        if isinstance(row, dict):
            row_dict = row
        else:
            try:
                row_dict = dict(row)
            except Exception:
                row_dict = {
                    "tenant_id": row["tenant_id"],
                    "tier": row["tier"],
                    "key_sha256": row["key_sha256"],
                }

        # Note: Postgres table schema is frozen and does not include cache_mode,
        # tokens_per_minute, or spend_cap_per_day; Tenant models instantiated from
        # database rows automatically receive the default cache_mode="isolated" and
        # default to tokens_per_minute=None and spend_cap_per_day=None (unlimited).
        try:
            tenant = Tenant(**row_dict)
        except ValidationError as exc:
            raise ValueError(f"Tenant database row {idx} is invalid: {exc}") from exc

        if tenant.tenant_id in seen_ids:
            raise ValueError(f"Duplicate tenant_id in database: '{tenant.tenant_id}'")
        seen_ids.add(tenant.tenant_id)

        if tenant.key_sha256 in seen_hashes:
            raise ValueError(f"Duplicate key_sha256 in database: '{tenant.key_sha256}'")
        seen_hashes.add(tenant.key_sha256)

        tenants.append(tenant)

    return tenants


def load_admins(path: str | Path) -> AdminStore:
    """
    Loads and validates administrator configurations from a YAML file.

    Note on storage:
    - Admins are always loaded from YAML. Even when LANDAUER_TENANT_SOURCE is
      'postgres', admins remain exclusively configured via YAML as PostgreSQL
      stores only tenant records.

    Inputs:
    - path: Filesystem path to the tenants YAML file.

    Returns:
    - AdminStore containing the validated Admin models.

    Errors:
    - Raises ValueError naming the file and problem if the file is missing,
      unparseable, structurally invalid, contains duplicate admin IDs,
      duplicate admin hashes, or admin hashes that overlap with tenant hashes.
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

    # The 'admins' key is optional; missing key yields an empty AdminStore
    if "admins" not in data or data["admins"] is None:
        return AdminStore([])

    raw_admins = data["admins"]
    if not isinstance(raw_admins, list):
        raise ValueError(f"Tenant configuration file '{p}' 'admins' field must be a list")

    # Extract existing tenant hashes from the same YAML file if present, to prevent cross-role key collision
    tenant_hashes: set[str] = set()
    raw_tenants = data.get("tenants")
    if isinstance(raw_tenants, list):
        for t_entry in raw_tenants:
            if isinstance(t_entry, dict) and "key_sha256" in t_entry:
                tenant_hashes.add(t_entry["key_sha256"])

    admins: list[Admin] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for idx, entry in enumerate(raw_admins):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Tenant configuration file '{p}' admin entry at index {idx} must be a dictionary"
            )

        try:
            admin = Admin(**entry)
        except ValidationError as exc:
            raise ValueError(
                f"Tenant configuration file '{p}' admin entry at index {idx} is invalid: {exc}"
            ) from exc

        if admin.admin_id in seen_ids:
            raise ValueError(
                f"Tenant configuration file '{p}' contains duplicate admin_id '{admin.admin_id}'"
            )
        seen_ids.add(admin.admin_id)

        if admin.key_sha256 in seen_hashes:
            raise ValueError(
                f"Tenant configuration file '{p}' contains duplicate key_sha256 '{admin.key_sha256}'"
            )
        seen_hashes.add(admin.key_sha256)

        if admin.key_sha256 in tenant_hashes:
            raise ValueError(
                f"Tenant configuration file '{p}' admin key_sha256 '{admin.key_sha256}' "
                f"overlaps with an existing tenant key (one key must never hold two roles)"
            )

        admins.append(admin)

    return AdminStore(admins)


def validate_no_key_overlap(
    tenants: list[Tenant] | TenantStore,
    admins: list[Admin] | AdminStore,
) -> None:
    """
    Validates that no tenant and admin share the same key hash.

    Inputs:
    - tenants: List of Tenant objects or a TenantStore.
    - admins: List of Admin objects or an AdminStore.

    Errors:
    - Raises ValueError if any admin key_sha256 equals any tenant key_sha256.
    """
    tenant_list = tenants.tenants if isinstance(tenants, TenantStore) else tenants
    admin_list = admins.admins if isinstance(admins, AdminStore) else admins

    tenant_hashes = {t.key_sha256 for t in tenant_list}
    for a in admin_list:
        if a.key_sha256 in tenant_hashes:
            raise ValueError(
                f"Key collision: admin '{a.admin_id}' key_sha256 overlaps with a tenant key"
            )


_AUTH_ERROR_BYTES = json.dumps(
    {
        "error": {
            "message": "Invalid or missing API key",
            "type": "unauthorized",
            "code": 401,
        }
    }
).encode("utf-8")

_FORBIDDEN_ERROR_BYTES = json.dumps(
    {
        "error": {
            "message": "Admin role required",
            "type": "forbidden",
            "code": 403,
        }
    }
).encode("utf-8")

EXEMPT_ROUTES = {("GET", "/health"), ("GET", "/metrics")}


def extract_raw_key(request: Request) -> Optional[str]:
    """
    Extracts the API credential from either the X-API-Key or Authorization: Bearer header.

    Inputs:
    - request: The incoming HTTP request.

    Returns:
    - Plaintext key string, or None if credentials are missing, invalid, or conflicting.

    Header Precedence & Collision Rules:
    - X-API-Key: used if present and non-empty.
    - Authorization: Bearer <token>: used if present and valid (scheme case-insensitive per RFC 6750).
    - If BOTH headers are present and non-empty with differing values, returns None.
      Security Note: Ambiguous credentials must not silently pick one header over another,
      as that could mask configuration errors or permit header-smuggling bypasses.
    - Unsupported schemes (e.g. "Basic ...") or empty token remainders return None.
    """
    raw_x_key = request.headers.get("x-api-key")
    x_key = raw_x_key.strip() if raw_x_key and raw_x_key.strip() else None

    raw_auth = request.headers.get("authorization")
    bearer_key: Optional[str] = None
    if raw_auth:
        stripped_auth = raw_auth.strip()
        if stripped_auth.lower().startswith("bearer "):
            remainder = stripped_auth[7:].strip()
            if remainder:
                bearer_key = remainder

    # Check for presence of both credential headers
    if x_key is not None and raw_auth is not None and raw_auth.strip():
        # Both headers are present: they must match exactly, otherwise reject as ambiguous
        if bearer_key is None or x_key != bearer_key:
            # Ambiguous credentials: do not guess or silently prioritize one over the other
            return None
        return x_key

    if x_key is not None:
        return x_key

    if bearer_key is not None:
        return bearer_key

    return None


def make_api_key_middleware(
    store: TenantStore,
    admin_store: Optional[AdminStore] = None,
) -> Callable[[Request, Callable[[Request], Any]], Any]:
    """
    Factory creating an ASGI HTTP middleware that validates credentials and enforces RBAC.

    Inputs:
    - store: Validated TenantStore used for tenant key lookup.
    - admin_store: Optional AdminStore used for admin key lookup. If None, treated as empty.

    Returns:
    - Middleware coroutine function taking (request, call_next).
    """
    effective_admin_store = admin_store if admin_store is not None else AdminStore([])

    async def api_key_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Any:
        # Exempt specific paths that must remain publicly accessible without credentials
        if (request.method, request.url.path) in EXEMPT_ROUTES:
            return await call_next(request)

        path = request.url.path
        is_admin_path = path == "/admin" or path.startswith("/admin/")

        raw_key = extract_raw_key(request)

        if is_admin_path:
            admin = effective_admin_store.lookup(raw_key)
            if admin is not None:
                request.state.role = "admin"
                request.state.admin_id = admin.admin_id
                return await call_next(request)

            tenant = store.lookup(raw_key)
            if tenant is not None:
                return Response(
                    content=_FORBIDDEN_ERROR_BYTES,
                    status_code=403,
                    media_type="application/json",
                )

            return Response(
                content=_AUTH_ERROR_BYTES,
                status_code=401,
                media_type="application/json",
            )

        # Non-admin path: tenant lookup only
        tenant = store.lookup(raw_key)
        if tenant is None:
            return Response(
                content=_AUTH_ERROR_BYTES,
                status_code=401,
                media_type="application/json",
            )

        request.state.tenant_id = tenant.tenant_id
        request.state.tier = tenant.tier
        request.state.role = "tenant"
        request.state.cache_mode = tenant.cache_mode

        return await call_next(request)

    return api_key_middleware

