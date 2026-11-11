from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from adapters.router import EngineRouter
from engine_config import EngineSpec, build_adapter, save_engine_specs

logger = logging.getLogger("admin_routes")


class EngineAdmin:
    """
    Manages the live engine registry and synchronizes updates with configuration storage.

    Owns:
    - router: EngineRouter instance for live request routing.
    - specs: Current list of in-memory EngineSpec models.
    - config_path: Filesystem path to the YAML configuration file.
    - lock: asyncio.Lock guarding all mutating registry operations.
    """

    def __init__(
        self,
        router: EngineRouter,
        specs: list[EngineSpec],
        config_path: str | Path,
    ) -> None:
        self.router = router
        self.specs: list[EngineSpec] = list(specs)
        self.config_path: Path = Path(config_path)
        self.lock = asyncio.Lock()


_active_admin: Optional[EngineAdmin] = None


def init_admin(admin: EngineAdmin) -> EngineAdmin:
    """
    Sets the active module-level EngineAdmin instance.

    Inputs:
    - admin: EngineAdmin instance to wire into the admin routes.

    Returns:
    - The initialized EngineAdmin instance.
    """
    global _active_admin
    _active_admin = admin
    return admin


def get_engine_admin(request: Request) -> EngineAdmin:
    """
    FastAPI dependency resolving the current EngineAdmin instance.
    Checks request.app.state first, falling back to the module-level instance.

    Errors:
    - Raises RuntimeError if no EngineAdmin has been registered.
    """
    if hasattr(request.app.state, "engine_admin"):
        admin = request.app.state.engine_admin
        if isinstance(admin, EngineAdmin):
            return admin
    if _active_admin is not None:
        return _active_admin
    raise RuntimeError("EngineAdmin is not initialized")


router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/engines")
async def list_engines(
    admin: EngineAdmin = Depends(get_engine_admin),
) -> dict[str, Any]:
    """
    Returns the current in-memory engine specifications.

    Returns:
    - 200 OK with {"engines": [spec dicts]}.
    """
    async with admin.lock:
        return {
            "engines": [
                spec.model_dump(mode="python", exclude_none=True)
                for spec in admin.specs
            ]
        }


@router.post("/engines")
async def add_engine(
    spec: EngineSpec,
    admin: EngineAdmin = Depends(get_engine_admin),
) -> JSONResponse:
    """
    Dynamically registers a new engine and persists it to the configuration file.

    Inputs:
    - spec: Validated EngineSpec payload in request body.

    Returns:
    - 201 Created with the registered EngineSpec on success.
    - 409 Conflict if the engine id is already registered.
    - 500 Internal Server Error if writing to disk fails (in-memory state preserved).
    """
    async with admin.lock:
        # Check for duplicate engine id
        if any(existing.id == spec.id for existing in admin.specs):
            return JSONResponse(
                status_code=409,
                content={
                    "error": {
                        "message": f"Engine '{spec.id}' already exists",
                        "code": 409,
                    }
                },
            )

        # Step 1: Pre-validate adapter construction. Fail before modifying file if invalid.
        try:
            adapter = build_adapter(spec)
        except Exception as exc:
            logger.error("Failed to build adapter for engine '%s': %s", spec.id, exc)
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "message": f"Failed to instantiate adapter for engine '{spec.id}': {exc}",
                        "code": 400,
                    }
                },
            )

        # Step 2: Persist new specifications to configuration file atomically
        new_specs = list(admin.specs) + [spec]
        try:
            save_engine_specs(admin.config_path, new_specs)
        except OSError as exc:
            logger.error("Failed to save engine configuration: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Failed to persist engine configuration: {exc}",
                        "code": 500,
                    }
                },
            )

        # Step 3: Swap in-memory state on the router and admin holder
        admin.router.add_engine(spec, adapter=adapter)
        admin.specs = new_specs

        return JSONResponse(
            status_code=201,
            content=spec.model_dump(mode="python", exclude_none=True),
        )


@router.delete("/engines/{engine_id}")
async def delete_engine(
    engine_id: str,
    admin: EngineAdmin = Depends(get_engine_admin),
) -> JSONResponse:
    """
    Unregisters an engine and removes it from the configuration file.

    Inputs:
    - engine_id: Identifier of the engine to delete.

    Returns:
    - 200 OK with {"removed": engine_id} on success.
    - 404 Not Found if engine_id is not registered.
    - 500 Internal Server Error if saving to disk fails.
    """
    async with admin.lock:
        target_spec = next((s for s in admin.specs if s.id == engine_id), None)
        if target_spec is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {
                        "message": f"Engine '{engine_id}' not found",
                        "code": 404,
                    }
                },
            )

        # Step 1: Persist updated specifications list to disk
        new_specs = [s for s in admin.specs if s.id != engine_id]
        try:
            save_engine_specs(admin.config_path, new_specs)
        except OSError as exc:
            logger.error("Failed to save engine configuration: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": f"Failed to persist engine configuration: {exc}",
                        "code": 500,
                    }
                },
            )

        # Step 2: Unregister engine from router and update in-memory state
        admin.router.remove_engine(engine_id)
        admin.specs = new_specs

        return JSONResponse(
            status_code=200,
            content={"removed": engine_id},
        )
