import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import yaml

from adapters.router import EngineRouter
from engine_config import (
    EngineSpec,
    default_config_path,
    load_engine_specs,
    save_engine_specs,
)


def test_valid_load(tmp_path: Path) -> None:
    """Verifies that load_engine_specs successfully parses a compliant YAML file."""
    config_file = tmp_path / "valid_config.yaml"
    content = {
        "engines": [
            {
                "id": "vllm_prod",
                "type": "vllm",
                "url": "http://127.0.0.1:8001/v1",
                "tier": "premium",
            },
            {
                "id": "local_cpp",
                "type": "llamacpp",
                "url": "http://localhost:11434",
                "tier": "free",
            },
            {
                "id": "custom",
                "type": "custom_runtime",
                "tier": "experimental",
            },
        ]
    }
    config_file.write_text(yaml.safe_dump(content), encoding="utf-8")

    specs = load_engine_specs(config_file)
    assert len(specs) == 3
    assert specs[0].id == "vllm_prod"
    assert specs[0].type == "vllm"
    assert specs[0].url == "http://127.0.0.1:8001/v1"
    assert specs[0].tier == "premium"
    assert specs[2].id == "custom"
    assert specs[2].type == "custom_runtime"
    assert specs[2].url is None
    assert specs[2].tier == "experimental"


def test_missing_file(tmp_path: Path) -> None:
    """Verifies that attempting to load a non-existent configuration file raises ValueError."""
    non_existent = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match="Configuration file not found"):
        load_engine_specs(non_existent)


def test_engines_missing(tmp_path: Path) -> None:
    """Verifies that missing or non-list 'engines' key raises ValueError."""
    missing_key_file = tmp_path / "no_engines.yaml"
    missing_key_file.write_text("other_key: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required 'engines' key"):
        load_engine_specs(missing_key_file)

    not_a_list_file = tmp_path / "engines_not_list.yaml"
    not_a_list_file.write_text("engines: not_a_list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'engines'.*must be a list"):
        load_engine_specs(not_a_list_file)

    non_mapping_file = tmp_path / "not_mapping.yaml"
    non_mapping_file.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping/dict"):
        load_engine_specs(non_mapping_file)


def test_duplicate_id(tmp_path: Path) -> None:
    """Verifies that duplicate engine ids in configuration raise ValueError."""
    config_file = tmp_path / "duplicate.yaml"
    content = {
        "engines": [
            {
                "id": "vllm_local",
                "type": "vllm",
                "url": "http://127.0.0.1:8001/v1",
                "tier": "premium",
            },
            {
                "id": "vllm_local",
                "type": "llamacpp",
                "url": "http://localhost:11434",
                "tier": "free",
            },
        ]
    }
    config_file.write_text(yaml.safe_dump(content), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate engine id 'vllm_local'"):
        load_engine_specs(config_file)


def test_bad_id_pattern(tmp_path: Path) -> None:
    """Verifies that engine ids not matching ^[a-z0-9_]{1,64}$ raise ValueError."""
    bad_ids = ["Uppercase_ID", "with-hyphen", "with space", "", "a" * 65]
    for bad_id in bad_ids:
        config_file = tmp_path / "bad_id.yaml"
        content = {
            "engines": [
                {
                    "id": bad_id,
                    "type": "custom_runtime",
                    "tier": "test",
                }
            ]
        }
        config_file.write_text(yaml.safe_dump(content), encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid engine entry"):
            load_engine_specs(config_file)


def test_vllm_without_url(tmp_path: Path) -> None:
    """Verifies that network engines missing a url raise ValueError."""
    config_file = tmp_path / "vllm_no_url.yaml"
    content = {
        "engines": [
            {
                "id": "vllm_no_url",
                "type": "vllm",
                "tier": "premium",
            }
        ]
    }
    config_file.write_text(yaml.safe_dump(content), encoding="utf-8")
    with pytest.raises(ValueError, match="url is required for engine type 'vllm'"):
        load_engine_specs(config_file)


def test_custom_runtime_with_url(tmp_path: Path) -> None:
    """Verifies that custom_runtime engine with a url raises ValueError."""
    config_file = tmp_path / "custom_with_url.yaml"
    content = {
        "engines": [
            {
                "id": "custom_with_url",
                "type": "custom_runtime",
                "url": "http://localhost:8000",
                "tier": "experimental",
            }
        ]
    }
    config_file.write_text(yaml.safe_dump(content), encoding="utf-8")
    with pytest.raises(ValueError, match="url must be None/absent for engine type 'custom_runtime'"):
        load_engine_specs(config_file)


def test_save_load_round_trip(tmp_path: Path) -> None:
    """Verifies that saving engine specs and reloading them returns equal specs."""
    target_file = tmp_path / "roundtrip.yaml"
    original_specs = [
        EngineSpec(
            id="vllm_local",
            type="vllm",
            url="http://127.0.0.1:8001/v1",
            tier="premium",
        ),
        EngineSpec(
            id="llamacpp_local",
            type="llamacpp",
            url="http://localhost:11434",
            tier="free",
        ),
        EngineSpec(
            id="custom_runtime",
            type="custom_runtime",
            tier="experimental",
        ),
    ]

    save_engine_specs(target_file, original_specs)
    loaded_specs = load_engine_specs(target_file)

    assert loaded_specs == original_specs


def test_save_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    """Verifies that save_engine_specs cleans up all temporary files."""
    target_file = tmp_path / "clean_save.yaml"
    specs = [
        EngineSpec(
            id="custom_engine",
            type="custom_runtime",
            tier="experimental",
        )
    ]

    save_engine_specs(target_file, specs)

    all_files = list(tmp_path.iterdir())
    assert len(all_files) == 1
    assert all_files[0].name == "clean_save.yaml"


def test_committed_config_yaml_loads() -> None:
    """Verifies that the repo root config.yaml loads cleanly with the expected 4 engine ids."""
    config_path = default_config_path()
    specs = load_engine_specs(config_path)

    expected_ids = ["vllm_local", "llamacpp_local", "colab_cloud", "custom_runtime"]
    actual_ids = [spec.id for spec in specs]

    assert actual_ids == expected_ids
    assert specs[0].url == "http://127.0.0.1:8001/v1"
    assert specs[1].url == "http://localhost:11434"
    assert specs[2].url == "http://localhost:8888/v1"
    assert specs[3].url is None


class FakeAdapter:
    """Minimal fake async-generator engine adapter for router unit tests without network or GPU."""

    def __init__(self, name: str = "fake", should_fail: bool = False) -> None:
        self.name = name
        self.should_fail = should_fail
        self.closed = False

    async def generate(
        self,
        prompt: str,
        tier: str = "free",
        max_tokens: int = 100,
        messages: list[dict] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        if self.should_fail:
            raise RuntimeError(f"Fake adapter '{self.name}' simulated failure")
        yield f"token_from_{self.name}"

    async def close(self) -> None:
        self.closed = True

    async def health_check(self) -> dict:
        return {"status": "ok", "engine": self.name}


def test_router_add_then_route_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that adding an engine dynamically to the router makes it routable."""
    async def _test() -> None:
        router = EngineRouter(adapters={"existing": FakeAdapter("existing")})
        spec = EngineSpec(id="custom_added", type="custom_runtime", tier="test")

        monkeypatch.setattr("engine_config.build_adapter", lambda s: FakeAdapter(s.id))

        router.add_engine(spec)
        assert "custom_added" in router.adapters

        tokens = [token async for token in router.route_request("custom_added", "test prompt")]
        assert tokens == ["token_from_custom_added"]

    asyncio.run(_test())


def test_router_add_duplicate_raises() -> None:
    """Verifies that adding an engine with an existing id raises ValueError."""
    router = EngineRouter(adapters={"eng1": FakeAdapter("eng1")})
    spec = EngineSpec(id="eng1", type="custom_runtime", tier="test")
    with pytest.raises(ValueError, match="already registered"):
        router.add_engine(spec)


def test_router_remove_then_routing() -> None:
    """Verifies that removing an engine prevents routing to it and respects fallback rules."""
    async def _test() -> None:
        fake_vllm = FakeAdapter("vllm_local", should_fail=True)
        fake_llama = FakeAdapter("llamacpp_local")
        router = EngineRouter(
            adapters={"vllm_local": fake_vllm, "llamacpp_local": fake_llama}
        )

        # Verify normal fallback vllm -> llamacpp works when both are registered
        tokens = [t async for t in router.route_request("vllm_local", "prompt")]
        assert tokens == ["token_from_llamacpp_local"]

        # Remove llamacpp_local; fallback is no longer registered -> raises RuntimeError
        removed = router.remove_engine("llamacpp_local")
        assert removed is fake_llama
        assert not fake_llama.closed  # Client connection is NOT closed to protect in-flight requests

        with pytest.raises(RuntimeError, match="failed and no fallback is configured"):
            async for _ in router.route_request("vllm_local", "prompt"):
                pass

        # Attempting to remove absent engine raises KeyError
        with pytest.raises(KeyError, match="not registered"):
            router.remove_engine("absent_engine")

    asyncio.run(_test())


def test_router_dict_identity_changes_after_add_and_remove(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that add_engine and remove_engine perform copy-on-write without in-place mutation."""
    router = EngineRouter(adapters={"eng1": FakeAdapter("eng1")})
    initial_dict = router.adapters

    monkeypatch.setattr("engine_config.build_adapter", lambda s: FakeAdapter(s.id))
    router.add_engine(EngineSpec(id="eng2", type="custom_runtime", tier="test"))

    assert router.adapters is not initial_dict
    assert "eng2" in router.adapters
    assert "eng2" not in initial_dict

    after_add_dict = router.adapters
    router.remove_engine("eng2")

    assert router.adapters is not after_add_dict
    assert "eng2" not in router.adapters
    assert "eng2" in after_add_dict


def test_router_unknown_engine_without_vllm_local_raises_runtime_error() -> None:
    """Verifies that an unknown engine name raises RuntimeError when vllm_local is not registered."""
    async def _test() -> None:
        router = EngineRouter(adapters={"custom_runtime": FakeAdapter("custom_runtime")})
        with pytest.raises(RuntimeError, match="Unknown engine 'unknown_eng'"):
            async for _ in router.route_request("unknown_eng", "prompt"):
                pass

    asyncio.run(_test())


def _create_test_admin_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, Any, Path]:
    """Helper creating a fresh minimal FastAPI app with admin_routes mounted and a test config file."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from admin_routes import EngineAdmin, init_admin, router as admin_router

    config_file = tmp_path / "admin_test_config.yaml"
    initial_specs = [
        EngineSpec(
            id="vllm_local",
            type="vllm",
            url="http://127.0.0.1:8001/v1",
            tier="premium",
        ),
        EngineSpec(
            id="custom_runtime",
            type="custom_runtime",
            tier="experimental",
        ),
    ]
    save_engine_specs(config_file, initial_specs)

    adapters = {
        "vllm_local": FakeAdapter("vllm_local"),
        "custom_runtime": FakeAdapter("custom_runtime"),
    }
    router = EngineRouter(adapters=adapters)
    admin = EngineAdmin(router=router, specs=initial_specs, config_path=config_file)
    init_admin(admin)

    monkeypatch.setattr("admin_routes.build_adapter", lambda s: FakeAdapter(s.id))
    monkeypatch.setattr("engine_config.build_adapter", lambda s: FakeAdapter(s.id))

    app = FastAPI()
    app.state.engine_admin = admin
    app.include_router(admin_router)

    return TestClient(app), admin, config_file


def test_admin_get_engines_lists_current_specs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that GET /admin/engines lists all current in-memory engine specs."""
    client, _admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    response = client.get("/admin/engines")
    assert response.status_code == 200
    data = response.json()
    assert "engines" in data
    ids = [entry["id"] for entry in data["engines"]]
    assert ids == ["vllm_local", "custom_runtime"]


def test_admin_post_adds_and_persists_to_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that POST /admin/engines registers a new engine and saves it to disk."""
    client, admin, config_file = _create_test_admin_app(tmp_path, monkeypatch)

    payload = {
        "id": "colab_cloud",
        "type": "colab",
        "url": "http://localhost:8888/v1",
        "tier": "best_effort",
    }
    response = client.post("/admin/engines", json=payload)
    assert response.status_code == 201
    assert response.json()["id"] == "colab_cloud"

    # Verify updated in memory
    assert any(s.id == "colab_cloud" for s in admin.specs)
    assert "colab_cloud" in admin.router.adapters

    # Verify persisted to disk
    persisted = load_engine_specs(config_file)
    assert any(s.id == "colab_cloud" for s in persisted)


def test_admin_post_duplicate_returns_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that POST /admin/engines with an existing id returns 409 Conflict."""
    client, _admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    payload = {
        "id": "vllm_local",
        "type": "vllm",
        "url": "http://127.0.0.1:8001/v1",
        "tier": "premium",
    }
    response = client.post("/admin/engines", json=payload)
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == 409
    assert "already exists" in error["message"]


def test_admin_post_invalid_body_returns_422(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that POST /admin/engines with an invalid payload returns 422 Unprocessable Entity."""
    client, _admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    invalid_payload = {
        "id": "BAD-ID-WITH-HYPHEN",
        "type": "invalid_engine_type",
    }
    response = client.post("/admin/engines", json=invalid_payload)
    assert response.status_code == 422


def test_admin_delete_removes_from_memory_and_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that DELETE /admin/engines/{id} removes the engine from memory and disk."""
    client, admin, config_file = _create_test_admin_app(tmp_path, monkeypatch)

    response = client.delete("/admin/engines/custom_runtime")
    assert response.status_code == 200
    assert response.json() == {"removed": "custom_runtime"}

    # Verify removed from in-memory state
    assert not any(s.id == "custom_runtime" for s in admin.specs)
    assert "custom_runtime" not in admin.router.adapters

    # Verify removed from disk configuration
    persisted = load_engine_specs(config_file)
    assert not any(s.id == "custom_runtime" for s in persisted)


def test_admin_delete_unknown_returns_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that DELETE /admin/engines/{id} for an unregistered id returns 404 Not Found."""
    client, _admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    response = client.delete("/admin/engines/nonexistent_id")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == 404
    assert "not found" in error["message"]


def test_admin_save_failure_returns_500_and_keeps_old_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that when save_engine_specs raises OSError, a 500 error is returned and state is unchanged."""
    client, admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    def _failing_save(path: Any, specs: Any) -> None:
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr("admin_routes.save_engine_specs", _failing_save)

    payload = {
        "id": "failed_engine",
        "type": "custom_runtime",
        "tier": "experimental",
    }
    response = client.post("/admin/engines", json=payload)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == 500

    # In-memory specs must remain completely unchanged
    assert not any(s.id == "failed_engine" for s in admin.specs)
    assert "failed_engine" not in admin.router.adapters

    # GET endpoint still returns the original list
    get_response = client.get("/admin/engines")
    assert get_response.status_code == 200
    ids = [entry["id"] for entry in get_response.json()["engines"]]
    assert ids == ["vllm_local", "custom_runtime"]


def test_admin_post_reuses_built_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that POST /admin/engines reuses the adapter instance built during validation."""
    client, admin, _config_file = _create_test_admin_app(tmp_path, monkeypatch)

    call_count = 0
    fake_adapter_instance = FakeAdapter("counting_adapter")

    def _counting_build(spec: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return fake_adapter_instance

    monkeypatch.setattr("admin_routes.build_adapter", _counting_build)

    payload = {
        "id": "single_build_engine",
        "type": "custom_runtime",
        "tier": "experimental",
    }
    response = client.post("/admin/engines", json=payload)
    assert response.status_code == 201
    assert call_count == 1
    assert admin.router.adapters["single_build_engine"] is fake_adapter_instance
