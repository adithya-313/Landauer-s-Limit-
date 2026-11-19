"""Unit tests for run manifest creation, storage, and cryptographic verification."""

import asyncio
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Dict, List
import pytest

from model_manifest import ModelChecksumError
from run_with_manifest import (
    REQUIRED_MANIFEST_FIELDS,
    build_manifest,
    get_gpu_info,
    store_manifest,
    verify_engine_model,
    write_manifest,
)
from scripts.verify_run_manifest import verify_run_manifest_file


def _create_mock_environment(tmp_path: Path) -> Dict[str, Any]:
    """Helper to create dummy lockfiles, config files, and manifest for tests."""
    locks_dir = tmp_path / "locks"
    locks_dir.mkdir()
    lock_files = [
        locks_dir / "requirements-gateway.lock",
        locks_dir / "requirements-ci.lock",
        locks_dir / "requirements-migrations.lock",
    ]
    for lf in lock_files:
        lf.write_text("dummy lock content\n", encoding="utf-8")

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    cfg_file = cfg_dir / "config.yaml"
    cfg_file.write_text("dummy config content\n", encoding="utf-8")

    manifest_file = tmp_path / "manifest.yaml"
    import yaml

    data = {
        "schema_version": 1,
        "models": {
            "custom_runtime": {
                "id": "custom_runtime",
                "source": "huggingface",
                "repo": "org/custom",
                "revision": "rev111",
                "quantization": "bnb-4bit",
                "verified": True,
                "note": "verified",
                "files": {"file.bin": hashlib.sha256(b"dummy").hexdigest()},
            },
            "Qwen/Qwen2.5-1.5B-Instruct-AWQ": {
                "id": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                "source": "huggingface",
                "repo": "org/awq",
                "revision": "rev222",
                "quantization": "awq",
                "verified": False,
                "note": "unverified",
                "files": {},
            },
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    telem = out_dir / "telemetry_run1.jsonl"
    power = out_dir / "power_draw_run1.log"
    telem.write_bytes(b'{"metric": 1}\n')
    power.write_bytes(b"150W\n")

    return {
        "lockfiles": [str(p) for p in lock_files],
        "config_files": [str(cfg_file)],
        "manifest_file": str(manifest_file),
        "telem_path": str(telem),
        "power_path": str(power),
        "output_files": {
            str(telem).replace("\\", "/"): hashlib.sha256(b'{"metric": 1}\n').hexdigest(),
            str(power).replace("\\", "/"): hashlib.sha256(b"150W\n").hexdigest(),
        },
    }


def test_build_manifest_fields(tmp_path):
    """Verify build_manifest contains every REQUIRED_MANIFEST_FIELDS key and expected nulls."""
    env = _create_mock_environment(tmp_path)
    manifest = build_manifest(
        run_id="run-1234",
        started_at="2026-11-14T10:00:00Z",
        finished_at="2026-11-14T10:00:30Z",
        engine="custom_runtime",
        quantization="bnb-4bit",
        concurrency=1,
        duration_seconds=30,
        injected_failure=None,
        git_commit="abcdef123456",
        git_dirty=False,
        gpu_name="Mock GPU",
        gpu_driver="550.0",
        cuda_version="12.4",
        gpu_info_error=None,
        output_files=env["output_files"],
        manifest_path=env["manifest_file"],
        lockfiles=env["lockfiles"],
        config_files=env["config_files"],
    )

    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in manifest, f"Missing required field: {field}"

    assert manifest["routing_rule_set_version"] is None
    assert manifest["slo_version"] is None
    assert "routing rule set versions start in Phase 17" in manifest["version_notes"]
    assert manifest["engine_model"]["id"] == "custom_runtime"
    assert manifest["model_revisions"]["custom_runtime"] == "rev111"


def test_verify_run_manifest_detects_tampering(tmp_path):
    """Verify valid manifest passes, and flipping one byte in telemetry causes verification failure."""
    env = _create_mock_environment(tmp_path)
    manifest = build_manifest(
        run_id="run-abc",
        started_at="2026-11-14T10:00:00Z",
        finished_at="2026-11-14T10:00:30Z",
        engine="custom_runtime",
        quantization="bnb-4bit",
        concurrency=1,
        duration_seconds=30,
        injected_failure=None,
        git_commit="abcdef123456",
        git_dirty=False,
        gpu_name="Mock GPU",
        gpu_driver="550.0",
        cuda_version="12.4",
        gpu_info_error=None,
        output_files=env["output_files"],
        manifest_path=env["manifest_file"],
        lockfiles=env["lockfiles"],
        config_files=env["config_files"],
    )

    runs_dir = tmp_path / "runs"
    manifest_path = write_manifest(manifest, runs_dir=runs_dir)

    # Clean check: should pass
    ret_pass = verify_run_manifest_file(manifest_path)
    assert ret_pass == 0

    # Sabotage: flip 1 byte in telemetry file
    telem_file = Path(env["telem_path"])
    raw_bytes = bytearray(telem_file.read_bytes())
    raw_bytes[0] ^= 0xFF
    telem_file.write_bytes(bytes(raw_bytes))

    # Verification must fail
    ret_fail = verify_run_manifest_file(manifest_path)
    assert ret_fail == 1


def test_verify_run_manifest_missing_required_key(tmp_path):
    """Verify missing required key in manifest causes verify_run_manifest to fail."""
    manifest_file = tmp_path / "bad_manifest.json"
    import json

    data = {"schema_version": 1, "run_id": "r1"}  # missing almost all keys
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    ret = verify_run_manifest_file(manifest_file)
    assert ret == 1


def test_get_gpu_info_parsing():
    """Verify get_gpu_info parses nvidia-smi correctly and records errors gracefully."""

    def mock_run_success(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        if "--query-gpu=name,driver_version" in cmd:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="NVIDIA GeForce RTX 4090, 550.54.14\n", stderr=""
            )
        elif cmd == ["nvidia-smi"]:
            return subprocess.CompletedProcess(
                cmd, 0, stdout="... Driver Version: 550.54.14   CUDA Version: 12.4 ...\n", stderr=""
            )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="Unknown command")

    name, driver, cuda, err = get_gpu_info(run_command=mock_run_success)
    assert name == "NVIDIA GeForce RTX 4090"
    assert driver == "550.54.14"
    assert cuda == "12.4"
    assert err is None

    def mock_run_missing(cmd: List[str], **kwargs: Any) -> subprocess.CompletedProcess:
        raise FileNotFoundError("nvidia-smi not found")

    name, driver, cuda, err = get_gpu_info(run_command=mock_run_missing)
    assert name is None
    assert driver is None
    assert cuda is None
    assert err is not None
    assert "nvidia-smi not found" in err


def test_write_manifest_refuses_overwrite(tmp_path):
    """Verify write_manifest refuses to overwrite an existing manifest file."""
    manifest = {"run_id": "unique-run-123", "value": "first"}
    runs_dir = tmp_path / "runs"

    written_path = write_manifest(manifest, runs_dir=runs_dir)
    assert written_path.is_file()

    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest(manifest, runs_dir=runs_dir)


def test_store_manifest_records_parameters_and_closes_connection():
    """Verify store_manifest passes correct fields and guarantees connection close."""

    class MockConnection:
        def __init__(self):
            self.execute_args = None
            self.closed = False

        async def execute(self, query: str, *args: Any) -> None:
            self.execute_args = (query, args)

        async def close(self) -> None:
            self.closed = True

    mock_conn = MockConnection()

    async def mock_connect(dsn: str) -> MockConnection:
        return mock_conn

    manifest = {
        "run_id": "test-run-id",
        "started_at": "2026-11-14T12:00:00+00:00",
        "git_commit": "commit123",
        "engine": "custom_runtime",
        "quantization": "bnb-4bit",
        "concurrency": 4,
    }

    asyncio.run(
        store_manifest(
            manifest=manifest,
            sha="sha256hex" * 8,
            dsn="postgresql://mock",
            connect=mock_connect,
        )
    )

    assert mock_conn.closed is True
    assert mock_conn.execute_args is not None
    q, args = mock_conn.execute_args
    assert args[0] == "test-run-id"
    assert args[2] == "commit123"
    assert args[5] == 4
    assert args[6] == "sha256hex" * 8

    # Test that connection is closed even if execute raises
    class FailingConnection(MockConnection):
        async def execute(self, query: str, *args: Any) -> None:
            raise RuntimeError("Database execution error")

    failing_conn = FailingConnection()

    async def mock_failing_connect(dsn: str) -> FailingConnection:
        return failing_conn

    with pytest.raises(RuntimeError, match="Database execution error"):
        asyncio.run(
            store_manifest(
                manifest=manifest,
                sha="sha256hex" * 8,
                dsn="postgresql://mock",
                connect=mock_failing_connect,
            )
        )

    assert failing_conn.closed is True


def test_verify_engine_model_validation(tmp_path):
    """Verify unknown engine raises ValueError, and unverified model raises ModelChecksumError."""
    # 1. Unknown engine -> ValueError
    with pytest.raises(ValueError, match="Unknown engine"):
        verify_engine_model("invalid_engine_name")

    # 2. vllm_local maps to AWQ which is verified: false -> ModelChecksumError
    benchmark_called = False

    async def mock_run_benchmark(*args: Any, **kwargs: Any) -> None:
        nonlocal benchmark_called
        benchmark_called = True

    with pytest.raises(ModelChecksumError, match="cannot be verified"):
        verify_engine_model("vllm_local")

    assert benchmark_called is False

