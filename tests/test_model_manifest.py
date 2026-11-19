"""Tests for model_manifest schema validation and checksum verification."""

import hashlib
import pytest
import yaml

from model_manifest import (
    ModelChecksumError,
    load_manifest,
    sha256_of_file,
    verify_file,
    verify_model,
)


def _compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_load_manifest_validates_schema(tmp_path):
    """Test 1: load_manifest validates schema (rejects missing fields, invalid hex hashes, wrong version)."""
    # Test wrong schema version
    bad_v = tmp_path / "bad_v.yaml"
    bad_v.write_text("schema_version: 2\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_manifest(bad_v)

    # Test missing models
    no_m = tmp_path / "no_m.yaml"
    no_m.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'models' field"):
        load_manifest(no_m)

    # Test missing id
    no_id = tmp_path / "no_id.yaml"
    data = {
        "schema_version": 1,
        "models": {
            "m1": {
                "source": "huggingface",
                "repo": "org/m1",
                "revision": "r1",
                "quantization": "fp16",
                "verified": True,
                "note": "test note",
                "files": {"f.bin": "a" * 64},
            }
        },
    }
    with open(no_id, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="missing required keys"):
        load_manifest(no_id)

    # Test invalid source
    bad_src = tmp_path / "bad_src.yaml"
    data["models"]["m1"]["id"] = "m1"
    data["models"]["m1"]["source"] = "invalid_source"
    with open(bad_src, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="source 'invalid_source' must be one of"):
        load_manifest(bad_src)
    data["models"]["m1"]["source"] = "huggingface"

    # Test non-boolean verified
    bad_ver = tmp_path / "bad_ver.yaml"
    data["models"]["m1"]["verified"] = "yes"  # type: ignore[assignment]
    with open(bad_ver, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="boolean 'verified'"):
        load_manifest(bad_ver)

    # Test invalid hex hash (wrong length)
    short_hash = tmp_path / "short_hash.yaml"
    data["models"]["m1"]["verified"] = True
    data["models"]["m1"]["files"] = {"f.bin": "abcd"}
    with open(short_hash, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="invalid sha256 hash"):
        load_manifest(short_hash)

    # Test invalid hex hash (non-hex characters)
    bad_hex = tmp_path / "bad_hex.yaml"
    data["models"]["m1"]["files"] = {"f.bin": "g" * 64}
    with open(bad_hex, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="invalid sha256 hash"):
        load_manifest(bad_hex)


def test_verify_file_detects_corruption(tmp_path):
    """Test 2: verify_file detects corruption (temporary file, flip a byte)."""
    p = tmp_path / "artifact.bin"
    content = b"original cryptographic artifact content"
    p.write_bytes(content)
    expected_hash = _compute_sha256(content)

    # Mutate 1 byte
    mutated = bytearray(content)
    mutated[0] ^= 0xFF
    p.write_bytes(bytes(mutated))

    with pytest.raises(ModelChecksumError, match="Checksum mismatch"):
        verify_file(p, expected_hash)

    # Missing file raises FileNotFoundError
    missing = tmp_path / "nonexistent.bin"
    with pytest.raises(FileNotFoundError):
        verify_file(missing, expected_hash)


def test_missing_file_raises_filenotfounderror(tmp_path):
    """Missing file raises FileNotFoundError for both sha256_of_file and verify_file."""
    missing = tmp_path / "does_not_exist.bin"
    with pytest.raises(FileNotFoundError):
        sha256_of_file(missing)

    with pytest.raises(FileNotFoundError):
        verify_file(missing, "0" * 64)


def test_sha256_of_file_reads_multiple_chunks_past_1mib(tmp_path):
    """Test 3 MiB files where only the last byte differs to prove multi-chunk iteration."""
    size = 3 * 1024 * 1024
    f1_bytes = bytearray(b"A" * size)
    f2_bytes = bytearray(b"A" * size)
    # Only the very last byte differs (byte index 3,145,727)
    f1_bytes[-1] = ord(b"X")
    f2_bytes[-1] = ord(b"Y")

    f1 = tmp_path / "file1_3mib.bin"
    f2 = tmp_path / "file2_3mib.bin"
    f1.write_bytes(bytes(f1_bytes))
    f2.write_bytes(bytes(f2_bytes))

    h1 = sha256_of_file(f1)
    h2 = sha256_of_file(f2)

    # Hashes must differ because the last byte differs
    assert h1 != h2
    # Hashes must match standard hashlib sha256 of the entire files
    assert h1 == hashlib.sha256(f1_bytes).hexdigest().lower()
    assert h2 == hashlib.sha256(f2_bytes).hexdigest().lower()


def test_verify_file_accepts_valid_file(tmp_path):
    """Test 3: verify_file accepts valid file."""
    p = tmp_path / "valid.bin"
    content = b"pristine content"
    p.write_bytes(content)
    expected_hash = _compute_sha256(content)

    # Should not raise
    verify_file(p, expected_hash)
    assert sha256_of_file(p) == expected_hash


def test_verify_model_passes_on_mock(tmp_path):
    """Test 4: verify_model passes on a mock model entry with valid files in tmp_path."""
    files_root = tmp_path / "files"
    files_root.mkdir()
    f1 = files_root / "weight.safetensors"
    f2 = files_root / "config.json"
    f1.write_bytes(b"mock weight binary")
    f2.write_bytes(b"mock config json")

    manifest_file = tmp_path / "manifest.yaml"
    manifest_data = {
        "schema_version": 1,
        "models": {
            "mock-org/mock-model": {
                "id": "mock-org/mock-model",
                "source": "huggingface",
                "repo": "mock-org/mock-model",
                "revision": "abcdef123456",
                "quantization": "fp16",
                "verified": True,
                "note": "mock note",
                "files": {
                    "weight.safetensors": _compute_sha256(b"mock weight binary"),
                    "config.json": _compute_sha256(b"mock config json"),
                },
            }
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest_data, fh)

    # Should verify without error
    verify_model("mock-org/mock-model", files_root=files_root, manifest_path=manifest_file)


def test_verify_model_raises_on_unverified_model(tmp_path):
    """Test 5: verify_model raises ModelChecksumError on unverified model."""
    manifest_file = tmp_path / "manifest.yaml"
    manifest_data = {
        "schema_version": 1,
        "models": {
            "unverified-model": {
                "id": "unverified-model",
                "source": "huggingface",
                "repo": "org/repo",
                "revision": "abcdef123456",
                "quantization": "fp16",
                "verified": False,
                "note": "unverified note",
                "files": {"file.bin": "a" * 64},
            }
        },
    }
    with open(manifest_file, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest_data, fh)

    with pytest.raises(ModelChecksumError, match="marked unverified in manifest"):
        verify_model("unverified-model", files_root=tmp_path, manifest_path=manifest_file)


def test_real_manifest_passes_schema_validation():
    """Test 6: models/manifest.yaml passes schema validation when loaded by load_manifest()."""
    manifest = load_manifest("models/manifest.yaml")
    assert manifest["schema_version"] == 1
    models = manifest["models"]
    expected_ids = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        "custom_runtime",
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "ollama/qwen2.5:1.5b",
        "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]
    assert len(models) == 6
    for mid in expected_ids:
        assert mid in models
        entry = models[mid]
        assert entry["id"] == mid
        assert entry["source"] in ("huggingface", "ollama", "local_export")
        assert isinstance(entry["repo"], str) and entry["repo"].strip()
        assert isinstance(entry["revision"], str) and entry["revision"].strip()
        assert isinstance(entry["quantization"], str) and entry["quantization"].strip()
        assert isinstance(entry["verified"], bool)
        assert isinstance(entry["note"], str) and entry["note"].strip()
        assert isinstance(entry["files"], dict)
        if entry["verified"]:
            assert len(entry["files"]) > 0


def test_resolve_and_verify_model(tmp_path, monkeypatch):
    """Test resolve_and_verify_model returns revision on success and errors properly."""
    from runtime_core.batch_engine import resolve_and_verify_model

    fake_cache = tmp_path / "fake_hub_cache"
    fake_snap = fake_cache / "models--testorg--testmodel" / "snapshots" / "pinnedrev123"
    fake_snap.mkdir(parents=True)
    fake_file = fake_snap / "model.bin"
    file_content = b"fake weights data"
    fake_file.write_bytes(file_content)
    file_sha = hashlib.sha256(file_content).hexdigest()

    mock_manifest = {
        "schema_version": 1,
        "models": {
            "testorg/testmodel": {
                "id": "testorg/testmodel",
                "source": "huggingface",
                "repo": "testorg/testmodel",
                "revision": "pinnedrev123",
                "quantization": "fp16",
                "verified": True,
                "note": "verified test model",
                "files": {"model.bin": file_sha},
            },
            "testorg/unverified": {
                "id": "testorg/unverified",
                "source": "huggingface",
                "repo": "testorg/unverified",
                "revision": "unverifiedrev",
                "quantization": "fp16",
                "verified": False,
                "note": "unverified test model",
                "files": {},
            },
        },
    }

    monkeypatch.setattr("model_manifest.load_manifest", lambda *args, **kwargs: mock_manifest)
    monkeypatch.setattr("runtime_core.batch_engine.HF_HUB_CACHE", str(fake_cache))

    # 1. Unknown model -> ValueError
    with pytest.raises(ValueError, match="Unknown model: nonexistent"):
        resolve_and_verify_model("nonexistent")

    # 2. Unverified model -> ModelChecksumError
    with pytest.raises(ModelChecksumError, match="marked unverified in manifest"):
        resolve_and_verify_model("testorg/unverified")

    # 3. Valid verified model -> returns revision
    rev = resolve_and_verify_model("testorg/testmodel")
    assert rev == "pinnedrev123"

    # 4. Checksum mismatch -> ModelChecksumError
    fake_file.write_bytes(b"corrupted weights data")
    with pytest.raises(ModelChecksumError, match="Checksum mismatch"):
        resolve_and_verify_model("testorg/testmodel")


def test_guardrail_verifies_onnx_checksum():
    """Verify clean load passes."""
    import guardrail

    guardrail.verify_guardrail_onnx()


def test_guardrail_refuses_corrupted_onnx(tmp_path):
    """Flip 1 byte in a copy of onnx, verify ModelChecksumError raised."""
    import guardrail

    onnx_copy = tmp_path / "all-MiniLM-L6-v2.onnx"
    sha_copy = tmp_path / "all-MiniLM-L6-v2.onnx.sha256"

    with open("all-MiniLM-L6-v2.onnx", "rb") as f:
        data = bytearray(f.read())
    data[0] ^= 0xFF
    onnx_copy.write_bytes(bytes(data))

    with open("all-MiniLM-L6-v2.onnx.sha256", "r", encoding="utf-8") as f:
        sha_copy.write_text(f.read(), encoding="utf-8")

    with pytest.raises(ModelChecksumError, match="Checksum mismatch"):
        guardrail.verify_guardrail_onnx(str(onnx_copy), str(sha_copy))


def test_guardrail_refuses_missing_sidecar(tmp_path):
    """Missing .sha256 raises FileNotFoundError naming sidecar path and creation command."""
    import guardrail

    onnx_copy = tmp_path / "all-MiniLM-L6-v2.onnx"
    missing_sha = tmp_path / "missing.sha256"

    with open("all-MiniLM-L6-v2.onnx", "rb") as f:
        onnx_copy.write_bytes(f.read())

    with pytest.raises(FileNotFoundError) as exc_info:
        guardrail.verify_guardrail_onnx(str(onnx_copy), str(missing_sha))

    err_msg = str(exc_info.value)
    assert str(missing_sha) in err_msg
    assert "python -c" in err_msg
    assert "all-MiniLM-L6-v2.onnx.sha256" in err_msg

