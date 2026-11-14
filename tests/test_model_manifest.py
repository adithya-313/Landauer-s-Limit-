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
                "revision": "r1",
                "source": "s1",
                "verified": True,
                "files": {"f.bin": "a" * 64},
            }
        },
    }
    with open(no_id, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)
    with pytest.raises(ValueError, match="non-empty string 'id'"):
        load_manifest(no_id)

    # Test non-boolean verified
    bad_ver = tmp_path / "bad_ver.yaml"
    data["models"]["m1"]["id"] = "m1"
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
                "revision": "abcdef123456",
                "source": "mock-org/mock-model",
                "verified": True,
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
                "revision": "abcdef123456",
                "source": "org/repo",
                "verified": False,
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
        "sentence-transformers/all-MiniLM-L6-v2",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "Qwen2.5-1.5B-Instruct-GGUF",
    ]
    for mid in expected_ids:
        assert mid in models
        assert models[mid]["id"] == mid
        assert models[mid]["revision"]
        assert models[mid]["source"]
        assert isinstance(models[mid]["verified"], bool)
        assert len(models[mid]["files"]) > 0


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
    """Missing .sha256 raises ModelChecksumError."""
    import guardrail

    onnx_copy = tmp_path / "all-MiniLM-L6-v2.onnx"
    missing_sha = tmp_path / "missing.sha256"

    with open("all-MiniLM-L6-v2.onnx", "rb") as f:
        onnx_copy.write_bytes(f.read())

    with pytest.raises(ModelChecksumError, match="Missing ONNX checksum sidecar"):
        guardrail.verify_guardrail_onnx(str(onnx_copy), str(missing_sha))

