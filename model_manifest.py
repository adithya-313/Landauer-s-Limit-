"""Model manifest loading and cryptographic checksum verification.

Provenance and Purpose:
Phase 11 artifact integrity management. Ensures model weights, ONNX runtimes,
and configuration files match exact pinned cryptographic hashes before execution,
preventing silent corruption, supply-chain tampering, or unintended weight drifting.

Schema Documentation (models/manifest.yaml):
- schema_version: integer (must be 1)
- models: dictionary mapping model identifiers to metadata records
    - id: string, unique model identifier
    - revision: string, git commit hash or immutable tag
    - source: string, Hugging Face repository or origin URL
    - verified: boolean, whether files have been cryptographically verified on disk
    - files: dictionary mapping relative file paths to 64-character lowercase sha256 hex strings
"""

import hashlib
from pathlib import Path
from typing import Any, Union
import yaml


class ModelChecksumError(Exception):
    """Raised when model weights or artifacts fail cryptographic hash verification."""


def sha256_of_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute lowercase SHA-256 hex digest of a file in 1 MiB chunks to avoid memory bloat."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def load_manifest(path: Union[str, Path] = "models/manifest.yaml") -> dict[str, Any]:
    """Load and strictly validate the model manifest schema."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Model manifest not found: {p}")

    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError("Manifest root must be a mapping/dict")

    if data.get("schema_version") != 1:
        raise ValueError(
            f"Invalid or unsupported schema_version: {data.get('schema_version')}, expected 1"
        )

    models = data.get("models")
    if not isinstance(models, dict):
        raise ValueError("'models' field must be a dictionary")

    for model_key, model_entry in models.items():
        if not isinstance(model_entry, dict):
            raise ValueError(f"Model entry '{model_key}' must be a dictionary")

        m_id = model_entry.get("id")
        if not isinstance(m_id, str) or not m_id.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'id'")

        m_rev = model_entry.get("revision")
        if not isinstance(m_rev, str) or not m_rev.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'revision'")

        m_src = model_entry.get("source")
        if not isinstance(m_src, str) or not m_src.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'source'")

        m_ver = model_entry.get("verified")
        if not isinstance(m_ver, bool):
            raise ValueError(f"Model '{model_key}' must have a boolean 'verified'")

        m_files = model_entry.get("files")
        if not isinstance(m_files, dict) or not m_files:
            raise ValueError(f"Model '{model_key}' must have a non-empty 'files' dictionary")

        for f_path, f_hash in m_files.items():
            if not isinstance(f_path, str) or not f_path.strip():
                raise ValueError(
                    f"Model '{model_key}' file path must be a non-empty string: {f_path}"
                )
            if (
                not isinstance(f_hash, str)
                or len(f_hash) != 64
                or not all(c in "0123456789abcdefABCDEF" for c in f_hash)
            ):
                raise ValueError(
                    f"Model '{model_key}' file '{f_path}' has invalid sha256 hash: {f_hash}"
                )

    return data


def verify_file(path: Union[str, Path], expected_sha256: str) -> None:
    """Verify that a single file matches the expected SHA-256 digest."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File to verify does not exist: {p}")

    actual_hash = sha256_of_file(p)
    if actual_hash.lower() != expected_sha256.lower():
        raise ModelChecksumError(
            f"Checksum mismatch for {p}: expected {expected_sha256.lower()}, got {actual_hash.lower()}"
        )


def verify_model(
    model_id: str,
    files_root: Union[str, Path],
    manifest_path: Union[str, Path] = "models/manifest.yaml",
) -> None:
    """Verify all files belonging to a model entry against the manifest."""
    manifest = load_manifest(manifest_path)
    models = manifest.get("models", {})
    if model_id not in models:
        raise KeyError(f"Model '{model_id}' not found in manifest: {manifest_path}")

    model_entry = models[model_id]
    if not model_entry.get("verified", False):
        raise ModelChecksumError(
            f"Model {model_id} marked unverified in manifest; run verification first"
        )

    root = Path(files_root)
    files = model_entry.get("files", {})
    for rel_path, expected_hash in files.items():
        file_path = root / rel_path
        verify_file(file_path, expected_hash)
