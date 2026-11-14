"""Model manifest loading and cryptographic checksum verification.

Provenance and Purpose:
Phase 11 artifact integrity management. Ensures model weights, ONNX runtimes,
and configuration files match exact pinned cryptographic hashes before execution,
preventing silent corruption, supply-chain tampering, or unintended weight drifting.

Schema Documentation (models/manifest.yaml):
- schema_version: integer (must be 1)
- models: dictionary mapping model identifiers to metadata records
    - id: string, unique model identifier
    - source: string, source origin ('huggingface', 'ollama', or 'local_export')
    - repo: string, repository identifier or registry namespace
    - revision: string, git commit hash, immutable tag, or manifest sha256
    - quantization: string, precision or quantization format (e.g. 'fp16', 'bnb-4bit-on-load')
    - verified: boolean, whether files have been cryptographically verified on disk
    - note: string, explanation of provenance, verification status, or operational scope
    - files: dictionary mapping relative file paths to 64-character lowercase sha256 hex strings
"""

import hashlib
from pathlib import Path
from typing import Any, Union
import yaml


class ModelChecksumError(Exception):
    """Raised when model weights or artifacts fail cryptographic hash verification."""


def sha256_of_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """Compute lowercase SHA-256 hex digest of a file using chunked streaming.

    Purpose:
        Compute cryptographic SHA-256 hash of a file while bounding memory usage,
        allowing safe computation across large multi-gigabyte safetensors artifacts.

    Inputs:
        path (str | Path): Path to the target file to be hashed.
        chunk_size (int, optional): Buffer size in bytes for reading (defaults to 1 MiB).

    Outputs:
        str: 64-character lowercase hexadecimal SHA-256 string.

    Key Logic:
        Opens the target file in binary read mode, reads data in fixed-size buffers
        in an explicit loop until EOF, updates the hashlib.sha256 state on each chunk,
        and returns the lowercase hexadecimal representation.

    Errors:
        FileNotFoundError: If the target path does not exist or is not a regular file.
        OSError: If a filesystem read or access error occurs during stream consumption.

    Complexity:
        Time: O(N) where N is file size in bytes (single sequential read pass).
        Space: O(1) auxiliary memory (bounded by chunk_size buffer).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def load_manifest(path: Union[str, Path] = "models/manifest.yaml") -> dict[str, Any]:
    """Load, parse, and strictly validate the schema of the model manifest YAML.

    Purpose:
        Provide a single validated source of truth for all machine learning models,
        revisions, sources, verification statuses, and artifact cryptographic digests.

    Inputs:
        path (str | Path, optional): Path to manifest YAML (defaults to 'models/manifest.yaml').

    Outputs:
        dict[str, Any]: Validated manifest data dictionary conforming to schema version 1.

    Key Logic:
        Loads the YAML document using safe_load, asserts root object is a mapping, verifies
        schema_version equals 1, and iterates through every model in 'models' enforcing
        required keys ('id', 'source', 'repo', 'revision', 'quantization', 'verified', 'note',
        'files'), valid source origins ('huggingface', 'ollama', 'local_export'), boolean
        verified status, non-empty files dictionary for verified models, and 64-character
        hex digests for all referenced files.

    Errors:
        FileNotFoundError: If the specified manifest file does not exist on disk.
        ValueError: If YAML root is not a dictionary, schema_version != 1, required fields
            are missing, fields have incorrect types, or file hashes are malformed.

    Complexity:
        Time: O(M * F) where M is model count and F is average file count per model.
        Space: O(M * F) for in-memory YAML representation.
    """
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

    valid_sources = {"huggingface", "ollama", "local_export"}
    required_keys = {
        "id",
        "source",
        "repo",
        "revision",
        "quantization",
        "verified",
        "note",
        "files",
    }

    for model_key, model_entry in models.items():
        if not isinstance(model_entry, dict):
            raise ValueError(f"Model entry '{model_key}' must be a dictionary")

        missing_keys = required_keys - set(model_entry.keys())
        if missing_keys:
            raise ValueError(
                f"Model '{model_key}' is missing required keys: {sorted(missing_keys)}"
            )

        m_id = model_entry.get("id")
        if not isinstance(m_id, str) or not m_id.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'id'")

        m_src = model_entry.get("source")
        if not isinstance(m_src, str) or m_src not in valid_sources:
            raise ValueError(
                f"Model '{model_key}' source '{m_src}' must be one of {sorted(valid_sources)}"
            )

        m_repo = model_entry.get("repo")
        if not isinstance(m_repo, str) or not m_repo.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'repo'")

        m_rev = model_entry.get("revision")
        if not isinstance(m_rev, str) or not m_rev.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'revision'")

        m_quant = model_entry.get("quantization")
        if not isinstance(m_quant, str) or not m_quant.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'quantization'")

        m_ver = model_entry.get("verified")
        if not isinstance(m_ver, bool):
            raise ValueError(f"Model '{model_key}' must have a boolean 'verified'")

        m_note = model_entry.get("note")
        if not isinstance(m_note, str) or not m_note.strip():
            raise ValueError(f"Model '{model_key}' must have a non-empty string 'note'")

        m_files = model_entry.get("files")
        if not isinstance(m_files, dict):
            raise ValueError(f"Model '{model_key}' must have a 'files' dictionary")

        if m_ver is True and not m_files:
            raise ValueError(f"Model '{model_key}' is marked verified: true but has empty 'files'")

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
    """Cryptographically verify that a file matches an expected SHA-256 digest.

    Purpose:
        Prevent loading corrupted, tampered, or mismatched files by calculating
        actual SHA-256 and validating exact equality against an expected digest.

    Inputs:
        path (str | Path): Path to the target file to be verified.
        expected_sha256 (str): Expected 64-character SHA-256 hex string.

    Outputs:
        None (returns silently when verification succeeds).

    Key Logic:
        Verifies existence of target file, computes actual lowercase digest using
        sha256_of_file, and compares against expected_sha256 (case-insensitive).
        If hashes do not match, raises ModelChecksumError with informative details.

    Errors:
        FileNotFoundError: If the target file does not exist on disk.
        ModelChecksumError: If computed digest differs from expected_sha256.

    Complexity:
        Time: O(N) where N is file size in bytes.
        Space: O(1) auxiliary memory.
    """
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
    """Verify all cryptographic artifacts associated with a model ID against the manifest.

    Purpose:
        Ensure that an entire model directory snapshot (weights, configs, tokenizers)
        matches the exact cryptographic state pinned in the manifest before execution.

    Inputs:
        model_id (str): Key identifying the model in the manifest.
        files_root (str | Path): Root directory path containing the model files on disk.
        manifest_path (str | Path, optional): Path to manifest file (defaults to 'models/manifest.yaml').

    Outputs:
        None (returns silently when all files verify successfully).

    Key Logic:
        Loads manifest, retrieves entry for model_id, verifies that verified is True,
        iterates through every relative file path specified in model_entry['files'],
        resolves full path against files_root, and calls verify_file for each artifact.

    Errors:
        KeyError: If model_id is not registered in the manifest.
        ModelChecksumError: If the model is marked unverified in the manifest, or if any
            individual artifact hash verification fails.
        FileNotFoundError: If manifest file or any referenced model file is missing.

    Complexity:
        Time: O(sum(file_sizes)) across all files comprising the model snapshot.
        Space: O(1) auxiliary memory beyond loaded manifest dictionary.
    """
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
