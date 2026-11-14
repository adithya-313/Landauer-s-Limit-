"""CLI verifier for cryptographic integrity of model artifacts."""

import argparse
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from huggingface_hub.constants import HF_HUB_CACHE

from model_manifest import (
    ModelChecksumError,
    load_manifest,
    verify_file,
)


def resolve_files_root(
    model_id: str,
    model_entry: dict,
    hf_cache: Path,
    repo_root: Path,
) -> Path:
    """Resolve local path to model artifacts based on model metadata."""
    source = model_entry.get("source", "")
    revision = model_entry.get("revision", "")

    if source == "ollama":
        return Path.home() / ".ollama" / "models" / "blobs"

    repo = model_entry.get("repo", model_id)
    if source == "local_export":
        if "/" in repo:
            org, name = repo.split("/", 1)
            return hf_cache / f"models--{org}--{name}" / "snapshots" / revision
        return repo_root

    if source == "huggingface":
        if "/" in repo:
            org, name = repo.split("/", 1)
            hf_snap = hf_cache / f"models--{org}--{name}" / "snapshots" / revision
            if hf_snap.exists():
                return hf_snap
            ci_path = repo_root / "ci-models"
            if ci_path.exists():
                return ci_path
            return hf_snap
        return repo_root

    return repo_root


def verify_models_cli() -> int:
    """Run model verification according to CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model identifier to verify",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all models in manifest; reports failure for unverified models",
    )
    parser.add_argument(
        "--manifest",
        default="models/manifest.yaml",
        help="Path to model manifest file (default: models/manifest.yaml)",
    )
    parser.add_argument(
        "--hf-cache",
        default=str(HF_HUB_CACHE),
        help="Path to local Hugging Face hub cache directory",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    hf_cache = Path(args.hf_cache)
    repo_root = Path(".")

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to load manifest {manifest_path}: {exc}\n")
        return 1

    models = manifest.get("models", {})

    if args.model:
        if args.model not in models:
            sys.stderr.write(f"Model '{args.model}' not found in {manifest_path}\n")
            return 1
        targets = [args.model]
    elif args.all:
        targets = list(models.keys())
    else:
        targets = list(models.keys())

    has_failure = False

    for model_id in targets:
        m_entry = models[model_id]
        if not m_entry.get("verified", False):
            note = m_entry.get("note", "unverified model")
            print(f"[FAIL] {model_id}: cannot be verified on this machine: {note}")
            has_failure = True
            continue

        files_root = resolve_files_root(model_id, m_entry, hf_cache, repo_root)
        files = m_entry.get("files", {})
        if not files_root.exists():
            print(f"[FAIL] {model_id}: files root directory not found: {files_root}")
            has_failure = True
            continue

        model_failed = False
        for rel_path, expected_hash in files.items():
            file_path = files_root / rel_path
            try:
                verify_file(file_path, expected_hash)
            except FileNotFoundError:
                print(f"[FAIL] {model_id}: file not found at {file_path}")
                model_failed = True
                has_failure = True
                break
            except ModelChecksumError as err:
                print(f"[FAIL] {model_id}: {err}")
                model_failed = True
                has_failure = True
                break

        if not model_failed:
            rev_str = str(m_entry.get("revision", ""))[:12]
            print(f"[PASS] {model_id}: all {len(files)} files verified ({rev_str})")

    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(verify_models_cli())
