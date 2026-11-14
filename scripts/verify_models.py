"""CLI verifier for cryptographic integrity of model artifacts."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_manifest import (
    ModelChecksumError,
    load_manifest,
    verify_file,
)


def resolve_files_root(model_id: str, model_entry: dict, hf_cache: Path, repo_root: Path) -> Path:
    """Resolve local path to model artifacts."""
    files = model_entry.get("files", {})
    if "all-MiniLM-L6-v2.onnx" in files:
        return repo_root

    source = model_entry.get("source", "")
    revision = model_entry.get("revision", "")
    if "/" in source:
        org, repo = source.split("/", 1)
        return hf_cache / f"models--{org}--{repo}" / "snapshots" / revision

    return repo_root


def verify_models_cli() -> int:
    """Run model verification according to CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model identifier to verify (default: all verified: true models)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all models in manifest; warn on models marked verified: false",
    )
    parser.add_argument(
        "--manifest",
        default="models/manifest.yaml",
        help="Path to model manifest file (default: models/manifest.yaml)",
    )
    parser.add_argument(
        "--hf-cache",
        default=str(Path.home() / ".cache" / "huggingface" / "hub"),
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
        targets = [m_id for m_id, m in models.items() if m.get("verified") is True]

    has_failure = False

    for model_id in targets:
        m_entry = models[model_id]
        if not m_entry.get("verified", False):
            print(f"[WARN] {model_id}: marked verified=false in manifest; skipping file verification")
            continue

        files_root = resolve_files_root(model_id, m_entry, hf_cache, repo_root)
        if not files_root.exists():
            print(f"[FAIL] {model_id}: files root directory not found: {files_root}")
            has_failure = True
            continue

        for rel_path, expected_hash in m_entry.get("files", {}).items():
            file_path = files_root / rel_path
            try:
                verify_file(file_path, expected_hash)
                print(f"[PASS] {model_id}/{rel_path} ({expected_hash[:12]}...)")
            except FileNotFoundError:
                print(f"[FAIL] {model_id}/{rel_path}: file not found at {file_path}")
                has_failure = True
            except ModelChecksumError as err:
                print(f"[FAIL] {model_id}/{rel_path}: {err}")
                has_failure = True

    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(verify_models_cli())
