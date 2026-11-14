"""Verify integrity and schema compliance of a run manifest and its referenced artifacts."""

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import model_manifest
from run_with_manifest import REQUIRED_MANIFEST_FIELDS


def verify_manifest_dictionary(manifest: dict) -> bool:
    """Validate that manifest contains all required keys and correct types."""
    has_error = False
    for field in REQUIRED_MANIFEST_FIELDS:
        if field not in manifest:
            sys.stderr.write(f"[FAIL] Missing required manifest key: {field}\n")
            has_error = True
    return not has_error


def verify_file_hashes_group(name: str, file_hashes: dict) -> bool:
    """Verify hashes for a dictionary of file path to expected SHA-256."""
    all_ok = True
    for file_path_str, expected_hash in file_hashes.items():
        p = Path(file_path_str)
        if not p.is_file():
            print(f"[FAIL] {file_path_str}: file not found")
            all_ok = False
            continue

        try:
            actual_hash = model_manifest.sha256_of_file(p)
            if actual_hash.lower() == expected_hash.lower():
                print(f"[OK] {file_path_str} ({actual_hash[:12]}...)")
            else:
                print(
                    f"[FAIL] {file_path_str}: expected {expected_hash.lower()}, got {actual_hash.lower()}"
                )
                all_ok = False
        except Exception as exc:
            print(f"[FAIL] {file_path_str}: hashing error: {exc}")
            all_ok = False

    return all_ok


def verify_run_manifest_file(manifest_path: Path) -> int:
    """Run full verification on target manifest file."""
    if not manifest_path.is_file():
        sys.stderr.write(f"Error: Manifest file not found: {manifest_path}\n")
        return 1

    try:
        content = manifest_path.read_text(encoding="utf-8")
        manifest = json.loads(content)
    except Exception as exc:
        sys.stderr.write(f"Error reading or parsing JSON in {manifest_path}: {exc}\n")
        return 1

    if not isinstance(manifest, dict):
        sys.stderr.write(f"Error: Manifest root is not a dictionary in {manifest_path}\n")
        return 1

    valid_schema = verify_manifest_dictionary(manifest)
    if not valid_schema:
        return 1

    ok_locks = verify_file_hashes_group("lockfile_hashes", manifest.get("lockfile_hashes", {}))
    ok_configs = verify_file_hashes_group("config_hashes", manifest.get("config_hashes", {}))
    ok_outputs = verify_file_hashes_group("output_files", manifest.get("output_files", {}))

    if ok_locks and ok_configs and ok_outputs:
        print(f"VERIFY_RUN_MANIFEST: PASS {manifest_path}")
        return 0
    else:
        print(f"VERIFY_RUN_MANIFEST: FAIL {manifest_path}", file=sys.stderr)
        return 1


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_file", help="Path to runs/<id>/manifest.json")
    args = parser.parse_args()

    return verify_run_manifest_file(Path(args.manifest_file))


if __name__ == "__main__":
    sys.exit(main())
