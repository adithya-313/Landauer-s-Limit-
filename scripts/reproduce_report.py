"""Reproduce Phase 8 report and verify byte-for-byte fidelity."""

import argparse
import difflib
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path


def verify_checksums(input_dir: Path, sums_file: Path) -> tuple[bool, str]:
    """Verify SHA256SUMS file against files in input_dir."""
    if not sums_file.is_file():
        return False, f"Checksums file not found: {sums_file}"

    with open(sums_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        return False, f"Checksums file is empty: {sums_file}"

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            return False, f"Malformed line {line_num} in {sums_file}: {line}"
        expected_hash, rel_path = parts
        rel_path = rel_path.strip()
        target_path = input_dir / rel_path
        if not target_path.is_file():
            return False, f"Referenced file missing: {target_path} (from {sums_file}:{line_num})"

        hasher = hashlib.sha256()
        with open(target_path, "rb") as fh:
            while chunk := fh.read(1024 * 1024):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        if actual_hash.lower() != expected_hash.lower():
            return (
                False,
                f"Checksum mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}",
            )

    return True, ""


def reproduce(input_dir: Path, expected_report: Path) -> int:
    """Run reproduction and compare output against expected report."""
    sums_file = input_dir / "SHA256SUMS"
    valid, err_msg = verify_checksums(input_dir, sums_file)
    if not valid:
        sys.stderr.write(f"REPRODUCE_REPORT: FAIL (checksum verification failed: {err_msg})\n")
        return 1

    if not expected_report.is_file():
        sys.stderr.write(f"REPRODUCE_REPORT: FAIL (expected report missing: {expected_report})\n")
        return 1

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        cmd = [
            sys.executable,
            "generate_report.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(
                f"REPRODUCE_REPORT: FAIL (generate_report.py exited {proc.returncode}):\n{proc.stderr}\n"
            )
            return proc.returncode

        gen_md = out_dir / "report.md"
        if not gen_md.is_file():
            sys.stderr.write("REPRODUCE_REPORT: FAIL (generated report.md not found)\n")
            return 1

        with open(expected_report, "rb") as fh:
            expected_bytes = fh.read()
        with open(gen_md, "rb") as fh:
            actual_bytes = fh.read()

        if expected_bytes == actual_bytes:
            print("REPRODUCE_REPORT: PASS (exact byte match against committed report.md)")
            return 0

        # Mismatch: show diff
        expected_lines = expected_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
        actual_lines = actual_bytes.decode("utf-8", errors="replace").splitlines(keepends=True)
        diff = difflib.unified_diff(
            expected_lines,
            actual_lines,
            fromfile=str(expected_report),
            tofile=str(gen_md),
        )
        sys.stderr.write("REPRODUCE_REPORT: FAIL (report content mismatch):\n")
        sys.stderr.writelines(diff)
        return 1


def main() -> None:
    """CLI entry point for reproduce_report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="archive/phase8",
        help="Directory containing archived inputs (default: archive/phase8)",
    )
    parser.add_argument(
        "--expected-report",
        default="report.md",
        help="Path to expected report file (default: report.md)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    expected_report = Path(args.expected_report)
    ret = reproduce(input_dir, expected_report)
    sys.exit(ret)


if __name__ == "__main__":
    main()
