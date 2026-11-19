"""Tests for scripts/reproduce_report.py and generate_report.py --output-dir."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.reproduce_report import reproduce, verify_checksums


def test_reproduce_report_clean_run():
    """Test 1: scripts/reproduce_report.py runs clean and reproduces committed report.md byte-for-byte."""
    input_dir = Path("archive/phase8")
    expected_report = Path("report.md")
    assert input_dir.is_dir()
    assert expected_report.is_file()

    # Test via module function
    ret = reproduce(input_dir, expected_report)
    assert ret == 0

    # Test via CLI subprocess
    res = subprocess.run(
        [sys.executable, "scripts/reproduce_report.py", "--input-dir", str(input_dir), "--expected-report", str(expected_report)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "REPRODUCE_REPORT: PASS (exact byte match against committed report.md)" in res.stdout


def test_reproduce_report_detects_corrupted_archive(tmp_path):
    """Test 2: sha256 verification catches corrupted archive file (flip 1 byte in copy)."""
    archive_copy = tmp_path / "phase8"
    shutil.copytree("archive/phase8", archive_copy)

    target_file = archive_copy / "batch_events.jsonl"
    with open(target_file, "rb") as f:
        data = bytearray(f.read())
    assert len(data) > 0
    # Flip the first byte
    data[0] ^= 0xFF
    with open(target_file, "wb") as f:
        f.write(data)

    sums_file = archive_copy / "SHA256SUMS"
    valid, err_msg = verify_checksums(archive_copy, sums_file)
    assert not valid
    assert "Checksum mismatch for batch_events.jsonl" in err_msg

    ret = reproduce(archive_copy, Path("report.md"))
    assert ret != 0


def test_generate_report_output_dir_does_not_modify_root(tmp_path):
    """Test 3: --output-dir correctly routes output without modifying root report.md."""
    root_report = Path("report.md")
    with open(root_report, "rb") as f:
        original_bytes = f.read()
    orig_mtime = root_report.stat().st_mtime

    out_dir = tmp_path / "custom_output"
    res = subprocess.run(
        [sys.executable, "generate_report.py", "--input-dir", "archive/phase8", "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "report.html").is_file()

    with open(out_dir / "report.md", "rb") as f:
        generated_bytes = f.read()
    assert generated_bytes == original_bytes

    # Assert root report.md was not modified
    with open(root_report, "rb") as f:
        current_bytes = f.read()
    assert current_bytes == original_bytes
    assert root_report.stat().st_mtime == orig_mtime


def test_reproduce_report_fails_on_corrupted_expected_report(tmp_path):
    """Test 4: reproduce() and CLI fail if expected report differs by even one byte."""
    corrupted_report = tmp_path / "report.md"
    with open("report.md", "rb") as f:
        data = bytearray(f.read())
    assert len(data) > 0
    # Change ONE byte
    data[0] ^= 0xFF
    with open(corrupted_report, "wb") as f:
        f.write(data)

    input_dir = Path("archive/phase8")
    # Assert module function returns 1
    ret = reproduce(input_dir, corrupted_report)
    assert ret == 1

    # Assert CLI with --expected-report exits non-zero
    res = subprocess.run(
        [
            sys.executable,
            "scripts/reproduce_report.py",
            "--input-dir",
            str(input_dir),
            "--expected-report",
            str(corrupted_report),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode != 0
    assert "REPRODUCE_REPORT: FAIL (report content mismatch)" in res.stderr

