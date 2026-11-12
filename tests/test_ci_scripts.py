"""
Parametrized test runner for legacy standalone verification scripts.

Runs standalone verification scripts in isolated subprocesses and asserts clean
exit code 0, formatting failure outputs to aid in debugging Phase 1 and Phase 6.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPTS = [
    ("validate.py", "phase1-dataset"),
    ("tests/test_phase6d_1_hashing.py", "phase6-kv-hashing"),
    ("tests/test_phase6d_2_blocks.py", "phase6-kv-blocks"),
    ("tests/test_phase6d_3_registry.py", "phase6-kv-registry"),
]


@pytest.mark.parametrize(
    ("script_rel_path", "phase_id"),
    SCRIPTS,
    ids=[item[1] for item in SCRIPTS],
)
def test_legacy_script(script_rel_path: str, phase_id: str) -> None:
    """Runs a standalone script as a subprocess and verifies it exits with code 0."""
    script_path = REPO_ROOT / script_rel_path
    assert script_path.is_file(), f"Script not found at {script_path}"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        msg = (
            f"Script {script_rel_path} ({phase_id}) failed with exit code {proc.returncode}.\n"
            f"--- STDOUT TAIL ---\n{stdout_tail}\n"
            f"--- STDERR TAIL ---\n{stderr_tail}\n"
        )
        pytest.fail(msg)

