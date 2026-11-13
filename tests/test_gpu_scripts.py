"""
Parametrized test runner for scripts requiring an NVIDIA GPU or live server.

Runs GPU-dependent and live-model-server benchmark/test scripts in isolated subprocesses.
All test cases are marked with @pytest.mark.gpu so they are excluded in CPU CI environments.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

GPU_SCRIPTS = [
    "test_phase6c_exhaustion.py",
    "tests/test_phase6d_4_resume_prefill.py",
    "tests/test_phase6d_5_engine_wiring.py",
    "tests/test_phase6d_6_benchmark.py",
    "tests/test_phase6d_6_integration_benchmark.py",
    "runtime_core/test_batch_engine.py",
    "test_phase5.py",
    "test_stage3.py",
    "test_stage3_error.py",
    "test_stage3_staggered.py",
    # Legacy live-server / GPU scripts
    "tests/test_block_reuse.py",
    "tests/test_http_tier_eviction.py",
    "tests/test_phase6c.py",
    "tests/test_phase6d_0_template.py",
    "tests/test_supplementary.py",
    "tests/test_gateway.py",
]


@pytest.mark.parametrize(
    "script_rel_path",
    [pytest.param(p, marks=pytest.mark.gpu, id=Path(p).stem) for p in GPU_SCRIPTS],
)
def test_gpu_script(script_rel_path: str) -> None:
    """Executes a GPU / live-server script as a subprocess and asserts returncode == 0."""
    script_path = REPO_ROOT / script_rel_path
    assert script_path.is_file(), f"Script not found at {script_path}"

    env = dict(os.environ)
    # Ensure runtime_core and repo root are on PYTHONPATH
    existing_pythonpath = env.get("PYTHONPATH", "")
    runtime_core_dir = str(REPO_ROOT / "runtime_core")
    repo_root_str = str(REPO_ROOT)
    env["PYTHONPATH"] = (
        f"{repo_root_str}{os.pathsep}{runtime_core_dir}{os.pathsep}{existing_pythonpath}".rstrip(os.pathsep)
    )

    proc = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )

    if proc.returncode != 0:
        stdout_tail = "\n".join(proc.stdout.splitlines()[-20:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        msg = (
            f"GPU script {script_rel_path} failed with exit code {proc.returncode}.\n"
            f"--- STDOUT TAIL ---\n{stdout_tail}\n"
            f"--- STDERR TAIL ---\n{stderr_tail}\n"
        )
        pytest.fail(msg)
