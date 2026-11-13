"""
Tests configuration validation at gateway startup.

Verifies that malformed engine configurations cause gateway import to fail fast
with a clear error message in stderr, while the committed config.yaml imports cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_malformed_config_fails_startup(tmp_path: Path) -> None:
    """Verifies that an engine with type vllm and no url causes startup to fail loudly."""
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text(
        "engines:\n"
        "  - id: bad_vllm\n"
        "    type: vllm\n"
        "    tier: free\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env["LANDAUER_CONFIG_PATH"] = str(bad_config)

    proc = subprocess.run(
        [sys.executable, "-c", "import gateway"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode != 0, "Expected non-zero exit with malformed config"
    assert "Invalid engine entry" in proc.stderr, (
        f"Expected 'Invalid engine entry' in stderr, got:\n{proc.stderr}"
    )


def test_committed_config_starts_cleanly() -> None:
    """Positive control: importing gateway with committed config.yaml exits 0."""
    committed_config = REPO_ROOT / "config.yaml"
    assert committed_config.is_file(), "config.yaml not found at repo root"

    env = dict(os.environ)
    env["LANDAUER_CONFIG_PATH"] = str(committed_config)

    proc = subprocess.run(
        [sys.executable, "-c", "import gateway"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, (
        f"Committed config failed to import gateway (exit {proc.returncode}):\n{proc.stderr}"
    )
