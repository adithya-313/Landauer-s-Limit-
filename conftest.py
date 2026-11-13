"""
Root test configuration and test environment bootstrapping.

Runs at import time before any test modules import gateway.py or adapters.
Generates an ephemeral test tenant and configures environment variables so that
all test runs remain fully isolated from host credentials and production logs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import secrets
import tempfile

# Create an isolated temporary directory for test tenants and telemetry events
_TMP_DIR = tempfile.TemporaryDirectory(prefix="landauer_ci_env_")
_TEMP_PATH = Path(_TMP_DIR.name)

# Generate an ephemeral API key for the test session
_RAW_KEY = secrets.token_urlsafe(32)
_KEY_HASH = hashlib.sha256(_RAW_KEY.encode("utf-8")).hexdigest()

# Write a deterministic test tenants.yaml (premium tier for local_dev)
_TENANTS_FILE = _TEMP_PATH / "tenants.yaml"
_TENANTS_FILE.write_text(
    f"tenants:\n"
    f"  - tenant_id: local_dev\n"
    f"    tier: premium\n"
    f"    key_sha256: {_KEY_HASH}\n",
    encoding="utf-8",
)

_EVENTS_FILE = _TEMP_PATH / "gateway_events.jsonl"

# Always set environment variables to point to the isolated test directory
# so tests never depend on the user's private key or overwrite production logs.
os.environ["LANDAUER_API_KEY"] = _RAW_KEY
os.environ["LANDAUER_TENANTS_PATH"] = str(_TENANTS_FILE)
os.environ["LANDAUER_EVENTS_PATH"] = str(_EVENTS_FILE)

# Ensure LANDAUER_PG_DSN is unset so tests do not connect to a real Postgres database
os.environ.pop("LANDAUER_PG_DSN", None)

