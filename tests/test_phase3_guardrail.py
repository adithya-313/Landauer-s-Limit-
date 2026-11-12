"""
Tests for Phase 3 and Phase 7a guardrail behavior against the live gateway ASGI app.

Docstring: this tests the built Phase 7a behaviour (timeout = fail-closed 429; other errors = fail-open).
- On timeout, the guardrail fails closed: returns HTTP 429 and logs a reject_request action.
- On unexpected exception, the guardrail fails open: returns HTTP 200 with X-Guardrail-Degraded: true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from fastapi.testclient import TestClient
import pytest

import gateway
import guardrail
import protective_actions


@pytest.fixture
def client() -> TestClient:
    """Returns a TestClient instance wrapping the real gateway FastAPI application."""
    return TestClient(gateway.app)


@pytest.fixture(autouse=True)
def isolate_actions_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Monkeypatches ACTIONS_LOG in protective_actions to an isolated temporary file."""
    temp_actions = tmp_path / "actions_taken.jsonl"
    monkeypatch.setattr(protective_actions, "ACTIONS_LOG", temp_actions)
    return temp_actions


def test_guardrail_timeout_fails_closed(client: TestClient, monkeypatch: pytest.MonkeyPatch, isolate_actions_log: Path) -> None:
    """Verifies that guardrail check timeout fails closed with 429 and logs reject_request."""
    api_key = os.environ.get("LANDAUER_API_KEY", "")
    assert api_key, "LANDAUER_API_KEY must be set by root conftest"

    # Set timeout to 0.05s and make check sleep 0.2s deterministically
    monkeypatch.setattr(guardrail, "GUARDRAIL_TIMEOUT", 0.05)

    def slow_check(prompt: str) -> tuple[bool, float]:
        time.sleep(0.2)
        return False, 0.0

    monkeypatch.setattr(gateway._guardrail_instance, "check", slow_check)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 429

    assert isolate_actions_log.is_file(), "actions_taken.jsonl was not created"
    actions = [
        json.loads(line)
        for line in isolate_actions_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reject_actions = [a for a in actions if a.get("action") == "reject_request"]
    assert len(reject_actions) >= 1, f"Expected reject_request action in {actions}"


def test_guardrail_unexpected_error_fails_open(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that unexpected guardrail error fails open with 200 and X-Guardrail-Degraded: true."""
    api_key = os.environ.get("LANDAUER_API_KEY", "")
    assert api_key, "LANDAUER_API_KEY must be set by root conftest"

    def broken_check(prompt: str) -> tuple[bool, float]:
        raise RuntimeError("Simulated unexpected guardrail crash")

    monkeypatch.setattr(gateway._guardrail_instance, "check", broken_check)

    # Mock semantic cache hit so fail-open returns immediately without blocking on background queue worker
    async def fake_check_cache(messages: list) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "cached response"}}]}

    monkeypatch.setattr(gateway._cache_instance, "check_cache", fake_check_cache)

    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
    }
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    response = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.headers.get("X-Guardrail-Degraded") == "true"
