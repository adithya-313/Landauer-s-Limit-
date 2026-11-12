"""
Tests request payload size guardrails against the live gateway ASGI application.

Verifies that requests exceeding MAX_PAYLOAD_SIZE_BYTES receive HTTP 413 with error code 413,
and requests omitting the Content-Length header receive HTTP 411 Length Required.
"""

from __future__ import annotations

import os

from fastapi.testclient import TestClient
import pytest

import gateway
from payload_guard import MAX_PAYLOAD_SIZE_BYTES


@pytest.fixture
def client() -> TestClient:
    """Returns a TestClient instance wrapping the real gateway FastAPI application."""
    return TestClient(gateway.app)


def test_oversized_payload_returns_413(client: TestClient) -> None:
    """Verifies that a payload exceeding MAX_PAYLOAD_SIZE_BYTES returns HTTP 413 with error code 413."""
    api_key = os.environ.get("LANDAUER_API_KEY", "")
    assert api_key, "LANDAUER_API_KEY must be set by root conftest"

    # Build payload strictly larger than MAX_PAYLOAD_SIZE_BYTES
    oversized_data = b"x" * (MAX_PAYLOAD_SIZE_BYTES + 1)
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    response = client.post("/v1/chat/completions", content=oversized_data, headers=headers)
    assert response.status_code == 413
    error_payload = response.json()
    assert "error" in error_payload
    assert error_payload["error"].get("code") == 413


def test_missing_content_length_returns_411(client: TestClient) -> None:
    """Verifies that requests omitting Content-Length header return HTTP 411."""
    api_key = os.environ.get("LANDAUER_API_KEY", "")
    assert api_key, "LANDAUER_API_KEY must be set by root conftest"

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }

    # Streaming content via generator causes TestClient to omit Content-Length
    def stream_content():
        yield b'{"model": "test", "messages": [{"role": "user", "content": "ping"}]}'

    response = client.post("/v1/chat/completions", content=stream_content(), headers=headers)
    assert response.status_code == 411
    error_payload = response.json()
    assert "error" in error_payload
    assert error_payload["error"].get("code") == 411
