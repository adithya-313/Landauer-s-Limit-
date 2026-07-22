"""
test_phase4.py
==============
Phase 4 v3 — Deterministic Semantic Cache Test Suite.

These tests verify the dual-lock cache mechanism:

  Test 1 — Version Collision Defense:
    "install Python 3.14" should NOT match "install Python 3.15"
    because the lock strings differ (python|3.14||install vs python|3.15||install).

  Test 2 — Coreference (Pronoun) Defense:
    Query "How to install it?" with history ["What is Docker?"] should NOT
    match the same query with history ["What is Java?"] because the
    historical entity extraction resolves to different entities.

  Test 3 — The Unknown Entity Bypass (Safety First):
    "How to install Rust?" — "rust" is NOT in the seed entity list, so
    generate_lock must return None, bypassing the cache entirely.

We mock the embedding function to return deterministic dummy vectors so
that the vector similarity test passes deterministically (we control the
cosine distance by controlling where the vectors point).
"""

import time
import threading
import sys
from typing import List, Dict, Optional
from collections import OrderedDict

import numpy as np
import pytest
import requests
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

from guardrail import EMBEDDING_DIM


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 9873
BASE_URL = f"http://{HOST}:{PORT}"


# ---------------------------------------------------------------------------
# MOCK EMBEDDING FUNCTION
# ---------------------------------------------------------------------------
# We generate a deterministic unit vector for each unique text so that
# identical texts produce identical vectors (cosine sim = 1.0) and different
# texts produce near-orthogonal vectors (cosine sim ~ 0.0).
#
# We use Python's built-in hash to seed numpy's random generator, which
# gives us deterministic, reproducible vectors across test runs.
# ---------------------------------------------------------------------------

_EMBEDDING_CACHE: OrderedDict = OrderedDict()


def mock_embed_text(text: str) -> np.ndarray:
    """
    Deterministic mock embedding: same text → same unit vector.

    This lets us reliably test the cache without loading the real ONNX model.
    Identical texts produce cosine similarity 1.0; different texts produce
    near-zero similarity.
    """
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    # Seed numpy's RNG with the hash of the text for reproducibility.
    seed = hash(text) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(1, EMBEDDING_DIM).astype(np.float32)

    # L2-normalise so inner product = cosine similarity.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    _EMBEDDING_CACHE[text] = vec
    return vec


# ---------------------------------------------------------------------------
# HELPER: Override the semantic_cache module before importing gateway
# ---------------------------------------------------------------------------

def patch_cache_with_mock():
    """
    Replace the real DeterministicSemanticCache's embedding_function with
    mock_embed_text so tests do not depend on the ONNX model.
    """
    import semantic_cache as sc
    sc.semantic_cache.embedding_function = mock_embed_text
    sc.semantic_cache.clear()


# ---------------------------------------------------------------------------
# SERVER LIFECYCLE
# ---------------------------------------------------------------------------

_server_thread = None


def setup_module():
    """Start the server once before all tests run."""
    global _server_thread

    # Patch the cache before importing gateway (gateway imports semantic_cache
    # at module level).
    patch_cache_with_mock()

    from gateway import app

    print(f"[INFO] Starting server on {HOST}:{PORT} ...")
    _server_thread = ServerThread(app, HOST, PORT)
    _server_thread.start()
    if not wait_for_server(BASE_URL):
        print("[FATAL] Server did not start in time.")
        sys.exit(1)
    print("[INFO] Server is ready.\n")


def teardown_module():
    """Shut down the server after all tests finish."""
    global _server_thread
    if _server_thread:
        print(f"\n[INFO] Shutting down server ...")
        _server_thread.shutdown()
        _server_thread.join(timeout=5)


class ServerThread(threading.Thread):
    """A daemon thread that runs the uvicorn server."""

    def __init__(self, app, host: str, port: int):
        super().__init__(daemon=True)
        self.server = Server(Config(app=app, host=host, port=port, log_level="error"))

    def run(self):
        self.server.run()

    def shutdown(self):
        self.server.should_exit = True


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Ping the /health endpoint until the server responds or we time out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# TEST HELPERS
# ---------------------------------------------------------------------------

def send_chat_completion(messages: list) -> requests.Response:
    """Helper to POST /v1/chat/completions."""
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {"model": "test-model", "messages": messages}
    headers = {"Content-Type": "application/json"}
    return requests.post(url, json=payload, headers=headers)


def get_cache_header(response: requests.Response) -> str:
    """Extract the X-Cache header (upper-cased) or return an empty string."""
    return response.headers.get("X-Cache", "").upper()


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

class TestDeterministicSemanticCache:
    """
    Dual-lock semantic cache tests.

    These tests are ORDER-DEPENDENT — they rely on the cache being populated
    by earlier tests. pytest runs them in declaration order by default.
    """

    def test_1_version_collision_defense(self):
        """
        Insert: "How to install Python 3.14" (Cache Miss, saves to cache).
        Request: "How to install Python 3.15".
        Assert: Cache Miss — lock "python|3.14||install" != "python|3.15||install".
        """
        # Step 1: Send the first request — should be a cache MISS.
        messages_1 = [
            {"role": "user", "content": "How to install Python 3.14"},
        ]
        response_1 = send_chat_completion(messages_1)
        assert response_1.status_code == 200
        header_1 = get_cache_header(response_1)
        assert header_1 == "MISS", (
            f"Expected MISS on first insert, got {header_1}"
        )
        print(f"  [STEP 1] First request (Python 3.14): X-Cache={header_1}")

        # Step 2: Send a request with a different version.
        # The lock for this will be "python|3.15||install", which does NOT
        # match "python|3.14||install" from step 1.
        messages_2 = [
            {"role": "user", "content": "How to install Python 3.15"},
        ]
        response_2 = send_chat_completion(messages_2)
        assert response_2.status_code == 200
        header_2 = get_cache_header(response_2)

        # Assert: Must be a Cache Miss because the version differs.
        assert header_2 == "MISS", (
            f"Expected MISS for version 3.15 (different lock), "
            f"got {header_2}"
        )
        print(f"  [STEP 2] Second request (Python 3.15): X-Cache={header_2}")
        print(f"  [PASS] test_1_version_collision_defense")

    def test_2_coreference_defense(self):
        """
        Insert: Context ["What is Docker?"], Query ["How to install it?"].
        Request: Context ["What is Java?"], Query ["How to install it?"].
        Assert: Cache Miss — historical entity extraction resolves to
        "docker" vs "java", producing different locks.
        """
        # Step 1: Insert with Docker context.
        messages_1 = [
            {"role": "user", "content": "What is Docker?"},
            {"role": "assistant", "content": "Docker is a container platform."},
            {"role": "user", "content": "How to install it?"},
        ]
        response_1 = send_chat_completion(messages_1)
        assert response_1.status_code == 200
        header_1 = get_cache_header(response_1)
        assert header_1 == "MISS", (
            f"Expected MISS on first insert (Docker), got {header_1}"
        )
        print(f"  [STEP 1] Docker context: X-Cache={header_1}")

        # Step 2: Same pronoun query but with Java context.
        # The lock should resolve to "java" (from history), not "docker".
        messages_2 = [
            {"role": "user", "content": "What is Java?"},
            {"role": "assistant", "content": "Java is a programming language."},
            {"role": "user", "content": "How to install it?"},
        ]
        response_2 = send_chat_completion(messages_2)
        assert response_2.status_code == 200
        header_2 = get_cache_header(response_2)

        # Assert: Must be a Cache Miss because the historical entity is
        # "java", which produces a different lock from "docker".
        assert header_2 == "MISS", (
            f"Expected MISS for Java context (different lock), "
            f"got {header_2}"
        )
        print(f"  [STEP 2] Java context: X-Cache={header_2}")
        print(f"  [PASS] test_2_coreference_defense")

    def test_3_unknown_entity_bypass(self):
        """
        Request: "How to install Rust?".
        Assert: Cache Miss — "Rust" is not in the seed_entities list, so
        generate_lock must return None, bypassing the cache entirely.
        """
        messages = [
            {"role": "user", "content": "How to install Rust?"},
        ]
        response = send_chat_completion(messages)
        assert response.status_code == 200
        header = get_cache_header(response)

        # Assert: Must be a Cache Miss because "rust" is unknown.
        assert header == "MISS", (
            f"Expected MISS for unknown entity 'Rust', got {header}"
        )
        print(f"  [STEP 1] Unknown entity (Rust): X-Cache={header}")
        print(f"  [PASS] test_3_unknown_entity_bypass")
