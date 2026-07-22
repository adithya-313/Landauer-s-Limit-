"""
test_phase4.py
==============
Self-Validation suite for Phase 4 — Semantic Cache.

Blueprint test scenarios (exact from the specification):

1. Single-turn MISS:
   POST /v1/chat/completions with {"messages": [{"role": "user",
   "content": "What is Python?"}]}
   -> assert X-Cache: MISS

2. Multi-turn MISS (first occurrence):
   POST /v1/chat/completions with [
     {"role": "user", "content": "What is Python?"},
     {"role": "assistant", "content": "A programming language"},
     {"role": "user", "content": "How do I install it?"}
   ]
   -> assert X-Cache: MISS

3. Multi-turn HIT (exact repeat of the multi-turn payload above):
   POST /v1/chat/completions with same payload as test 2
   -> assert X-Cache: HIT

Run with:  python -m pytest test_phase4.py -v
"""

import json
import time
import threading
import sys

import requests
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = 9874
BASE_URL = f"http://{HOST}:{PORT}"


# ---------------------------------------------------------------------------
# HELPER: Run uvicorn in a background daemon thread
# ---------------------------------------------------------------------------

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
# FIXTURE: Server lifecycle (used by all tests)
# ---------------------------------------------------------------------------

_server_thread = None


def setup_module():
    """Start the server once before all tests run."""
    global _server_thread
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


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def send_chat_completion(messages: list) -> requests.Response:
    """
    Helper to send a POST /v1/chat/completions request.

    Parameters
    ----------
    messages : list of dict
        The messages array (e.g. [{"role": "user", "content": "..."}])

    Returns
    -------
    requests.Response
    """
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "test-model",
        "messages": messages,
    }
    headers = {"Content-Type": "application/json"}
    return requests.post(url, json=payload, headers=headers)


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

class TestSemanticCache:
    """
    Semantic cache test suite.

    These tests are ORDER-DEPENDENT — test 3 relies on the cache being
    populated by test 2. pytest runs them in declaration order by default.
    """

    def test_1_single_turn_miss(self):
        """
        Send a single-turn query and verify it is a cache MISS.
        This is the first time we ask "What is Python?" so it cannot be cached.
        """
        messages = [
            {"role": "user", "content": "What is Python?"},
        ]

        response = send_chat_completion(messages)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:200]}"
        )

        cache_header = response.headers.get("X-Cache", "").upper()
        assert cache_header == "MISS", (
            f"Expected X-Cache: MISS, got: {cache_header}"
        )

        # Verify the response body has the expected shape.
        body = response.json()
        assert "choices" in body
        assert len(body["choices"]) > 0
        assert "Python" in body["choices"][0]["message"]["content"]

        print(f"[PASS] test_1_single_turn_miss — X-Cache: {cache_header}")

    def test_2_multi_turn_miss(self):
        """
        Send a multi-turn query and verify it is a cache MISS.
        The cache key includes the assistant context, so this is distinct
        from the single-turn query above.
        """
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language"},
            {"role": "user", "content": "How do I install it?"},
        ]

        response = send_chat_completion(messages)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:200]}"
        )

        cache_header = response.headers.get("X-Cache", "").upper()
        assert cache_header == "MISS", (
            f"Expected X-Cache: MISS, got: {cache_header}"
        )

        body = response.json()
        assert "choices" in body
        assert len(body["choices"]) > 0

        print(f"[PASS] test_2_multi_turn_miss — X-Cache: {cache_header}")

    def test_3_multi_turn_hit(self):
        """
        Repeat the EXACT same multi-turn payload as test_2.
        Because the cache was populated by test_2, this should return HIT.
        """
        messages = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "A programming language"},
            {"role": "user", "content": "How do I install it?"},
        ]

        response = send_chat_completion(messages)

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text[:200]}"
        )

        cache_header = response.headers.get("X-Cache", "").upper()
        assert cache_header == "HIT", (
            f"Expected X-Cache: HIT, got: {cache_header}"
        )

        # On a HIT, the cached response should match the original response.
        body = response.json()
        assert "choices" in body
        assert len(body["choices"]) > 0
        # The response text should be the same mock text from test_2.
        assert "install" in body["choices"][0]["message"]["content"].lower()

        print(f"[PASS] test_3_multi_turn_hit — X-Cache: {cache_header}")
