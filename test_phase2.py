"""
test_phase2.py
==============
Self-Validation script for the Phase 2 FastAPI Gateway.

This test:
1. Starts uvicorn in a background thread.
2. Waits for the server to be ready.
3. Sends a normal-sized request and expects HTTP 200 with the correct schema.
4. Sends a 5 MB garbage request and expects HTTP 413 Payload Too Large.
5. Cleans up the server thread.

Run it with:  python test_phase2.py
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
PORT = 9876
BASE_URL = f"http://{HOST}:{PORT}"
MAX_PAYLOAD = 65536  # Must match the constant in gateway.py


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


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Ping the /health endpoint until the server responds or we time out."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------

def test_normal_request():
    """
    Send a small, valid chat completion request.
    Expect: HTTP 200, correct OpenAI-style response body.
    """
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
        ],
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}: {response.text}"
    )

    body = response.json()
    assert body["object"] == "chat.completion", f"Unexpected object: {body.get('object')}"
    assert "id" in body, "Missing 'id' in response"
    assert "choices" in body, "Missing 'choices' in response"
    assert len(body["choices"]) > 0, "Empty choices list"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert len(body["choices"][0]["message"]["content"]) > 0

    print(f"[PASS] test_normal_request — 200 OK with valid schema")
    return True


def test_oversized_request():
    """
    Send a request with Content-Length exceeding 64 KB.
    Expect: HTTP 413 Payload Too Large, WITHOUT the server reading the body.
    """
    url = f"{BASE_URL}/v1/chat/completions"

    # Build a payload body that is roughly 5 MB of garbage JSON.
    huge_body = json.dumps({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "A" * (5 * 1024 * 1024)}  # 5 MB string
        ],
    })

    # Send with an explicit Content-Length header.
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(huge_body)),
    }

    response = requests.post(url, data=huge_body, headers=headers)

    assert response.status_code == 413, (
        f"Expected 413, got {response.status_code}: {response.text[:200]}"
    )

    body = response.json()
    assert "error" in body, "Missing 'error' in 413 response"
    assert "too large" in body["error"]["message"].lower(), (
        f"Unexpected error message: {body['error']['message']}"
    )

    print(f"[PASS] test_oversized_request — 413 Payload Too Large ({len(huge_body)} bytes sent)")
    return True


def test_missing_content_length():
    """
    Send a request WITHOUT a Content-Length header, using chunked
    transfer encoding. Expect: HTTP 411 Length Required.

    We use http.client's low-level `putrequest` / `putheader` API and
    manually send the body as chunks so that Content-Length is never set.
    """
    import http.client

    body_bytes = json.dumps({
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hi"}],
    }).encode("utf-8")

    # Build the request manually to avoid auto Content-Length injection.
    conn = http.client.HTTPConnection(HOST, PORT)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Host", f"{HOST}:{PORT}")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Transfer-Encoding", "chunked")
    # Note: NO Content-Length header.
    conn.endheaders()

    # Send body in chunked encoding format.
    chunk_size_hex = f"{len(body_bytes):x}"
    conn.send(chunk_size_hex.encode("utf-8"))
    conn.send(b"\r\n")
    conn.send(body_bytes)
    conn.send(b"\r\n")
    conn.send(b"0\r\n\r\n")  # Final zero-length chunk

    response = conn.getresponse()
    status = response.status
    body_text = response.read().decode("utf-8")
    conn.close()

    assert status == 411, (
        f"Expected 411, got {status}: {body_text[:200]}"
    )

    body = json.loads(body_text)
    assert "error" in body, "Missing 'error' in 411 response"

    print(f"[PASS] test_missing_content_length — 411 Length Required (chunked, no Content-Length)")
    return True


# ---------------------------------------------------------------------------
# MAIN: Boot server, run tests, shut down
# ---------------------------------------------------------------------------

def main():
    # Import the FastAPI app AFTER defining the test constants.
    from gateway import app

    print("=" * 60)
    print("PHASE 2 — GATEWAY SELF-VALIDATION")
    print("=" * 60)

    # Start the server in a background thread.
    print(f"[INFO] Starting server on {HOST}:{PORT} ...")
    server_thread = ServerThread(app, HOST, PORT)
    server_thread.start()

    # Wait for the server to become available.
    if not wait_for_server(BASE_URL):
        print("[FAIL] Server did not start in time.")
        sys.exit(1)
    print(f"[INFO] Server is ready.\n")

    # Track results.
    passed = 0
    failed = 0

    # Test 1: Normal request → 200
    try:
        test_normal_request()
        passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_normal_request: {e}")
        failed += 1
    except Exception as e:
        print(f"[FAIL] test_normal_request (unexpected): {e}")
        failed += 1

    # Test 2: Oversized request → 413
    try:
        test_oversized_request()
        passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_oversized_request: {e}")
        failed += 1
    except Exception as e:
        print(f"[FAIL] test_oversized_request (unexpected): {e}")
        failed += 1

    # Test 3: Missing Content-Length → 411
    try:
        test_missing_content_length()
        passed += 1
    except AssertionError as e:
        print(f"[FAIL] test_missing_content_length: {e}")
        failed += 1
    except Exception as e:
        print(f"[FAIL] test_missing_content_length (unexpected): {e}")
        failed += 1

    # Shut down the server.
    print(f"\n[INFO] Shutting down server ...")
    server_thread.shutdown()
    server_thread.join(timeout=5)

    # Final summary.
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL VALIDATIONS PASSED")


if __name__ == "__main__":
    main()
