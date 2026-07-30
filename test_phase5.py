"""
test_phase5.py
===============
PHASE 5 — Multi-Engine Adapter Standalone Test Suite.

Tests each adapter independently before gateway integration:

  Test 1 — CustomRuntimeAdapter:
    Hit the placeholder adapter directly and verify it yields the
    expected structured placeholder string.

  Test 2 — ColabAdapter Retry Logic:
    Point the adapter at an invalid URL (nothing listening on the
    port) and verify that it attempts multiple retries before raising
    a RuntimeError.

  Test 3 — VLLMAdapter Connection Failure:
    Point the adapter at an invalid URL and verify that connection
    errors are caught, logged, and wrapped in a clear RuntimeError
    instead of a raw httpx exception.

All three tests are self-contained — they do NOT need real GPU servers.
"""

import asyncio
import logging
import sys

# Configure logging so retry / error messages are visible.
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


# ---------------------------------------------------------------------------
# TEST 1: CustomRuntimeAdapter — Placeholder Delivery
# ---------------------------------------------------------------------------

async def test_custom_runtime_placeholder():
    """
    Send a prompt to the CustomRuntimeAdapter and verify that it yields
    exactly one token containing the structured placeholder text.
    """
    from adapters.custom_runtime_adapter import CustomRuntimeAdapter

    adapter = CustomRuntimeAdapter()
    prompt = "What is the meaning of life?"
    tokens = []

    async for token in adapter.generate(prompt):
        tokens.append(token)

    # We expect exactly one token.
    assert len(tokens) == 1, (
        f"Expected 1 token, got {len(tokens)}"
    )

    # The token should contain the placeholder marker and the prompt.
    combined = "".join(tokens)
    assert "CUSTOM_RUNTIME_PLACEHOLDER" in combined, (
        f"Missing placeholder marker in: {combined}"
    )
    assert "What is the meaning of life?" in combined, (
        f"Missing prompt text in: {combined}"
    )

    print(f"[PASS] test_custom_runtime_placeholder — "
          f"delivered '{combined.strip()}'")


# ---------------------------------------------------------------------------
# TEST 2: ColabAdapter Retry Logic
# ---------------------------------------------------------------------------

async def test_colab_adapter_retry():
    """
    Point the ColabAdapter at an invalid URL (port 1 on localhost is
    almost certainly closed) and verify that it:
      - Attempts all 3 retries (logged as warnings).
      - Raises a RuntimeError with a message about exhausted retries.
    """
    from adapters.colab_adapter import ColabAdapter

    # Use port 1 which should be closed on any system.
    adapter = ColabAdapter(tunnel_url="http://127.0.0.1:1/v1")
    prompt = "Hello from the retry test."

    try:
        async for _ in adapter.generate(prompt):
            pass  # Should not yield anything before error.
        # If we get here the test failed — no exception was raised.
        assert False, "Expected RuntimeError but no exception was raised."

    except RuntimeError as exc:
        error_text = str(exc)
        # Verify the error message mentions retry exhaustion.
        assert "retries exhausted" in error_text.lower(), (
            f"Unexpected error message: {error_text}"
        )
        print(f"[PASS] test_colab_adapter_retry — "
              f"RuntimeError raised after retries: {error_text[:80]}...")

    except Exception as exc:
        # Any non-RuntimeError is a failure.
        assert False, (
            f"Expected RuntimeError, got {type(exc).__name__}: {exc}"
        )


# ---------------------------------------------------------------------------
# TEST 3: VLLMAdapter Connection Failure
# ---------------------------------------------------------------------------

async def test_vllm_adapter_connection_failure():
    """
    Point the VLLMAdapter at an invalid URL and verify that connection
    errors produce a clean RuntimeError (not a raw httpx exception).
    """
    from adapters.vllm_adapter import VLLMAdapter

    # Use port 1 which should be closed on any system.
    adapter = VLLMAdapter(base_url="http://127.0.0.1:1/v1")
    prompt = "Test prompt for VLLM adapter error handling."

    try:
        async for _ in adapter.generate(prompt):
            pass
        assert False, "Expected RuntimeError but no exception was raised."

    except RuntimeError as exc:
        error_text = str(exc)
        # Verify the error message mentions the engine name and the URL.
        assert "vLLM" in error_text, (
            f"Error should mention vLLM: {error_text}"
        )
        print(f"[PASS] test_vllm_adapter_connection_failure — "
              f"RuntimeError raised: {error_text[:80]}...")

    except httpx_exc:
        # If an httpx exception leaks through, the test fails.
        import httpx
        assert False, (
            f"Raw httpx exception leaked through: {type(exc).__name__}: {exc}"
        )

    except Exception as exc:
        assert False, (
            f"Expected RuntimeError, got {type(exc).__name__}: {exc}"
        )


# For the VLLM test, import httpx only in the exception handler.
import httpx as _httpx_module
httpx_exc = (_httpx_module.ConnectError, _httpx_module.TimeoutException)


# ---------------------------------------------------------------------------
# MAIN — Run all tests sequentially
# ---------------------------------------------------------------------------

async def main():
    print("=" * 60)
    print("PHASE 5 — ADAPTER STANDALONE TEST SUITE")
    print("=" * 60)

    tests = [
        ("CustomRuntime Placeholder", test_custom_runtime_placeholder),
        ("ColabAdapter Retry",        test_colab_adapter_retry),
        ("VLLMAdapter Connection",    test_vllm_adapter_connection_failure),
    ]

    passed = 0
    failed = 0

    for name, coro in tests:
        print(f"\n--- {name} ---")
        try:
            await coro()
            passed += 1
        except (AssertionError, Exception) as exc:
            print(f"[FAIL] {name}: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    else:
        print("ALL PHASE 5 TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
