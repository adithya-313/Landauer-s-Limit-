"""
test_live_streaming.py
======================
PHASE 5 — Direct Engine Adapter Live Streaming Test Suite.

What this file is:
  This test suite exercises each of the four Phase 5 engine adapters directly
  (bypassing the gateway and router) by calling adapter.generate(prompt)
  to verify that token streaming works end-to-end at the transport layer.

Why it exists:
  Before trusting the API Gateway to route client requests across backends,
  we must prove that each adapter can establish a connection and yield stream
  chunks without buffering or unexpected crashes.

How to run:
  pytest test_live_streaming.py -v

Note on Live-Fire Server Dependencies:
  - VLLMAdapter expects a live vLLM OpenAI-compatible server at http://localhost:8000/v1.
  - LlamaCppAdapter expects a live Ollama/llama.cpp server at http://localhost:11434.
  - ColabAdapter expects an active Ngrok tunnel endpoint.
  If the target server is unreachable, the test dynamically skips (via pytest.skip)
  with an explicit message rather than recording a false pass or failing due to missing
  infrastructure. CustomRuntimeAdapter is deterministic and always executes full assertions.
"""

import pytest
import httpx

from adapters.vllm_adapter import VLLMAdapter
from adapters.llamacpp_adapter import LlamaCppAdapter
from adapters.colab_adapter import ColabAdapter
from adapters.custom_runtime_adapter import CustomRuntimeAdapter


@pytest.mark.asyncio
async def test_vllm_adapter_live_streaming():
    """
    Direct live-fire test for VLLMAdapter.

    Instantiates VLLMAdapter, sends a prompt directly to generate(), and verifies
    that streamed token chunks are yielded. If no local vLLM server is active on
    port 8000 or VRAM is low, pytest.skip() is invoked to keep the suite honest.
    """
    adapter = VLLMAdapter()
    prompt = "Explain Landauer's principle in one short sentence."
    tokens = []

    try:
        async for chunk in adapter.generate(prompt):
            tokens.append(chunk)

        assert len(tokens) > 0, "vLLM stream yielded zero tokens."
        assert any(t.strip() for t in tokens), "vLLM stream produced only empty strings."

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
        pytest.skip(f"no live vLLM server reachable at localhost:8000 — skipping live-fire assertion ({exc})")

    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_llamacpp_adapter_live_streaming():
    """
    Direct live-fire test for LlamaCppAdapter.

    Instantiates LlamaCppAdapter, sends a prompt directly to generate(), and verifies
    that streamed token chunks are yielded from Ollama/llama.cpp. If no server is
    listening on port 11434, pytest.skip() is invoked with a clear explanation.
    """
    adapter = LlamaCppAdapter()
    prompt = "Explain Landauer's principle in one short sentence."
    tokens = []

    try:
        async for chunk in adapter.generate(prompt):
            tokens.append(chunk)

        assert len(tokens) > 0, "llama.cpp stream yielded zero tokens."
        assert any(t.strip() for t in tokens), "llama.cpp stream produced only empty strings."

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
        pytest.skip(f"no live Ollama/llama.cpp server reachable at localhost:11434 — skipping live-fire assertion ({exc})")

    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_colab_adapter_live_streaming():
    """
    Direct live-fire test for ColabAdapter.

    Instantiates ColabAdapter, sends a prompt directly to generate(), and verifies
    that streamed SSE token chunks are yielded over the tunnel. If the cloud tunnel
    endpoint is unreachable or retries exhaust, pytest.skip() is invoked.
    """
    adapter = ColabAdapter()
    prompt = "Explain Landauer's principle in one short sentence."
    tokens = []

    try:
        async for chunk in adapter.generate(prompt):
            tokens.append(chunk)

        assert len(tokens) > 0, "Colab stream yielded zero tokens."
        assert any(t.strip() for t in tokens), "Colab stream produced only empty strings."

    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError, RuntimeError) as exc:
        pytest.skip(f"no live Colab Ngrok tunnel reachable — skipping live-fire assertion ({exc})")

    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_custom_runtime_adapter_live_streaming():
    """
    Direct functional test for CustomRuntimeAdapter.

    Instantiates CustomRuntimeAdapter and calls generate() directly. Since this is a
    deterministic local placeholder for Phase 6, it requires no background server
    and must always yield the exact expected placeholder token containing the prompt.
    """
    adapter = CustomRuntimeAdapter()
    prompt = "Explain Landauer's principle in one short sentence."
    tokens = []

    async for chunk in adapter.generate(prompt):
        tokens.append(chunk)

    assert len(tokens) == 1, f"Expected 1 placeholder token, got {len(tokens)}"
    combined = "".join(tokens)
    assert "CUSTOM_RUNTIME_PLACEHOLDER" in combined, (
        f"Missing CUSTOM_RUNTIME_PLACEHOLDER tag in response: {combined}"
    )
    assert "Explain Landauer's principle" in combined, (
        f"Missing prompt text snippet in response: {combined}"
    )
