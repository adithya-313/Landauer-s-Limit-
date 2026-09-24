"""
colab_adapter.py
================
PHASE 5 — Colab T4 Cloud Engine Adapter (Best-Effort Tier).

Connects via httpx to an Ngrok tunnel endpoint that points to a Colab
T4 instance running an OpenAI-compatible server.  This is the cheapest
tier — no latency guarantees.

Key behaviours:
- Every request is logged with the [BEST-EFFORT CLOUD TIER] tag.
- Retry logic: up to 3 retries with exponential backoff (1 s, 2 s, 4 s)
  on connection drops.
- Parses server-sent events (SSE) like the vLLM adapter.
"""

import asyncio
import json
import logging
import random
from typing import AsyncIterator, Dict, Any

import httpx

logger = logging.getLogger("ColabAdapter")

# Default Ngrok tunnel URL (should be overridden via env var or config).
DEFAULT_TUNNEL_URL = "http://localhost:8888/v1"

# Retry parameters for best-effort recovery.
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0  # doubles each retry: 1s, 2s, 4s


class ColabAdapter:
    """
    Adapter for a Colab T4 instance behind an Ngrok tunnel.

    Parameters
    ----------
    tunnel_url : str
        The Ngrok tunnel base URL.
    """

    def __init__(
        self,
        tunnel_url: str = DEFAULT_TUNNEL_URL,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        self.tunnel_url = tunnel_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.AsyncClient(base_url=self.tunnel_url, timeout=300.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, tier: str = "free", max_tokens: int = 100) -> AsyncIterator[str]:
        # NOTE: This adapter does not currently use the tier parameter; it exists only to satisfy the shared BaseEngineAdapter contract.
        """
        Stream a completion from the Colab T4 instance.

        Retries up to 3 times with exponential backoff on connection errors.
        Logs every request with [BEST-EFFORT CLOUD TIER] for observability.
        """
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        last_exception = None

        for attempt in range(1, MAX_RETRIES + 1):
            logger.info(
                "[BEST-EFFORT CLOUD TIER] ColabAdapter attempt %d/%d — "
                "prompt=%.50s",
                attempt, MAX_RETRIES, prompt,
            )
            try:
                async with self._client.stream(
                    "POST", "/chat/completions", json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    # If we get here the stream completed successfully.
                    return

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exception = exc
                logger.warning(
                    "[BEST-EFFORT CLOUD TIER] Attempt %d failed: %s",
                    attempt, exc,
                )
                if attempt < MAX_RETRIES:
                    # Backoff formula: base_delay * 2^attempt + jitter
                    # base_delay = 1.0 seconds, max_attempts = 3
                    # Jitter is uniform random [0, 1) seconds added to each wait.
                    # Why jitter? If multiple requests all fail at the same moment and retry
                    # at the exact same time, they hammer the server together (thundering herd).
                    # Jitter staggers them so retries are spread out over time.
                    wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    logger.info(
                        "Retrying in %.2f seconds (base=%.1f + jitter) ...", wait, BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    )
                    await asyncio.sleep(wait)
                continue

            except httpx.HTTPStatusError as exc:
                # Non-retryable HTTP error.
                raise RuntimeError(
                    f"Colab adapter HTTP {exc.response.status_code}: {exc}"
                )

        # All retries exhausted.
        raise RuntimeError(
            f"Colab adapter: all {MAX_RETRIES} retries exhausted. "
            f"Last error: {last_exception}"
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Ping the tunnel endpoint.
        """
        try:
            resp = await self._client.get("/models", timeout=10.0)
            return {"status": "ok" if resp.status_code == 200 else "degraded"}
        except Exception as exc:
            return {"status": "degraded", "reason": str(exc)}

    async def close(self):
        """Release the httpx client session."""
        await self._client.aclose()
