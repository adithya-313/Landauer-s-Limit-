"""
llamacpp_adapter.py
===================
PHASE 5 — llama.cpp / Ollama Local Engine Adapter.

Connects to a local llama.cpp server or Ollama instance via httpx.
The default endpoint matches Ollama's /api/generate route, but the
same adapter can point at any llama.cpp-compatible server.

The adapter parses the streaming JSON-lines format that Ollama and
llama.cpp server emit, extracting the "response" field from each line.
"""

import json
import logging
from typing import AsyncIterator, Dict, Any

import httpx

logger = logging.getLogger("LlamaCppAdapter")

# Default endpoint for a local Ollama server.
DEFAULT_BASE_URL = "http://localhost:11434"


class LlamaCppAdapter:
    """
    Adapter for a local llama.cpp / Ollama server.

    Parameters
    ----------
    base_url : str
        The base URL of the Ollama or llama.cpp server
        (default http://localhost:11434).
    model_name : str
        The Ollama model tag to request. Must match exactly what
        `ollama list` shows. Default is "qwen2.5:1.5b" for consistency
        with the AWQ and FP16 tiers (project decision: all tiers use
        Qwen2.5 where a compatible checkpoint exists).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = "qwen2.5:1.5b",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, tier: str = "free", max_tokens: int = 100) -> AsyncIterator[str]:
        # NOTE: This adapter does not currently use the tier parameter; it exists only to satisfy the shared BaseEngineAdapter contract.
        """
        Stream a completion from the local llama.cpp / Ollama server.

        Steps:
          1. POST to /api/generate with streaming enabled.
          2. Read JSON lines from the response body.
          3. Extract the "response" field and yield it.
        """
        # The 'model' key must match the Ollama model tag exactly
        # (as shown by `ollama list`). Using qwen2.5:1.5b to align with
        # the Qwen2.5 family used in the AWQ and FP16 tiers — switched
        # from the previous hardcoded "llama3" per project decision.
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST", "/api/generate", json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        # Skip any malformed lines gracefully.
                        continue

        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as exc:
            logger.error("llama.cpp / Ollama connection error: %s", exc)
            raise RuntimeError(
                f"Engine llama.cpp unreachable at {self.base_url}: {exc}"
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Ping the server to confirm it is alive.
        """
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            if resp.status_code == 200:
                return {"status": "ok"}
            return {"status": "degraded", "reason": f"HTTP {resp.status_code}"}
        except Exception as exc:
            return {"status": "degraded", "reason": str(exc)}

    async def close(self):
        """Release the httpx client session."""
        await self._client.aclose()
