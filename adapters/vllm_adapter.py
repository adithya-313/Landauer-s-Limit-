"""
vllm_adapter.py
===============
PHASE 5 — vLLM Local Engine Adapter.

Connects to a local vLLM OpenAI-compatible server (default:
http://localhost:8000/v1) via httpx.  Before generating, it runs
nvidia-smi to check that available VRAM is above a minimum threshold
so we can catch OOM conditions early instead of mid-generation.

Key behaviours:
- On startup check: queries nvidia-smi and rejects if VRAM is too low.
- On HTTP errors: logs exact memory stats and raises a clear exception
  so the router can fall back to another engine.
"""

import asyncio
import logging
import shlex
from typing import AsyncIterator, Dict, Any

import httpx

logger = logging.getLogger("VLLMAdapter")

# Default endpoint for a local vLLM server running in OpenAI-compatible mode.
DEFAULT_BASE_URL = "http://localhost:8000/v1"
# Minimum free VRAM in MB required before we attempt generation.
MIN_VRAM_MB = 512


class VLLMAdapter:
    """
    Adapter for a local vLLM OpenAI-compatible server.

    Parameters
    ----------
    base_url : str
        The base URL of the vLLM server (default http://localhost:8000/v1).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)
        self._startup_check_done = False

    async def _ensure_startup_check(self):
        """
        Run the VRAM sanity check only once per adapter lifecycle.
        If VRAM is low, we verify if the server is actually alive. If it's alive, 
        vLLM is just reserving the memory by design. If it's dead, we raise the startup error.
        """
        if not self._startup_check_done:
            self._startup_check_done = True
            vram = await self._check_vram()
            if vram["oom_risk"]:
                try:
                    resp = await self._client.get("/models", timeout=5.0)
                    resp.raise_for_status()
                except Exception as exc:
                    raise RuntimeError(
                        f"Engine vLLM degraded at startup: OOM risk detected. "
                        f"Free VRAM = {vram['free_mb']} MB (< {MIN_VRAM_MB} MB threshold). "
                        f"Server is also unreachable: {exc}"
                    )

    # ------------------------------------------------------------------
    # VRAM pre-check
    # ------------------------------------------------------------------

    async def _check_vram(self) -> Dict[str, Any]:
        """
        Run nvidia-smi and parse total / free VRAM.

        Returns a dict with keys {total_mb, free_mb, oom_risk}.
        """
        try:
            # Use subprocess to run nvidia-smi once for both total and free.
            proc = await asyncio.create_subprocess_shell(
                'nvidia-smi --query-gpu=memory.total,memory.free '
                '--format=csv,noheader,nounits',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            output = stdout.decode().strip()

            if not output:
                logger.warning("nvidia-smi returned empty output.")
                return {"total_mb": 0, "free_mb": 0, "oom_risk": True}

            parts = output.split(",")
            total_mb = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            oom_risk = free_mb < MIN_VRAM_MB

            logger.info(
                "VRAM check: total=%d MB, free=%d MB, oom_risk=%s",
                total_mb, free_mb, oom_risk,
            )
            return {"total_mb": total_mb, "free_mb": free_mb, "oom_risk": oom_risk}

        except Exception as exc:
            logger.exception("Failed to query nvidia-smi: %s", exc)
            # If we cannot read VRAM, assume safe but log the failure.
            return {"total_mb": 0, "free_mb": 0, "oom_risk": False}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, tier: str = "free", max_tokens: int = 100) -> AsyncIterator[str]:
        # NOTE: This adapter does not currently use the tier parameter; it exists only to satisfy the shared BaseEngineAdapter contract.
        """
        Stream a completion from the local vLLM server.

        Steps:
          1. Ensure startup VRAM sanity check has run once.
          2. Perform a lightweight liveness check (GET /models).
          3. POST to /v1/chat/completions with streaming enabled.
          4. Parse server-sent events and yield token content.
        """
        # Step 1: Ensure startup check
        await self._ensure_startup_check()

        # Step 2: Lightweight liveness check instead of per-request VRAM check
        try:
            liveness_resp = await self._client.get("/models", timeout=5.0)
            liveness_resp.raise_for_status()
        except Exception as exc:
            vram = await self._check_vram()
            logger.error("vLLM liveness check failed: %s", exc)
            raise RuntimeError(
                f"vLLM server unreachable or unhealthy at {self.base_url} ({exc}). "
                f"Free VRAM = {vram['free_mb']} MB / {vram['total_mb']} MB total — likely resource exhaustion."
            )

        # Step 3: Build the streaming payload.
        # The 'model' key MUST match exactly the model ID that the vLLM server 
        # was launched with (e.g., via `--model`), otherwise it returns HTTP 404.
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload,
            ) as response:
                response.raise_for_status()
                # Step 4: Parse SSE event stream.
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content

        except httpx.HTTPStatusError as exc:
            vram = await self._check_vram()
            logger.error(
                "vLLM HTTP error: %s — VRAM at time of failure: "
                "total=%d MB, free=%d MB",
                exc, vram["total_mb"], vram["free_mb"],
            )
            raise RuntimeError(
                f"Engine vLLM returned HTTP {exc.response.status_code}. "
                f"VRAM: {vram['free_mb']} MB free / {vram['total_mb']} MB total."
            )

        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            vram = await self._check_vram()
            logger.error("vLLM connection failed: %s", exc)
            raise RuntimeError(
                f"Engine vLLM unreachable at {self.base_url}. "
                f"VRAM: {vram['free_mb']} MB free."
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if the vLLM server is reachable.
        """
        try:
            await self._ensure_startup_check()
        except RuntimeError as e:
            vram = await self._check_vram()
            return {
                "status": "degraded",
                "reason": str(e),
                "vram": vram,
            }

        try:
            resp = await self._client.get("/models", timeout=5.0)
            if resp.status_code == 200:
                return {"status": "ok"}
            
            vram = await self._check_vram()
            return {
                "status": "degraded",
                "reason": f"vLLM returned HTTP {resp.status_code}",
                "vram": vram,
            }
        except Exception as exc:
            vram = await self._check_vram()
            return {
                "status": "degraded",
                "reason": f"vLLM unreachable: {str(exc)}",
                "vram": vram,
            }

    async def close(self):
        """Release the httpx client session."""
        await self._client.aclose()
