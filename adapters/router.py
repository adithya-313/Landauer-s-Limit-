"""
router.py
=========
PHASE 5 — Multi-Engine Router.

The EngineRouter maintains a map of registered engine adapters and
dispatches each generation request to the selected engine.

Key behaviour:
- If no engine is specified, defaults to "vllm_local".
- If the selected engine fails (raises RuntimeError), the router
  automatically falls back to "llamacpp_local" and logs the fallback.
- The fallback chain is single-level: vLLM -> llama.cpp.  Colab and
  custom runtime are never used as automatic fallbacks (they require
  explicit selection).
"""

import logging
from typing import AsyncIterator, Optional

from .custom_runtime_adapter import CustomRuntimeAdapter

logger = logging.getLogger("EngineRouter")


class EngineRouter:
    """
    Routes generation requests to the appropriate engine adapter.

    Parameters
    ----------
    adapters : dict, optional
        A pre-configured dict of {name: adapter_instance}.
        If not provided, the router creates default instances.
    """

    def __init__(self, adapters: Optional[dict] = None):
        if adapters is not None:
            self.adapters = adapters
        else:
            # Lazy imports so only the selected engine is loaded.
            from .vllm_adapter import VLLMAdapter
            from .llamacpp_adapter import LlamaCppAdapter
            from .colab_adapter import ColabAdapter

            self.adapters = {
                "vllm_local": VLLMAdapter(),
                "llamacpp_local": LlamaCppAdapter(),
                "colab_cloud": ColabAdapter(),
                "custom_runtime": CustomRuntimeAdapter(),
            }

        # Define the fallback chain for resilience.
        self._fallback_map = {
            "vllm_local": "llamacpp_local",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_request(
        self, engine_name: str, prompt: str, tier: str = "free"
    ) -> AsyncIterator[str]:
        """
        Generate tokens using the specified engine, with fallback.

        Parameters
        ----------
        engine_name : str
            One of "vllm_local", "llamacpp_local", "colab_cloud",
            "custom_runtime".
        prompt : str
            The user prompt to send to the engine.

        Yields
        ------
        str
            Token strings from the engine output.
        """
        # Step 1: Normalise the engine name.
        engine_name = engine_name or "vllm_local"

        # Step 2: Look up the adapter.
        adapter = self.adapters.get(engine_name)
        if adapter is None:
            logger.warning(
                "Unknown engine '%s' — falling back to vllm_local.",
                engine_name,
            )
            engine_name = "vllm_local"
            adapter = self.adapters["vllm_local"]

        # Step 3: Attempt generation on the primary engine.
        try:
            async for token in adapter.generate(prompt, tier=tier):
                yield token
            # If we get here the stream completed successfully.
            return

        except RuntimeError as exc:
            logger.error(
                "Engine '%s' failed: %s",
                engine_name, exc,
            )

        # Step 4: Check if a fallback is defined.
        fallback_name = self._fallback_map.get(engine_name)
        if fallback_name is None:
            # No fallback for this engine — propagate the error.
            raise RuntimeError(
                f"Engine '{engine_name}' failed and no fallback is configured."
            )

        # Step 5: Attempt generation on the fallback engine.
        fallback_adapter = self.adapters.get(fallback_name)
        logger.warning(
            "Falling back from '%s' to '%s'.",
            engine_name, fallback_name,
        )
        async for token in fallback_adapter.generate(prompt, tier=tier):
            yield token

    async def health_check(self, engine_name: str = "vllm_local") -> dict:
        """
        Return health status for a specific engine.
        """
        adapter = self.adapters.get(engine_name)
        if adapter is None:
            return {"status": "unknown", "engine": engine_name}
        return await adapter.health_check()
