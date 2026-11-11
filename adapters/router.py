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
from typing import Any, AsyncIterator, Optional

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

    @classmethod
    def from_specs(cls, specs: list) -> "EngineRouter":
        """
        Constructs an EngineRouter instance by building adapters from a list of EngineSpecs.

        Inputs:
        - specs: List of validated EngineSpec models.

        Returns:
        - EngineRouter configured with the instantiated adapters.
        """
        from engine_config import build_adapter

        built_adapters = {spec.id: build_adapter(spec) for spec in specs}
        return cls(adapters=built_adapters)

    def add_engine(self, spec: Any) -> None:
        """
        Dynamically registers a new engine adapter using copy-on-write.

        Inputs:
        - spec: Validated EngineSpec describing the new engine to add.

        Side Effects:
        - Builds the adapter and replaces self.adapters with a new dictionary copy.
          In-flight requests holding the previous dictionary reference are unaffected.

        Errors:
        - Raises ValueError if an engine with the given id is already registered.
        """
        if spec.id in self.adapters:
            raise ValueError(f"Engine '{spec.id}' is already registered")

        from engine_config import build_adapter

        adapter = build_adapter(spec)

        # Copy-then-assign protects concurrent in-flight requests holding the old dict
        new_adapters = dict(self.adapters)
        new_adapters[spec.id] = adapter
        self.adapters = new_adapters

    def remove_engine(self, engine_id: str) -> Any:
        """
        Dynamically unregisters an engine adapter using copy-on-write.

        Inputs:
        - engine_id: Identifier of the engine to remove.

        Returns:
        - The removed adapter instance.

        Side Effects:
        - Replaces self.adapters with a new dictionary copy omitting the removed engine.
          Does NOT close the removed adapter's client connection because in-flight
          requests may still be actively using it to complete generation.

        Errors:
        - Raises KeyError if engine_id is not registered.
        """
        if engine_id not in self.adapters:
            raise KeyError(f"Engine '{engine_id}' is not registered")

        removed_adapter = self.adapters[engine_id]

        # Copy-then-assign protects in-flight readers
        new_adapters = dict(self.adapters)
        del new_adapters[engine_id]
        self.adapters = new_adapters

        # Do NOT close removed_adapter here: in-flight requests may still be streaming from it.
        return removed_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def route_request(
        self, engine_name: str, prompt: str, tier: str = "free", max_tokens: int = 100, messages: Optional[list[dict]] = None
    ) -> AsyncIterator[str]:
        """
        Routes the request to the primary engine. If that engine fails,
        attempts to failover to a designated fallback engine.

        Parameters
        ----------
        engine_name : str
            One of "vllm_local", "llamacpp_local", "colab_cloud",
            "custom_runtime".
        prompt : str
            The user prompt to send to the engine.
        tier : str
            The tier of the request.
        max_tokens : int
            The maximum number of tokens to generate.

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
            if "vllm_local" in self.adapters:
                logger.warning(
                    "Unknown engine '%s' — falling back to vllm_local.",
                    engine_name,
                )
                engine_name = "vllm_local"
                adapter = self.adapters["vllm_local"]
            else:
                raise RuntimeError(f"Unknown engine '{engine_name}'")

        # Step 3: Attempt generation on the primary engine.
        try:
            async for token in adapter.generate(prompt, tier=tier, max_tokens=max_tokens, messages=messages):
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
        if fallback_name is None or fallback_name not in self.adapters:
            # No fallback for this engine or fallback not registered — propagate the error.
            raise RuntimeError(
                f"Engine '{engine_name}' failed and no fallback is configured."
            )

        # Step 5: Attempt generation on the fallback engine.
        fallback_adapter = self.adapters[fallback_name]
        logger.warning(
            "Falling back from '%s' to '%s'.",
            engine_name, fallback_name,
        )
        async for token in fallback_adapter.generate(prompt, tier=tier, messages=messages):
            yield token

    async def health_check(self, engine_name: str = "vllm_local") -> dict:
        """
        Return health status for a specific engine.
        """
        adapter = self.adapters.get(engine_name)
        if adapter is None:
            return {"status": "unknown", "engine": engine_name}
        return await adapter.health_check()
