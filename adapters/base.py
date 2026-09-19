"""
base.py
=======
PHASE 5 — Abstract Base Class for all Engine Adapters.

Every engine (local vLLM, llama.cpp, Colab T4, custom runtime) must
implement the two methods defined here. This lets the router treat all
engines uniformly regardless of their internal transport mechanism.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any


class BaseEngineAdapter(ABC):
    """
    Abstract base class that every engine adapter must implement.

    Two required methods:
      1. generate(prompt) — yields tokens as a string stream.
      2. health_check()   — returns a dict with status info.
    """

    @abstractmethod
    async def generate(self, prompt: str, tier: str = "free") -> AsyncIterator[str]:
        """
        Stream generated tokens for the given prompt.

        Parameters
        ----------
        prompt : str
            The user prompt to generate tokens for.
        tier : str
            The requester's service tier (e.g. "free" or "premium").
            Used by engines that implement tier-aware behavior (e.g. custom_runtime
            for priority eviction). Most adapters (vLLM, llama.cpp, etc.) do not 
            implement tier-aware logic natively and will ignore this parameter. 
            The default exists so adapters that don't use tier-based logic remain 
            unaffected and don't need special-case handling in the router.

        Yields
        ------
        str
            Each yielded string is a token or chunk of the response.
        """
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Return a snapshot of the engine's health.

        Returns
        -------
        dict
            Must include at least {"status": "ok"} or {"status": "degraded", "reason": "..."}.
        """
        raise NotImplementedError
