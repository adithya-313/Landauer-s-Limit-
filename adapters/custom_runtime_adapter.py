"""
custom_runtime_adapter.py
=========================
PHASE 5/6 — Custom Runtime Engine Adapter (Placeholder).

This adapter is the placeholder slot for the Phase 6 Runtime Core.
It does NOT connect to a real server.  Instead, it returns a structured
placeholder string so that the gateway integration can be verified end
to end before the real runtime exists.

When Phase 6 delivers the actual runtime, only this file needs to be
replaced — the router and all other adapters stay untouched.
"""

import logging
from typing import AsyncIterator, Dict, Any

logger = logging.getLogger("CustomRuntimeAdapter")


class CustomRuntimeAdapter:
    """
    Placeholder adapter for the Phase 6 Custom Runtime.

    Returns a deterministic placeholder string wrapped in a structured
    format that can be verified through the gateway interface.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        """
        Yield a single structured placeholder token.

        The output format is:
          " [CUSTOM_RUNTIME_PLACEHOLDER: <prompt snippet>] "
        """
        snippet = prompt[:80] + "..." if len(prompt) > 80 else prompt
        placeholder = f" [CUSTOM_RUNTIME_PLACEHOLDER: {snippet}] "
        # Yield the whole string as one token so it is easy to verify.
        yield placeholder

    async def health_check(self) -> Dict[str, Any]:
        """
        Placeholder is always healthy.
        """
        return {
            "status": "ok",
            "engine": "custom_runtime_placeholder",
            "note": "Awaiting Phase 6 runtime core.",
        }
