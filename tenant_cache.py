"""
tenant_cache.py
===============
Tenant-aware semantic cache wrapper providing namespace isolation across tenants.

Known limitation:
All namespaces share one underlying FAISS vector index with top-k=5 candidates.
Therefore, many tenants caching semantically near-identical questions can push
a tenant's own entry outside the top-5 candidate window retrieved during FAISS
search, resulting in a false cache MISS. This limitation impacts hit rate under
high cross-tenant query overlap, but NEVER compromises cache isolation, as exact
namespace-prefixed lock matching prevents cross-tenant hits.
"""

from __future__ import annotations

from typing import Any, Optional

from semantic_cache import (
    DeterministicSemanticCache,
    semantic_cache as default_semantic_cache,
)


class TenantSemanticCache:
    """
    Wraps a DeterministicSemanticCache to enforce tenant-level isolation modes.

    Known limitation:
    All namespaces share one underlying FAISS index with top-k=5 candidates.
    Consequently, many tenants caching near-identical questions can push a
    tenant's own entry out of the top 5 nearest neighbors returned by vector
    search, leading to a false MISS. That costs hit rate, never isolation.
    """

    def __init__(
        self,
        inner: Optional[DeterministicSemanticCache] = None,
    ) -> None:
        """
        Initializes the tenant semantic cache wrapper.

        Inputs:
        - inner: Optional DeterministicSemanticCache instance. If None, defaults
                 to the global singleton instance from semantic_cache.py.
        """
        self.inner: DeterministicSemanticCache = (
            inner if inner is not None else default_semantic_cache
        )

    @staticmethod
    def namespace_for(tenant_id: str, cache_mode: str) -> Optional[str]:
        """
        Maps tenant_id and cache_mode to a cache lock namespace prefix.

        Inputs:
        - tenant_id: The authenticated tenant identifier.
        - cache_mode: Cache mode ('isolated', 'shared', or 'off').

        Returns:
        - Namespace prefix string ('tenant:<id>' or 'shared'), or None if 'off'.

        Errors:
        - Raises ValueError if cache_mode is unrecognized.
        """
        if cache_mode == "isolated":
            return f"tenant:{tenant_id}"
        elif cache_mode == "shared":
            return "shared"
        elif cache_mode == "off":
            return None
        else:
            raise ValueError(f"Unknown cache_mode: '{cache_mode}'")

    async def check(
        self,
        messages: list[dict[str, str]],
        tenant_id: str,
        cache_mode: str,
    ) -> Optional[dict[str, Any]]:
        """
        Checks the semantic cache under the tenant's configured cache mode.

        Inputs:
        - messages: Conversation history dicts.
        - tenant_id: Authenticated tenant ID.
        - cache_mode: Tenant cache mode ('isolated', 'shared', 'off').

        Returns:
        - Cached response dict if hit, or None if miss/off.
        """
        if cache_mode == "off":
            return None

        namespace = self.namespace_for(tenant_id, cache_mode)
        return await self.inner.check_cache(messages, namespace=namespace)

    async def insert(
        self,
        messages: list[dict[str, str]],
        response: dict[str, Any],
        tenant_id: str,
        cache_mode: str,
    ) -> None:
        """
        Inserts a response into the semantic cache under the tenant's namespace.

        Inputs:
        - messages: Conversation history dicts.
        - response: Completion response dict to cache.
        - tenant_id: Authenticated tenant ID.
        - cache_mode: Tenant cache mode ('isolated', 'shared', 'off').
        """
        if cache_mode == "off":
            return

        namespace = self.namespace_for(tenant_id, cache_mode)
        await self.inner.insert(messages, response, namespace=namespace)
