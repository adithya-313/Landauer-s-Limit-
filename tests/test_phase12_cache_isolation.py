"""
tests/test_phase12_cache_isolation.py
=====================================
Unit tests for Phase 12 tenant semantic cache isolation:
- Isolated mode: namespace 'tenant:<tenant_id>' ensures tenant A cannot hit tenant B's entries.
- Shared mode: namespace 'shared' allows cross-tenant cache hits.
- Off mode: bypasses cache entirely (no read, no write).
- Cross-mode isolation: isolated entries are never visible in shared mode and vice versa.
- Backward compatibility: namespace=None on inner cache functions identically to Phase 4.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

import numpy as np
import pytest

from semantic_cache import DeterministicSemanticCache
from tenant_cache import TenantSemanticCache

EMBEDDING_DIM: int = 384
_EMBEDDING_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()


def mock_embed_text(text: str) -> np.ndarray:
    """
    Deterministic mock embedding: same text -> identical unit vector.
    Different text -> reproducible pseudorandom unit vector.
    """
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    seed = hash(text) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(1, EMBEDDING_DIM).astype(np.float32)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    _EMBEDDING_CACHE[text] = vec
    return vec


def _create_fresh_cache() -> tuple[DeterministicSemanticCache, TenantSemanticCache]:
    """Builds a fresh inner cache with mock embeddings and wraps it in TenantSemanticCache."""
    inner = DeterministicSemanticCache(embedding_function=mock_embed_text)
    wrapper = TenantSemanticCache(inner=inner)
    return inner, wrapper


def test_lock_generation_is_non_none() -> None:
    """Confirms test prompt produces a valid lock string before testing cache operations."""
    messages = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
    inner, _ = _create_fresh_cache()
    lock = inner.extractor.generate_lock(messages)
    assert lock is not None
    assert "python" in lock


def test_isolated_cache_mode_isolates_tenants() -> None:
    """
    Verifies isolated cache mode:
    - Tenant A inserts an entry.
    - Tenant B checks the same messages -> MISS (None).
    - Tenant A checks the same messages -> HIT.
    """
    async def _test() -> None:
        _, cache = _create_fresh_cache()
        messages = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
        response_payload = {"choices": [{"message": {"role": "assistant", "content": "Install steps"}}]}

        # Tenant A inserts
        await cache.insert(messages, response_payload, tenant_id="tenant_a", cache_mode="isolated")

        # Tenant B queries identical messages -> must MISS
        result_b = await cache.check(messages, tenant_id="tenant_b", cache_mode="isolated")
        assert result_b is None, "Tenant B must not see Tenant A's isolated cache entry"

        # Tenant A queries identical messages -> must HIT
        result_a = await cache.check(messages, tenant_id="tenant_a", cache_mode="isolated")
        assert result_a is not None, "Tenant A must hit its own isolated entry"
        assert result_a == response_payload

    asyncio.run(_test())


def test_shared_cache_mode_allows_cross_tenant_hits() -> None:
    """
    Verifies shared cache mode:
    - Tenant A inserts in shared mode.
    - Tenant B queries the same messages in shared mode -> HIT.
    """
    async def _test() -> None:
        _, cache = _create_fresh_cache()
        messages = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
        response_payload = {"choices": [{"message": {"role": "assistant", "content": "Shared steps"}}]}

        await cache.insert(messages, response_payload, tenant_id="tenant_a", cache_mode="shared")

        result_b = await cache.check(messages, tenant_id="tenant_b", cache_mode="shared")
        assert result_b is not None, "Tenant B must hit Tenant A's entry in shared cache mode"
        assert result_b == response_payload

    asyncio.run(_test())


def test_off_cache_mode_bypasses_cache_completely() -> None:
    """
    Verifies off cache mode:
    - check returns None without reading inner cache.
    - insert does not store anything (inner cache size unchanged).
    """
    async def _test() -> None:
        inner, cache = _create_fresh_cache()
        messages = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
        response_payload = {"choices": [{"message": {"role": "assistant", "content": "Off steps"}}]}

        # Pre-insert one item under isolated mode to have nonzero cache size
        await cache.insert(messages, response_payload, tenant_id="tenant_x", cache_mode="isolated")
        initial_size = inner.size
        assert initial_size == 1

        # Check in off mode -> None
        check_result = await cache.check(messages, tenant_id="tenant_x", cache_mode="off")
        assert check_result is None

        # Insert in off mode -> size unchanged
        new_messages = [{"role": "user", "content": "How do I install docker on Ubuntu?"}]
        await cache.insert(new_messages, response_payload, tenant_id="tenant_x", cache_mode="off")
        assert inner.size == initial_size

    asyncio.run(_test())


def test_cross_mode_isolation_between_isolated_and_shared() -> None:
    """
    Verifies that isolated entries are invisible to shared-mode queries,
    and shared entries are invisible to isolated-mode queries.
    """
    async def _test() -> None:
        _, cache = _create_fresh_cache()
        messages_iso = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
        messages_shared = [{"role": "user", "content": "How do I install docker on Ubuntu?"}]

        resp_iso = {"choices": [{"message": {"role": "assistant", "content": "Isolated content"}}]}
        resp_shared = {"choices": [{"message": {"role": "assistant", "content": "Shared content"}}]}

        # Insert isolated
        await cache.insert(messages_iso, resp_iso, tenant_id="tenant_a", cache_mode="isolated")
        # Insert shared
        await cache.insert(messages_shared, resp_shared, tenant_id="tenant_b", cache_mode="shared")

        # Isolated entry checked in shared mode -> None
        assert await cache.check(messages_iso, tenant_id="tenant_a", cache_mode="shared") is None
        assert await cache.check(messages_iso, tenant_id="tenant_b", cache_mode="shared") is None

        # Shared entry checked in isolated mode -> None
        assert await cache.check(messages_shared, tenant_id="tenant_a", cache_mode="isolated") is None
        assert await cache.check(messages_shared, tenant_id="tenant_b", cache_mode="isolated") is None

    asyncio.run(_test())


def test_namespace_none_on_inner_cache_behaves_as_before() -> None:
    """
    Verifies that passing namespace=None to the inner cache preserves
    standard Phase 4 behaviour (insert then check yields a cache HIT).
    """
    async def _test() -> None:
        inner, _ = _create_fresh_cache()
        messages = [{"role": "user", "content": "How do I install python 3.12 on Ubuntu?"}]
        response_payload = {"choices": [{"message": {"role": "assistant", "content": "Legacy content"}}]}

        await inner.insert(messages, response_payload, namespace=None)
        hit = await inner.check_cache(messages, namespace=None)
        assert hit is not None
        assert hit == response_payload

    asyncio.run(_test())


def test_namespace_for_mapping_and_errors() -> None:
    """Verifies namespace_for maps modes correctly and raises ValueError for unknown modes."""
    assert TenantSemanticCache.namespace_for("t1", "isolated") == "tenant:t1"
    assert TenantSemanticCache.namespace_for("t1", "shared") == "shared"
    assert TenantSemanticCache.namespace_for("t1", "off") is None

    with pytest.raises(ValueError, match="Unknown cache_mode: 'unsupported'"):
        TenantSemanticCache.namespace_for("t1", "unsupported")
