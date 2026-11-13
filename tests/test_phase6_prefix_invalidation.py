"""
Tests prefix cache hash invalidation upon cached-free block eviction.

Verifies that when a cached block in cached_free is reclaimed and reused by
another sequence, its prior hash registration is cleanly purged from hash_registry,
its block_hash is reset to None, and subsequent prefix lookups do not return the
reused physical block.
"""

from __future__ import annotations

import dataclasses
import queue
import torch

from runtime_core.kv_cache_manager import KVCacheManager


@dataclasses.dataclass
class MockState:
    request_id: str
    prompt_len: int = 16
    tokens_produced: int = 0
    tier: str = "free"
    finished: bool = False
    response_queue: queue.Queue = dataclasses.field(default_factory=queue.Queue)


def test_prefix_invalidation_on_block_reuse() -> None:
    """Verifies hash registry purge and block_info reset when cached_free blocks are reclaimed."""
    device = torch.device("cpu")
    # Total pool size is 3 blocks: block 0, block 1, block 2
    kv = KVCacheManager(device=device, max_blocks=3, dtype=torch.float32)

    # 1. Sequence A allocates 2 blocks (32 tokens) -> uses physical blocks 0 and 1
    state_a = MockState(request_id="seq_a", prompt_len=32)
    tokens_a = list(range(32))
    kv.ensure_allocation(state_a, 32, [])
    hashed_count = kv.register_prompt_blocks("seq_a", tokens_a)
    assert hashed_count == 2, f"Expected 2 hashed blocks, got {hashed_count}"

    page_table_a = kv.page_table["seq_a"]
    block_0, block_1 = page_table_a[0], page_table_a[1]
    hash_0 = kv.block_info[block_0].block_hash
    hash_1 = kv.block_info[block_1].block_hash

    assert hash_0 is not None and hash_0 in kv.hash_registry
    assert hash_1 is not None and hash_1 in kv.hash_registry
    assert kv.hash_registry[hash_0] == block_0
    assert kv.hash_registry[hash_1] == block_1

    # 2. Free sequence A -> blocks 0 and 1 move to cached_free (in insertion order: 0, then 1)
    kv.free_sequence("seq_a")
    assert block_0 in kv.cached_free
    assert block_1 in kv.cached_free
    # free_blocks now contains only [block_2] (1 block left)
    assert len(kv.free_blocks) == 1

    # 3. Force reuse of block_0: sequence B requests 2 blocks (32 tokens).
    # First block is taken from free_blocks (block_2).
    # Second block MUST come from cached_free: popitem(last=False) pops the oldest entry (block_0).
    state_b = MockState(request_id="seq_b", prompt_len=32)
    kv.ensure_allocation(state_b, 32, [])
    page_table_b = kv.page_table["seq_b"]
    assert block_0 in page_table_b, f"Expected {block_0} to be reused for seq_b, got {page_table_b}"

    # Assert 1: The old hash of block_0 is no longer in hash_registry
    assert hash_0 not in kv.hash_registry, (
        f"Evicted block {block_0}'s hash {hash_0} must be deleted from hash_registry"
    )

    # Assert 2: block_info[block_0].block_hash is None
    assert kv.block_info[block_0].block_hash is None, (
        f"Evicted block {block_0}'s block_hash must be cleared to None"
    )

    # Assert 3: A subsequent prefix lookup for the old tokens does not hit block_0
    hits_new = kv.acquire_prefix("seq_c", tokens_a)
    assert hits_new == 0, f"Expected 0 hits for invalidated prefix, got {hits_new}"

    # Negative control: block_1 was NOT reused and remains valid in cached_free
    assert block_1 in kv.cached_free, f"Block {block_1} should still be in cached_free"
    assert hash_1 in kv.hash_registry, f"Un-reused block {block_1}'s hash must remain in registry"
    assert kv.block_info[block_1].block_hash == hash_1, f"Un-reused block {block_1} must keep its hash"

