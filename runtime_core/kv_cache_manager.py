"""
kv_cache_manager.py
===================
PHASE 6c — KV Cache Manager (Path B)

Explanation of the KV Cache:
When a Large Language Model generates text, it reads the entire conversation so far to predict the next word. 
To avoid re-reading and re-calculating the internal math for every past word (which would be extremely slow), 
it stores the intermediate mathematical representations (called Keys and Values) for each word. This stored 
memory is the "KV cache". 

Why fixed-size blocks?
Traditionally, the KV cache for a sequence is kept in a single contiguous block of memory. However, because 
we don't know in advance how long a user's generated response will be, memory must be over-provisioned or 
constantly resized, leading to memory fragmentation (holes of unused memory). Fixed-size blocks solve this by 
choosing a small set size (e.g., 16 tokens). When a sequence needs more memory, we just give it another block. 
These blocks can be scattered anywhere in the physical memory, which eliminates fragmentation and allows us 
to pack the GPU's memory tightly, similar to how operating systems manage RAM (a concept called PagedAttention 
in the vLLM architecture).

Why Path B? (The deliberate architectural choice)
True PagedAttention requires writing custom low-level CUDA kernels so the model's attention mechanism can directly 
read from scattered blocks in memory during its forward pass. This project has a strict ground rule that 
kernel-level CUDA authorship is out of scope. Therefore, we use "Path B":
1. The block allocator is the real, authoritative memory-management layer. It handles real allocation, reuse, 
   and eviction of blocks in pre-allocated physical memory tensors.
2. Just before the model's forward pass, we read from these scattered blocks to reconstruct a standard, contiguous 
   cache object. The model performs its standard forward pass on this contiguous tensor.
3. After the forward pass, we take the newly generated keys/values and write them back into the scattered blocks, 
   allocating new blocks as necessary.
This bridges the gap, allowing us to build and verify a real block allocator and page table system without 
having to write custom attention kernels.

Configuration for Qwen2.5-1.5B-Instruct:
- 28 hidden layers, 2 KV heads, head_dim 128.
- 1 token = 2 (keys & values) x 2 (KV heads) x 128 (head_dim) x 2 bytes (fp16) = 1,024 bytes.
- 28 layers = ~28,672 bytes (~28KB) per token.
- BLOCK_SIZE = 16 tokens.
- Cost per block = 448 KB.
"""

import torch
import time
import json
import subprocess
from typing import List, Dict, Tuple, Optional
import dataclasses
from collections import OrderedDict
from transformers.cache_utils import DynamicCache, DynamicLayer
from runtime_core.prefix_hashing import compute_block_hashes, GENESIS_PARENT_HASH
import logging

@dataclasses.dataclass
class BlockInfo:
    ref_count: int = 0
    block_hash: str | None = None
    parent_hash: str | None = None
    token_ids: tuple[int, ...] | None = None
    last_used: int = 0  # logical clock tick when this block was last touched


logger = logging.getLogger("KVCacheManager")

BLOCK_SIZE = 16
# Qwen2.5-1.5B-Instruct specific
NUM_LAYERS = 28
NUM_KV_HEADS = 2
HEAD_DIM = 128

# Kill-switch - if anything about prefix caching misbehaves, flip this to False 
# to restore exact pre-6d-3 behavior with zero other code changes needed.
PREFIX_CACHE_ENABLED = True


class KVCacheManager:
    def __init__(self, device: torch.device, max_blocks: int = 2000, dtype=torch.bfloat16):
        self.dtype = dtype
        """
        Initializes the physical block pool and free block tracking.
        """
        self.device = device
        self.max_blocks = max_blocks
        
        # Free physical block pool
        self.free_blocks: List[int] = list(range(max_blocks))
        
        # One BlockInfo per physical block, indexed the same way 
        # self.free_blocks/self.page_table reference physical block ids.
        self.block_info: List[BlockInfo] = [BlockInfo() for _ in range(max_blocks)]
        self._clock: int = 0
        
        # Three-state model: plain-free (free_blocks) -> cached-free (cached_free, still remembers content) -> in-use (ref_count > 0)
        self.cached_free: OrderedDict[int, None] = OrderedDict()
        self.hash_registry: dict[str, int] = {}
        self.prefix_cache_enabled: bool = PREFIX_CACHE_ENABLED
        
        # Page table mapping: request_id -> List[physical_block_id]
        self.page_table: Dict[str, List[int]] = {}
        
        # Physical block storage (The actual real allocation)
        # Shape for keys/values: [max_blocks, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
        # Memory calculation: 2000 blocks * 448 KB = ~896 MB
        print(f"Initializing KV Cache Manager pool with {max_blocks} blocks (~{max_blocks * 448 / 1024:.1f} MB)")
        
        self.physical_keys = torch.zeros(
            (max_blocks, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
            dtype=self.dtype,
            device=self.device
        )
        self.physical_values = torch.zeros(
            (max_blocks, NUM_LAYERS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM),
            dtype=self.dtype,
            device=self.device
        )
        
        self.last_log_time = 0.0

    def _log_event(self, event: dict):
        """Writes a record of eviction events to the batch engine's log file."""
        event["timestamp"] = time.time()
        try:
            with open("batch_events.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def _hand_out_block(self) -> Optional[int]:
        if self.free_blocks:
            return self.free_blocks.pop(0)
        elif self.cached_free:
            # OrderedDict is insertion-ordered, so popitem(last=False) returns the oldest entry
            block_id, _ = self.cached_free.popitem(last=False)
            
            # we're about to reuse/overwrite this block's memory,
            # so its old identity is no longer valid — must delete here, not later, or a future
            # lookup would falsely match content that's been evicted
            block_hash = self.block_info[block_id].block_hash
            if block_hash and block_hash in self.hash_registry and self.hash_registry[block_hash] == block_id:
                del self.hash_registry[block_hash]
            
            # clear the BlockInfo hash fields so it doesn't accidentally look like a registered block
            self.block_info[block_id].block_hash = None
            self.block_info[block_id].parent_hash = None
            self.block_info[block_id].token_ids = None
            
            return block_id
        else:
            return None

    def _allocate_block(self, active_slots: List['RequestState'], current_state: 'RequestState') -> Optional[int]:
        """
        Pulls a free physical block from the pool.
        If empty, evicts the most recently admitted sequence (LIFO) to free space,
        protecting older sequences that have been running longer.
        """
        phys_block = self._hand_out_block()
        if phys_block is None:
            success = self._evict_sequence(active_slots, current_state)
            if not success:
                return None
            phys_block = self._hand_out_block()
            if phys_block is None:
                raise RuntimeError("Failed to allocate block even after eviction attempt.")
            
        self._clock += 1
        # This is the single point where a block transitions from plain-free to in-use,
        # so ref_count starts at exactly 1 here (the one owner that just received it) 
        # - not 0, not incremented from some prior value.
        self.block_info[phys_block].ref_count = 1
        self.block_info[phys_block].last_used = self._clock
        return phys_block

    def _evict_sequence(self, active_slots: List['RequestState'], current_state: 'RequestState') -> bool:
        """
        DECISION: Priority-tier-based eviction policy.
        We prioritize keeping 'premium' tier requests alive over 'free' tier requests.
        When memory pressure occurs, we check the current requester's tier.
        If 'premium': we search for a 'free' tier sequence to evict. If none exist, we 
        evict the newest 'premium' one. The current request is exempt.
        If 'free': we search for a 'free' tier sequence to evict. If none exist, meaning
        all other active requests are 'premium', the 'free' requester itself fails cleanly
        rather than evicting a protected 'premium' request.
        
        When evicted (or failed), the sequence is marked finished and an error is sent to its queue.
        """
        evicted_req = None
        requester_tier = getattr(current_state, 'tier', 'free')
        current_req_id = current_state.request_id
        
        # First, try to find a 'free' tier sequence, starting from the newest
        for state in reversed(active_slots):
            if state.request_id != current_req_id and getattr(state, 'tier', 'free') == 'free':
                if not getattr(state, 'finished', False):
                    # evicting an all-shared sequence frees zero physical blocks, so it's
                    # pointless and must be skipped in favor of a real victim.
                    if any(self.block_info[bid].ref_count == 1 for bid in self.page_table.get(state.request_id, [])):
                        evicted_req = state
                        break
                
        # If no 'free' sequence was found, fallback behavior depends on the requester's tier
        if not evicted_req:
            if requester_tier == 'premium':
                # Premium can evict other premium requests if absolutely necessary
                for state in reversed(active_slots):
                    if state.request_id != current_req_id:
                        if not getattr(state, 'finished', False):
                            # evicting an all-shared sequence frees zero physical blocks, so it's
                            # pointless and must be skipped in favor of a real victim.
                            if any(self.block_info[bid].ref_count == 1 for bid in self.page_table.get(state.request_id, [])):
                                evicted_req = state
                                break
            else:
                # Free tier CANNOT evict premium requests. It must fail its own admission.
                current_state.finished = True
                current_state.response_queue.put({"type": "error", "content": "OOM: Cannot admit free tier because all memory is used by premium requests"})
                self._log_event({
                    "event": "admission_failed",
                    "request_id": current_req_id,
                    "tier": requester_tier,
                    "reason": "memory_pressure_premium_protected"
                })
                return False
                
        if not evicted_req:
            return False # No other sequence to evict, and we shouldn't get here for free tier
            
        # Free the blocks
        self.free_sequence(evicted_req.request_id)
        
        # Mark as failed
        evicted_req.finished = True
        evicted_req.response_queue.put({"type": "error", "content": "OOM: Evicted due to memory pressure"})
        
        self._log_event({
            "event": "eviction",
            "evicted_request_id": evicted_req.request_id,
            "evicted_tier": getattr(evicted_req, 'tier', 'free'),
            "requester_id": current_req_id,
            "requester_tier": requester_tier,
            "reason": "memory_pressure"
        })
        return True

    def ensure_allocation(self, state: 'RequestState', logical_length: int, active_slots: List['RequestState']):
        """
        Ensures a sequence has enough physical blocks allocated for its logical length.
        """
        request_id = state.request_id
        if request_id not in self.page_table:
            self.page_table[request_id] = []
            
        blocks_needed = (logical_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        current_blocks = len(self.page_table[request_id])
        
        while current_blocks < blocks_needed:
            phys_block = self._allocate_block(active_slots, state)
            if phys_block is None:
                return # Allocation failed
            self.page_table[request_id].append(phys_block)
            current_blocks += 1

    def free_sequence(self, request_id: str):
        """
        Returns a sequence's blocks to the free pool.
        If request_id not in self.page_table, this is a no-op (covers double-free).
        This is intentional, since phase 6d-3 sharing will make repeated frees of overlapping sequences normal.
        """
        if request_id not in self.page_table:
            return
            
        blocks = self.page_table.pop(request_id)
        for block_id in blocks:
            self.block_info[block_id].ref_count -= 1
            if self.block_info[block_id].ref_count == 0:
                if self.block_info[block_id].block_hash is not None:
                    # this makes the block "cached-free" rather than "plain-free" — it still remembers 
                    # its content in case something needs it again
                    self.cached_free[block_id] = None
                else:
                    self.free_blocks.append(block_id)
            elif self.block_info[block_id].ref_count < 0:
                # This should be structurally impossible and indicates a bug elsewhere if it ever fires.
                raise RuntimeError(f"Negative ref_count {self.block_info[block_id].ref_count} for block {block_id} (request {request_id})")

    def ingest_prefill(self, state: 'RequestState', cache: DynamicCache, active_slots: List['RequestState'], logical_offset: int = 0):
        """
        Takes a contiguous cache from a prefill forward pass and writes it into the block allocator.
        """
        # Figure out sequence length from cache
        if hasattr(cache, 'layers'):
            seq_len = cache.layers[0].keys.shape[2]
        else:
            seq_len = cache.key_cache[0].shape[2]
            
        # The cache is the FULL sequence length (prefix + suffix).
        self.ensure_allocation(state, seq_len, active_slots)
        
        if getattr(state, 'finished', False):
            return # Aborted due to OOM
            
        # Write the data into blocks
        blocks = self.page_table.get(state.request_id, [])
        
        # To avoid a slow Python loop over all layers and tokens, we copy layer by layer
        for layer_idx in range(NUM_LAYERS):
            if hasattr(cache, 'layers'):
                keys = cache.layers[layer_idx].keys[0]  # shape: [NUM_KV_HEADS, seq_len, HEAD_DIM]
                vals = cache.layers[layer_idx].values[0]
            else:
                keys = cache.key_cache[layer_idx][0]
                vals = cache.value_cache[layer_idx][0]
                
            num_total_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            # logical_offset tells us to skip the first N blocks because they are already shared
            for logical_block_idx in range(logical_offset, num_total_blocks):
                if logical_block_idx >= len(blocks):
                    continue
                phys_block = blocks[logical_block_idx]
                
                start_idx = logical_block_idx * BLOCK_SIZE
                end_idx = min(start_idx + BLOCK_SIZE, seq_len)
                slice_len = end_idx - start_idx
                
                # Copy into physical storage.
                self.physical_keys[phys_block, layer_idx, :, :slice_len, :] = keys[:, start_idx:end_idx, :]
                self.physical_values[phys_block, layer_idx, :, :slice_len, :] = vals[:, start_idx:end_idx, :]

    def reconstruct_caches(self, active_slots: List['RequestState'], max_len: int) -> DynamicCache:
        """
        Path B Bridge: Reconstructs a standard, contiguous DynamicCache for the model's forward pass.
        This reads from the scattered physical blocks and packs them into a padded tensor.
        """
        combined = DynamicCache()
        if not active_slots:
            return combined
            
        batch_size = len(active_slots)
        
        for layer_idx in range(NUM_LAYERS):
            # Pre-allocate contiguous tensor for this layer for all active requests
            # Shape: [batch_size, NUM_KV_HEADS, max_len, HEAD_DIM]
            layer_keys = torch.zeros((batch_size, NUM_KV_HEADS, max_len, HEAD_DIM), dtype=self.dtype, device=self.device)
            layer_vals = torch.zeros((batch_size, NUM_KV_HEADS, max_len, HEAD_DIM), dtype=self.dtype, device=self.device)
            
            for batch_idx, state in enumerate(active_slots):
                seq_len = state.prompt_len + state.tokens_produced
                blocks = self.page_table.get(state.request_id, [])
                
                # We pad on the left (so valid data is on the right, matching old combine_caches)
                pad_len = max_len - seq_len
                
                for logical_block_idx, phys_block in enumerate(blocks):
                    start_idx = logical_block_idx * BLOCK_SIZE
                    end_idx = min(start_idx + BLOCK_SIZE, seq_len)
                    slice_len = end_idx - start_idx
                    
                    # Target placement shifted by pad_len
                    target_start = pad_len + start_idx
                    target_end = target_start + slice_len
                    
                    try:
                        layer_keys[batch_idx, :, target_start:target_end, :] = self.physical_keys[phys_block, layer_idx, :, :slice_len, :]
                        layer_vals[batch_idx, :, target_start:target_end, :] = self.physical_values[phys_block, layer_idx, :, :slice_len, :]
                    except Exception as e:
                        print(f"ERROR in reconstruct: max_len={max_len}, seq_len={seq_len}, pad_len={pad_len}")
                        print(f"logical_block={logical_block_idx}, phys_block={phys_block}, blocks_len={len(blocks)}")
                        print(f"start_idx={start_idx}, end_idx={end_idx}, slice_len={slice_len}, target_start={target_start}, target_end={target_end}")
                        print(f"req_id={state.request_id}, prompt_len={state.prompt_len}, tokens_produced={state.tokens_produced}")
                        raise e
                    
            if hasattr(combined, 'layers'):
                layer = DynamicLayer()
                layer.keys = layer_keys
                layer.values = layer_vals
                layer.is_initialized = True
                combined.layers.append(layer)
            else:
                combined.key_cache.append(layer_keys)
                combined.value_cache.append(layer_vals)
                
        return combined

    def redistribute_caches(self, active_slots: List['RequestState'], cache: DynamicCache):
        """
        Path B Bridge: After the forward pass, extracts the newly generated token's key/value
        from the combined cache and writes it into the block allocator.
        """
        for batch_idx, state in enumerate(active_slots):
            # The newly generated token is at the very end of the cache
            # because we left-padded and generated 1 token.
            new_seq_len = state.prompt_len + state.tokens_produced + 1
            
            if getattr(state, 'finished', False):
                continue # Aborted due to OOM
                
            # Ensure block exists for this new length
            self.ensure_allocation(state, new_seq_len, active_slots)
            
            if getattr(state, 'finished', False):
                continue # Aborted due to OOM during allocation
                
            blocks = self.page_table.get(state.request_id, [])
            
            # The new token's index in the unpadded logical sequence
            token_idx = new_seq_len - 1
            
            # WRITE-GUARD: shared/cached prefix blocks are always PROMPT blocks (indices
            # 0 through max_shareable_blocks-1 of a sequence). Decode always writes to
            # token_idx = prompt_len + tokens_produced, whose block index is always
            # strictly greater than the highest shareable prefix block index for any
            # positive prompt_len (verified: highest shareable index is
            # ((prompt_len-1)//BLOCK_SIZE)-1, while the first generated token's block index
            # is prompt_len//BLOCK_SIZE, which is always greater). Decode can therefore
            # never write into a block another request might still be sharing.
            logical_block_idx = token_idx // BLOCK_SIZE
            offset_in_block = token_idx % BLOCK_SIZE
            if logical_block_idx >= len(blocks):
                continue
            phys_block = blocks[logical_block_idx]
            
            for layer_idx in range(NUM_LAYERS):
                if hasattr(cache, 'layers'):
                    new_key = cache.layers[layer_idx].keys[batch_idx:batch_idx+1, :, -1:, :]
                    new_val = cache.layers[layer_idx].values[batch_idx:batch_idx+1, :, -1:, :]
                else:
                    new_key = cache.key_cache[layer_idx][batch_idx:batch_idx+1, :, -1:, :]
                    new_val = cache.value_cache[layer_idx][batch_idx:batch_idx+1, :, -1:, :]
                    
                # Write back the 1 token to the physical block
                self.physical_keys[phys_block, layer_idx, :, offset_in_block:offset_in_block+1, :] = new_key[0]
                self.physical_values[phys_block, layer_idx, :, offset_in_block:offset_in_block+1, :] = new_val[0]

    def log_fragmentation_if_needed(self, active_slots: List['RequestState'], log_func):
        """
        Tracks and reports fragmentation once per second.
        Fragmentation = ratio of allocated-but-unused block space to total allocated space.
        """
        current_time = time.time()
        if current_time - self.last_log_time < 1.0:
            return
            
        self.last_log_time = current_time
        
        used_tokens = 0
        unique_blocks = set()
        
        for state in active_slots:
            seq_len = state.prompt_len + state.tokens_produced
            used_tokens += seq_len
            blocks = self.page_table.get(state.request_id, [])
            unique_blocks.update(blocks)
            
        total_allocated_tokens = len(unique_blocks) * BLOCK_SIZE
            
        if total_allocated_tokens > 0:
            fragmentation_ratio = (total_allocated_tokens - used_tokens) / total_allocated_tokens
        else:
            fragmentation_ratio = 0.0
            
        allocated_blocks = self.max_blocks - len(self.free_blocks)
            
        log_func({
            "event": "kv_cache_stats",
            "allocated_blocks": allocated_blocks,
            "free_blocks": len(self.free_blocks),
            "fragmentation_ratio": fragmentation_ratio,
            "total_allocated_tokens": total_allocated_tokens,
            "used_tokens": used_tokens,
            "cached_free_blocks": len(self.cached_free),
            "hashed_blocks": sum(1 for bi in self.block_info if bi.block_hash is not None),
            "shared_blocks": sum(1 for bi in self.block_info if bi.ref_count >= 2)
        })

    def check_invariants(self) -> list[str]:
        """
        Verifies internal block state consistency.
        Returns [] if healthy, otherwise a list of human-readable problem descriptions.
        """
        errors = []
        
        in_use_count = sum(1 for b in self.block_info if b.ref_count > 0)
        free_count = len(self.free_blocks)
        cached_free_count = len(self.cached_free)
        
        if free_count + cached_free_count + in_use_count != self.max_blocks:
            errors.append(f"Block count mismatch: {free_count} free + {cached_free_count} cached-free + {in_use_count} in-use != {self.max_blocks} max")
            
        actual_page_table_counts = {}
        for request_id, blocks in self.page_table.items():
            for bid in blocks:
                actual_page_table_counts[bid] = actual_page_table_counts.get(bid, 0) + 1
                
        for bid, actual_count in actual_page_table_counts.items():
            expected_count = self.block_info[bid].ref_count
            if actual_count != expected_count:
                errors.append(f"block {bid} has ref_count {expected_count} but appears {actual_count} times in page tables")
                
        free_set = set(self.free_blocks)
        for bid in actual_page_table_counts.keys():
            if bid in free_set:
                errors.append(f"block {bid} appears in both free_blocks and page_table")
            if bid in self.cached_free:
                errors.append(f"block {bid} appears in both cached_free and page_table")
                
        for bid in self.cached_free:
            if bid in free_set:
                errors.append(f"block {bid} appears in both free_blocks and cached_free")
                
        return errors

    def register_prompt_blocks(self, request_id: str, token_ids: list[int]) -> int:
        if not self.prefix_cache_enabled:
            return 0
            
        hashes = compute_block_hashes(token_ids)
        newly_hashed = 0
        blocks = self.page_table.get(request_id, [])
        
        for block_idx, hash_hex in hashes:
            if block_idx >= len(blocks):
                continue
            phys_block = blocks[block_idx]
            if self.block_info[phys_block].block_hash is not None:
                continue # never overwrite
                
            # If a collision with a different existing block happens, skip registering this one
            # rather than overwriting. This is a rare-but-possible edge case.
            if hash_hex in self.hash_registry and self.hash_registry[hash_hex] != phys_block:
                continue
                
            self.block_info[phys_block].block_hash = hash_hex
            if block_idx > 0:
                self.block_info[phys_block].parent_hash = hashes[block_idx - 1][1]
            else:
                self.block_info[phys_block].parent_hash = GENESIS_PARENT_HASH.hex()
            
            start_idx = block_idx * BLOCK_SIZE
            end_idx = min(start_idx + BLOCK_SIZE, len(token_ids))
            self.block_info[phys_block].token_ids = tuple(token_ids[start_idx:end_idx])
            
            self.hash_registry[hash_hex] = phys_block
            newly_hashed += 1
            
        return newly_hashed

    def acquire_prefix(self, request_id: str, token_ids: list[int]) -> int:
        # this method is only for NEW sequences, never for one already admitted
        if request_id in self.page_table:
            raise RuntimeError(f"Cannot acquire prefix for request_id {request_id} that is already admitted")
            
        if not self.prefix_cache_enabled:
            return 0
            
        hashes = compute_block_hashes(token_ids)
        hits = 0
        hit_blocks = []
        
        for block_idx, hash_hex in hashes:
            if hash_hex not in self.hash_registry:
                break # this and all later blocks are misses
                
            phys_block = self.hash_registry[hash_hex]
            
            start_idx = block_idx * BLOCK_SIZE
            end_idx = min(start_idx + BLOCK_SIZE, len(token_ids))
            expected_token_ids = tuple(token_ids[start_idx:end_idx])
            expected_parent_hash = hashes[block_idx - 1][1] if block_idx > 0 else GENESIS_PARENT_HASH.hex()
            
            if self.block_info[phys_block].token_ids != expected_token_ids or self.block_info[phys_block].parent_hash != expected_parent_hash:
                break
                
            self.block_info[phys_block].ref_count += 1
            self._clock += 1
            self.block_info[phys_block].last_used = self._clock
            
            if phys_block in self.cached_free:
                # it's no longer just sitting in reserve, it's actively in use again
                del self.cached_free[phys_block]
                
            hit_blocks.append(phys_block)
            hits += 1
            
        if hit_blocks:
            page_list = self.page_table.setdefault(request_id, [])
            for pb in reversed(hit_blocks):
                page_list.insert(0, pb)
                
        return hits

    def build_prefix_cache(self, block_ids: list[int]) -> DynamicCache:
        combined = DynamicCache()
        if not block_ids:
            return combined
            
        seq_len = len(block_ids) * BLOCK_SIZE
        
        for layer_idx in range(NUM_LAYERS):
            layer_keys = torch.zeros((1, NUM_KV_HEADS, seq_len, HEAD_DIM), dtype=self.dtype, device=self.device)
            layer_vals = torch.zeros((1, NUM_KV_HEADS, seq_len, HEAD_DIM), dtype=self.dtype, device=self.device)
            
            for logical_block_idx, phys_block in enumerate(block_ids):
                start_idx = logical_block_idx * BLOCK_SIZE
                end_idx = start_idx + BLOCK_SIZE # Prefix blocks are always complete
                
                layer_keys[0, :, start_idx:end_idx, :] = self.physical_keys[phys_block, layer_idx, :, :BLOCK_SIZE, :]
                layer_vals[0, :, start_idx:end_idx, :] = self.physical_values[phys_block, layer_idx, :, :BLOCK_SIZE, :]
                
            if hasattr(combined, 'layers'):
                layer = DynamicLayer()
                layer.keys = layer_keys
                layer.values = layer_vals
                layer.is_initialized = True
                combined.layers.append(layer)
            else:
                combined.key_cache.append(layer_keys)
                combined.value_cache.append(layer_vals)
                
        return combined
