import sys
import torch
import dataclasses
import queue
from runtime_core.kv_cache_manager import KVCacheManager
from runtime_core.prefix_hashing import compute_block_hashes

@dataclasses.dataclass
class MockState:
    request_id: str
    prompt_len: int = 16
    tokens_produced: int = 0
    tier: str = "free"
    finished: bool = False
    response_queue: queue.Queue = dataclasses.field(default_factory=queue.Queue)

def run_tests():
    total_cases = 0
    passed_cases = 0

    print("Running pure unit tests for Phase 6d-3 (Hash Registry & Cached-Free State)")
    
    def check(condition: bool, msg: str):
        nonlocal total_cases, passed_cases
        total_cases += 1
        if condition:
            print(f"PASS: {msg}")
            passed_cases += 1
        else:
            print(f"FAIL: {msg}")
            
    device = torch.device("cpu")
    
    # T1
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    stateA = MockState(request_id="A", prompt_len=48)
    kv.ensure_allocation(stateA, 48, [])
    seqA_tokens = [i for i in range(48)]
    hashed = kv.register_prompt_blocks("A", seqA_tokens)
    
    stateB = MockState(request_id="B", prompt_len=48)
    hits = kv.acquire_prefix("B", seqA_tokens)
    
    blocksA = kv.page_table["A"]
    blocksB = kv.page_table["B"]
    
    check(
        hashed == 3 and hits == 3 and blocksA == blocksB and all(kv.block_info[bid].ref_count == 2 for bid in blocksA),
        "T1: register_prompt_blocks and acquire_prefix work, shared blocks show ref_count == 2"
    )
    
    # T2
    kv.ensure_allocation(stateA, 64, [])
    blocksA_extended = kv.page_table["A"]
    private_block = blocksA_extended[-1]
    
    kv.free_sequence("A")
    check(
        all(kv.block_info[bid].ref_count == 1 for bid in blocksB) and kv.block_info[private_block].ref_count == 0 and private_block in kv.free_blocks,
        "T2: free_sequence on shared blocks decrements ref_count, frees private blocks"
    )
    
    # T3
    kv.free_sequence("B")
    check(
        all(bid in kv.cached_free for bid in blocksB) and all(kv.block_info[bid].ref_count == 0 for bid in blocksB),
        "T3(a): freed registered blocks enter cached_free"
    )
    
    stateC = MockState(request_id="C", prompt_len=48)
    hitsC = kv.acquire_prefix("C", seqA_tokens)
    blocksC = kv.page_table["C"]
    
    check(
        hitsC == 3 and blocksC == blocksB and all(bid not in kv.cached_free for bid in blocksC) and all(kv.block_info[bid].ref_count == 1 for bid in blocksC),
        "T3(b): acquire_prefix reclaims from cached_free"
    )
    
    # T4
    kv.free_sequence("C")
    stateD = MockState(request_id="D", prompt_len=128)
    kv.ensure_allocation(stateD, 8 * 16, [])
    
    check(
        len(kv.cached_free) == 0 and all(kv.block_info[bid].block_hash is None for bid in blocksC),
        "T4(a): pool exhaustion reclaims cached_free blocks, clearing their hash fields"
    )
    
    stateE = MockState(request_id="E")
    hitsE = kv.acquire_prefix("E", seqA_tokens)
    check(
        hitsE == 0,
        "T4(b): subsequent acquire_prefix misses after blocks are reclaimed"
    )
    
    # T5
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    state_shared1 = MockState(request_id="Shared1", tier="premium", prompt_len=32)
    kv.ensure_allocation(state_shared1, 32, [])
    seq_shared_tokens = [i for i in range(32)]
    kv.register_prompt_blocks("Shared1", seq_shared_tokens)
    
    state_shared2 = MockState(request_id="Shared2", tier="free", prompt_len=32)
    kv.acquire_prefix("Shared2", seq_shared_tokens)
    
    state_fill = MockState(request_id="Fill", tier="premium", prompt_len=96)
    kv.ensure_allocation(state_fill, 6 * 16, [state_shared1, state_shared2])
    
    state_OOM = MockState(request_id="OOM", tier="free", prompt_len=16)
    kv.ensure_allocation(state_OOM, 16, [state_shared1, state_shared2, state_fill])
    
    check(
        state_OOM.finished and not state_shared2.finished and not state_shared1.finished,
        "T5: all-shared sequence skipped for eviction, free tier OOMs instead of evicting premium"
    )
    
    # T6
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    stateF = MockState(request_id="F", prompt_len=16)
    kv.ensure_allocation(stateF, 16, [])
    seqF_tokens = [i for i in range(16)]
    kv.register_prompt_blocks("F", seqF_tokens)
    
    old_hash = kv.block_info[kv.page_table["F"][0]].block_hash
    kv.register_prompt_blocks("F", [99] * 16)
    new_hash = kv.block_info[kv.page_table["F"][0]].block_hash
    
    check(
        old_hash == new_hash,
        "T6(a): register_prompt_blocks does not overwrite existing hash"
    )
    
    kv.block_info[kv.page_table["F"][0]].token_ids = (1,) * 16
    hitsF = kv.acquire_prefix("F2", seqF_tokens)
    check(
        hitsF == 0,
        "T6(b): acquire_prefix misses on token_ids mismatch despite hash registry match"
    )
    
    # T7
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    kv.prefix_cache_enabled = False
    stateG = MockState(request_id="G", prompt_len=16)
    kv.ensure_allocation(stateG, 16, [])
    seqG_tokens = [i for i in range(16)]
    hashedG = kv.register_prompt_blocks("G", seqG_tokens)
    hitsG = kv.acquire_prefix("H", seqG_tokens)
    
    err_raised = False
    try:
        kv.acquire_prefix("G", seqG_tokens)
    except RuntimeError:
        err_raised = True
        
    check(
        hashedG == 0 and hitsG == 0 and err_raised,
        "T7: prefix_cache_enabled=False skips register/acquire, acquire_prefix on existing raises"
    )
    
    # T8
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    stateI = MockState(request_id="I", prompt_len=32)
    kv.ensure_allocation(stateI, 32, [])
    seqI = [i for i in range(32)]
    kv.register_prompt_blocks("I", seqI)
    
    stateJ = MockState(request_id="J", prompt_len=32)
    kv.acquire_prefix("J", seqI)
    
    captured = {}
    kv.log_fragmentation_if_needed([stateI, stateJ], lambda data: captured.update(data))
    
    check(
        captured.get("allocated_blocks") == 2 and captured.get("shared_blocks") == 2 and captured.get("hashed_blocks") == 2,
        "T8: log_fragmentation correctly accounts for shared blocks without double-counting"
    )
    
    print("\n--- TEST SUMMARY ---")
    print(f"Total Cases Run: {total_cases}")
    print(f"Total Passed:    {passed_cases}")
    print(f"Total Failed:    {total_cases - passed_cases}")

if __name__ == "__main__":
    run_tests()
