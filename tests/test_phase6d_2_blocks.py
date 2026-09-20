import sys
import torch
import dataclasses
from runtime_core.kv_cache_manager import KVCacheManager

@dataclasses.dataclass
class MockState:
    request_id: str
    prompt_len: int = 16
    tokens_produced: int = 0
    tier: str = "free"

def run_tests():
    total_cases = 0
    passed = 0
    failed = 0
    
    def assert_case(condition, name):
        nonlocal total_cases, passed, failed
        total_cases += 1
        if condition:
            print(f"PASS: {name}")
            passed += 1
        else:
            print(f"FAIL: {name}")
            failed += 1

    print("Running pure unit tests for Phase 6d-2 (Block Allocator)")
    device = torch.device("cpu")
    
    # 1. Double-free is a no-op
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    state1 = MockState(request_id="req-1")
    kv.ensure_allocation(state1, logical_length=32, active_slots=[state1]) # allocates 2 blocks
    free_blocks_before = len(kv.free_blocks)
    kv.free_sequence("req-1")
    free_blocks_after_first = len(kv.free_blocks)
    kv.free_sequence("req-1")
    free_blocks_after_second = len(kv.free_blocks)
    
    assert_case(
        free_blocks_before == 6 and free_blocks_after_first == 8 and free_blocks_after_second == 8,
        "Double-free is a no-op (free_blocks count remains unchanged on second call)"
    )

    # 2. ref_count never goes negative through normal use
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    state2 = MockState(request_id="req-2")
    try:
        kv.ensure_allocation(state2, logical_length=16, active_slots=[state2])
        kv.free_sequence("req-2")
        raised = False
    except RuntimeError:
        raised = True
    assert_case(not raised, "ref_count never goes negative through normal allocate/free cycle")

    # 3. Fragmentation calculation matches prior (non-shared) behavior
    kv = KVCacheManager(device=device, max_blocks=8, dtype=torch.float32)
    state3 = MockState(request_id="req-3", prompt_len=18) # 2 blocks allocated, 18 tokens used
    kv.ensure_allocation(state3, logical_length=18, active_slots=[state3])
    
    logged_stats = {}
    def mock_log(stats):
        logged_stats.update(stats)
        
    kv.last_log_time = 0.0 # Force log
    kv.log_fragmentation_if_needed([state3], mock_log)
    
    # 2 blocks * 16 = 32 total allocated tokens.
    assert_case(
        logged_stats.get("allocated_blocks") == 2 and logged_stats.get("total_allocated_tokens") == 32,
        "Fragmentation calculation (allocated_blocks/total_allocated_tokens) matches manual block count"
    )

    # 4. check_invariants() returns [] after a normal sequence
    assert_case(len(kv.check_invariants()) == 0, "check_invariants() returns [] after normal use")
    
    # 5. check_invariants() correctly CATCHES a manually broken state
    kv.block_info[0].ref_count = 999 # deliberately corrupt
    errors = kv.check_invariants()
    assert_case(
        len(errors) > 0 and any("ref_count" in err and "999" in err for err in errors),
        "check_invariants() catches manually corrupted ref_count"
    )

    print("\n--- TEST SUMMARY ---")
    print(f"Total Cases Run: {total_cases}")
    print(f"Total Passed:    {passed}")
    print(f"Total Failed:    {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
