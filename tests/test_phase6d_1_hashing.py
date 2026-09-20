import sys
from runtime_core.prefix_hashing import compute_block_hashes, max_shareable_blocks

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

    print("Running pure unit tests for Phase 6d-1 (Hashing)")

    # 1. Same token_ids list twice -> identical hash output
    base_tokens = list(range(32))
    hash1 = compute_block_hashes(base_tokens)
    hash2 = compute_block_hashes(base_tokens)
    assert_case(hash1 == hash2 and len(hash1) == 2, "Same token_ids list twice yields identical hashes")

    # 2. Same token_ids except one token changed in block 0 -> block 0 and 1 hash changes
    changed_tokens = list(range(32))
    changed_tokens[5] = 999
    hash_changed = compute_block_hashes(changed_tokens)
    assert_case(
        hash1[0][1] != hash_changed[0][1] and hash1[1][1] != hash_changed[1][1],
        "One token changed in block 0 changes both block 0 and block 1 hashes (chaining works)"
    )

    # 3. Identical 16-token content but hashed as block 0 vs artificially treated as a later block -> different hash
    # To construct this, we take tokens 16-31 of base_tokens and hash them directly (they become block 0).
    # Then we compare that hash to block 1 of the original 32-token sequence.
    block_1_content = list(range(16, 32))
    block_0_fake = compute_block_hashes(block_1_content)
    assert_case(
        block_0_fake[0][1] != hash1[1][1],
        "Identical 16-token content produces different hash when evaluated as block 0 vs block 1"
    )

    # 4. max_shareable_blocks edge cases
    assert_case(max_shareable_blocks(0) == 0, "max_shareable_blocks(0) == 0")
    assert_case(max_shareable_blocks(1) == 0, "max_shareable_blocks(1) == 0")
    assert_case(max_shareable_blocks(16) == 0, "max_shareable_blocks(16) == 0")
    assert_case(max_shareable_blocks(17) == 1, "max_shareable_blocks(17) == 1")
    assert_case(max_shareable_blocks(32) == 1, "max_shareable_blocks(32) == 1")
    assert_case(max_shareable_blocks(33) == 2, "max_shareable_blocks(33) == 2")

    # 5. Empty list input -> []
    assert_case(compute_block_hashes([]) == [], "Empty list returns empty list")

    # 6. ValueError cases
    def test_value_error(input_data, case_name):
        try:
            compute_block_hashes(input_data)
            return False
        except ValueError:
            return True
        except Exception:
            return False

    assert_case(test_value_error([-1, 2, 3], "negative int"), "ValueError raised on negative int")
    assert_case(test_value_error([1, 2.5, 3], "float inside list"), "ValueError raised on non-int (float)")
    assert_case(test_value_error([1, "a", 3], "string inside list"), "ValueError raised on non-int (string)")
    assert_case(test_value_error((1, 2, 3), "tuple instead of list"), "ValueError raised on tuple input")
    assert_case(test_value_error(None, "None input"), "ValueError raised on None input")

    print("\n--- TEST SUMMARY ---")
    print(f"Total Cases Run: {total_cases}")
    print(f"Total Passed:    {passed}")
    print(f"Total Failed:    {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
