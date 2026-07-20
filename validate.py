"""Self-validation script for test_dataset.json"""
import json
import os
from collections import Counter

assert os.path.exists("test_dataset.json"), "test_dataset.json NOT FOUND"
print("[PASS] test_dataset.json exists")

with open("test_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == 200, f"Expected 200, got {len(data)}"
print(f"[PASS] Total count: {len(data)}")

ids = []
empty_prompts = 0
for item in data:
    ids.append(item["id"])
    if len(item["turns"]) < 2:
        empty_prompts += 1
    for turn in item["turns"]:
        if not turn["content"].strip():
            empty_prompts += 1
assert empty_prompts == 0, f"Found {empty_prompts} empty prompts"
print("[PASS] No empty prompts or malformed turns")

assert len(ids) == len(set(ids)), "Duplicate IDs found"
print(f"[PASS] No duplicate IDs ({len(set(ids))} unique)")

token_lengths = []
for item in data:
    tl = sum(len(turn["content"].split()) for turn in item["turns"])
    token_lengths.append(tl)

print(
    f"[PASS] Token-length stats — "
    f"Min: {min(token_lengths)}, "
    f"Max: {max(token_lengths)}, "
    f"Avg: {sum(token_lengths) // len(token_lengths)}, "
    f"Median: {sorted(token_lengths)[len(token_lengths) // 2]}"
)

print("  Histogram:")
for lo, hi in [(0, 50), (50, 200), (200, 500), (500, 1000), (1000, 2000)]:
    count = sum(1 for tl in token_lengths if lo <= tl < hi)
    if count:
        print(f"    [{lo:>4}-{hi:>4}]: {count}")

multi_turn = [item for item in data if len(item["turns"]) > 2]
assert len(multi_turn) > 0, "No multi-turn conversations found"
print(f"[PASS] Multi-turn (>2 turns): {len(multi_turn)}/{len(data)}")

drift_items = [item for item in data if item["drift_eval"]]
assert len(drift_items) == 20, f"Expected 20 drift_eval, got {len(drift_items)}"
print(f"[PASS] drift_eval=True count: {len(drift_items)}")

drift_cats = Counter(item["category"] for item in drift_items)
print(f"[PASS] drift_eval category spread: {dict(drift_cats)}")

print()
print("ALL VALIDATIONS PASSED")
