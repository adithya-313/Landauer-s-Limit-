import sys
import os
import time
import json
import queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime_core.batch_engine import BatchEngine
from runtime_core.kv_cache_manager import KVCacheManager

def get_latest_log_for_request(request_id):
    try:
        with open("batch_events.jsonl", "r") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                event = json.loads(line.strip())
                if "requests_admitted" in event and request_id in event["requests_admitted"]:
                    stats = event.get("prefix_stats", [])
                    for stat in stats:
                        if stat["request_id"] == request_id:
                            return stat
            except:
                pass
    except FileNotFoundError:
        pass
    return None

def wait_for_response(response_q):
    tokens = []
    while True:
        try:
            msg = response_q.get(timeout=30)
            if msg["type"] == "token":
                tokens.append(msg["content"])
            elif msg["type"] == "done":
                break
            elif msg["type"] == "error":
                print("Error from engine:", msg)
                break
        except queue.Empty:
            print("Timeout waiting for response")
            break
    return "".join(tokens)

def main():
    if os.path.exists("batch_events.jsonl"):
        os.remove("batch_events.jsonl")

    print("--- Starting BatchEngine ---")
    engine = BatchEngine()
    engine.start()
    time.sleep(2) # Give it a moment to spin up

    try:
        print("\n=== CASE 1: Two requests with same system prompt ===")
        sys_prompt = "You are a helpful assistant. Always be concise."
        
        req1_q = queue.Queue()
        engine.pending_queue.put({
            "id": "Req1",
            "prompt": "",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "response_queue": req1_q,
            "max_tokens": 10
        })
        
        ans1 = wait_for_response(req1_q)
        print("Req1 Output:", repr(ans1))
        
        # Wait a bit for processing and logging
        time.sleep(1)
        
        req2_q = queue.Queue()
        engine.pending_queue.put({
            "id": "Req2",
            "prompt": "",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "response_queue": req2_q,
            "max_tokens": 10
        })
        
        ans2 = wait_for_response(req2_q)
        print("Req2 Output:", repr(ans2))
        
        time.sleep(1)
        stat2 = get_latest_log_for_request("Req2")
        print("Req2 Prefix Stats:", stat2)
        if stat2 and stat2.get("prefix_hit") and stat2.get("prefix_blocks_reused", 0) > 0:
            print("CASE 1 PASS")
        else:
            print("CASE 1 FAIL: Expected prefix hit for Req2")

        print("\n=== CASE 2: Evict blocks, test cold fallback ===")
        # Manually clear the hash registry and cached free blocks to simulate eviction
        engine.kv_manager.hash_registry.clear()
        for b in list(engine.kv_manager.cached_free.keys()):
            engine.kv_manager.free_blocks.append(b)
        engine.kv_manager.cached_free.clear()
        # Also clear free blocks just to be safe they aren't hashed
        
        req3_q = queue.Queue()
        engine.pending_queue.put({
            "id": "Req3",
            "prompt": "",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is the capital of Spain?"}
            ],
            "response_queue": req3_q,
            "max_tokens": 6
        })
        
        ans3 = wait_for_response(req3_q)
        print("Req3 Output:", repr(ans3))
        
        time.sleep(1)
        stat3 = get_latest_log_for_request("Req3")
        print("Req3 Prefix Stats:", stat3)
        if stat3 and not stat3.get("prefix_hit"):
            print("CASE 2 PASS")
        else:
            print("CASE 2 FAIL: Expected prefix miss for Req3")

        print("\n=== CASE 3: Deliberate hit-path failure ===")
        # We need a hit first, so let's send Req4 to cache something
        req4_q = queue.Queue()
        engine.pending_queue.put({
            "id": "Req4",
            "prompt": "",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is 3+3?"}
            ],
            "response_queue": req4_q,
            "max_tokens": 10
        })
        wait_for_response(req4_q)
        time.sleep(1)
        
        # Monkeypatch build_prefix_cache to raise an exception
        original_build = engine.kv_manager.build_prefix_cache
        def broken_build(*args, **kwargs):
            raise ValueError("Deliberate injection for CASE 3")
        engine.kv_manager.build_prefix_cache = broken_build
        
        req5_q = queue.Queue()
        engine.pending_queue.put({
            "id": "Req5",
            "prompt": "",
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is 4+4?"}
            ],
            "response_queue": req5_q,
            "max_tokens": 10
        })
        
        ans5 = wait_for_response(req5_q)
        print("Req5 Output:", repr(ans5))
        
        time.sleep(1)
        stat5 = get_latest_log_for_request("Req5")
        print("Req5 Prefix Stats:", stat5)
        if stat5 and not stat5.get("prefix_hit"):
            print("CASE 3 PASS")
        else:
            print("CASE 3 FAIL: Expected graceful fallback to cold path for Req5")
            
        # Restore monkeypatch
        engine.kv_manager.build_prefix_cache = original_build

        print("\n=== CASE 4: check_invariants ===")
        time.sleep(1) # Ensure engine is idle
        errors = engine.kv_manager.check_invariants()
        print("Invariant errors:", errors)
        if not errors:
            print("CASE 4 PASS")
        else:
            print("CASE 4 FAIL")
            # Debug the missing block
            all_blocks = set(range(engine.kv_manager.max_blocks))
            in_free = set(engine.kv_manager.free_blocks)
            in_cached = set(engine.kv_manager.cached_free.keys())
            in_use = {b for b, info in enumerate(engine.kv_manager.block_info) if info.ref_count > 0}
            missing = all_blocks - in_free - in_cached - in_use
            print(f"Missing blocks: {missing}")
            for m in missing:
                info = engine.kv_manager.block_info[m]
                print(f"Block {m} info: ref_count={info.ref_count}, hash={info.block_hash}, parent={info.parent_hash}")
                # check if it's in page table
                for req_id, blocks in engine.kv_manager.page_table.items():
                    if m in blocks:
                        print(f"Block {m} is in page_table for {req_id}")

    finally:
        engine.stop()

if __name__ == "__main__":
    main()
