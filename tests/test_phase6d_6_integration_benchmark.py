import sys
import os
import time
import json
import queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime_core.batch_engine import BatchEngine

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

    print("--- Starting BatchEngine for 6d-6 Integration ---")
    engine = BatchEngine()
    engine.start()
    time.sleep(2) # Give it a moment to spin up

    try:
        print("\n=== Test 1: Hit path with submit() ===")
        sys_prompt = "You are a helpful assistant. Always be concise."
        
        # Request 1 (fills cache)
        req1_id, req1_q = engine.submit(
            prompt="", 
            max_tokens=10, 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is 2+2?"}
            ]
        )
        ans1 = wait_for_response(req1_q)
        print(f"Req1 ({req1_id}) Output:", repr(ans1))
        
        time.sleep(1)
        
        # Request 2 (should hit)
        req2_id, req2_q = engine.submit(
            prompt="", 
            max_tokens=10, 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        )
        ans2 = wait_for_response(req2_q)
        print(f"Req2 ({req2_id}) Output:", repr(ans2))
        
        time.sleep(1)
        stat2 = get_latest_log_for_request(req2_id)
        print("Req2 Prefix Stats:", stat2)
        
        if stat2 and stat2.get("prefix_hit") and stat2.get("prefix_blocks_reused", 0) > 0:
            ttft = stat2.get("ttft_seconds")
            if ttft is not None and isinstance(ttft, float) and 0 < ttft < 60:
                print("Test 1 PASS")
            else:
                print(f"Test 1 FAIL: Invalid ttft_seconds: {ttft}")
        else:
            print("Test 1 FAIL: Expected prefix hit for Req2")

        print("\n=== Test 2: Evict cache, cold fallback with submit() ===")
        # Evict cache completely
        engine.kv_manager.hash_registry.clear()
        for b in list(engine.kv_manager.cached_free.keys()):
            engine.kv_manager.free_blocks.append(b)
        engine.kv_manager.cached_free.clear()
        
        # Request 3 (should miss)
        req3_id, req3_q = engine.submit(
            prompt="", 
            max_tokens=6, 
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": "What is the capital of Spain?"}
            ]
        )
        ans3 = wait_for_response(req3_q)
        print(f"Req3 ({req3_id}) Output:", repr(ans3))
        
        time.sleep(1)
        stat3 = get_latest_log_for_request(req3_id)
        print("Req3 Prefix Stats:", stat3)
        
        if stat3 and not stat3.get("prefix_hit"):
            ttft = stat3.get("ttft_seconds")
            if ttft is not None and isinstance(ttft, float) and 0 < ttft < 60:
                print("Test 2 PASS")
            else:
                print(f"Test 2 FAIL: Invalid ttft_seconds: {ttft}")
        else:
            print("Test 2 FAIL: Expected prefix miss for Req3")

    finally:
        engine.stop()
        print("\nBatchEngine stopped.")

if __name__ == "__main__":
    main()
