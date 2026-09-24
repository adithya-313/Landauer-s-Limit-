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
    return tokens

def generate_system_prompt(tokenizer, target_length):
    # Overprovision words
    text = "You are a helpful and highly capable AI assistant. " * (target_length // 2 + 5)
    tokens = tokenizer.encode(text)
    # Truncate exact
    trimmed = tokenizer.decode(tokens[:target_length])
    actual_len = len(tokenizer.encode(trimmed))
    return trimmed, actual_len

def run_condition(condition_name, enable_cache, sys_prompt, questions, max_tokens=40):
    print(f"\n========== RUNNING CONDITION: {condition_name} ==========")
    if os.path.exists("batch_events.jsonl"):
        os.remove("batch_events.jsonl")

    engine = BatchEngine(enable_prefix_cache=enable_cache)
    engine.start()
    time.sleep(2)

    results = []
    
    try:
        for i, q_text in enumerate(questions):
            req_id, req_q = engine.submit(
                prompt="",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q_text}
                ]
            )
            
            # Wait for response
            tokens = wait_for_response(req_q)
            
            # Determine if it's a loop
            # A natural truncation just means max_tokens is too low. A real loop has repetitive output.
            # We check if the last 6 tokens consist of only 1 or 2 unique tokens (typical of 'Madrid Madrid' or 'a a a').
            is_loop = False
            if len(tokens) >= max_tokens:
                last_few = tokens[-6:]
                if len(set(last_few)) <= 2:
                    is_loop = True
            
            time.sleep(0.5) # Let logs flush
            stat = get_latest_log_for_request(req_id)
            if not stat:
                stat = {}
                
            ttft = stat.get("ttft_seconds")
            hit = stat.get("prefix_hit", False)
            saved = stat.get("prefix_tokens_saved", 0)
            
            results.append({
                "request_id": req_id,
                "ttft": ttft,
                "hit": hit,
                "saved": saved,
                "possible_loop": is_loop
            })
            
            print(f"Req {i+1}/20 - Loop: {is_loop}, Hit: {hit}, TTFT: {ttft:.4f}s" if ttft else f"Req {i+1}/20 - Error")
            
    finally:
        engine.stop()
        
    return results

def main():
    # 1. Setup tokenizer for length generation
    print("Setting up prompts...")
    # Instantiate engine briefly just to get tokenizer, or load it directly
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    sys20, len20 = generate_system_prompt(tokenizer, 20)
    sys200, len200 = generate_system_prompt(tokenizer, 200)
    print(f"Generated sys_prompt_20: actual length {len20} tokens (target 20)")
    print(f"Generated sys_prompt_200: actual length {len200} tokens (target 200)")
    
    # 2. Load 20 questions
    with open("test_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["turns"][0]["content"] for item in dataset[:20]]
    
    print(f"Loaded {len(questions)} questions.")
    
    # 3. Run conditions
    cond_a = run_condition("Cache Disabled (using 200-token prompt)", False, sys200, questions)
    cond_b = run_condition("Cache Enabled (20-token prompt)", True, sys20, questions)
    cond_c = run_condition("Cache Enabled (200-token prompt)", True, sys200, questions)
    
    # 4. Summarize results
    def summarize(name, data):
        hits = [d for d in data if d["hit"]]
        misses = [d for d in data if not d["hit"]]
        loops = [d for d in data if d["possible_loop"]]
        
        hit_rate = len(hits) / len(data) if data else 0
        avg_saved = sum(d["saved"] for d in hits) / len(hits) if hits else 0
        
        # TTFT for hits
        hit_ttfts = [d["ttft"] for d in hits if d["ttft"] is not None]
        avg_hit_ttft = sum(hit_ttfts) / len(hit_ttfts) if hit_ttfts else 0
        
        # TTFT for misses
        miss_ttfts = [d["ttft"] for d in misses if d["ttft"] is not None]
        avg_miss_ttft = sum(miss_ttfts) / len(miss_ttfts) if miss_ttfts else 0
        
        print(f"\n--- {name} Summary ---")
        print(f"Hit Rate: {hit_rate:.2%} ({len(hits)}/{len(data)})")
        print(f"Avg Tokens Saved per Hit: {avg_saved:.1f}")
        print(f"Avg TTFT (Hits):  {avg_hit_ttft:.4f}s")
        print(f"Avg TTFT (Misses): {avg_miss_ttft:.4f}s")
        print(f"Possible Loops: {len(loops)}/{len(data)}")
        
        return {
            "hit_rate": hit_rate,
            "avg_saved_per_hit": avg_saved,
            "avg_ttft_hits": avg_hit_ttft,
            "avg_ttft_misses": avg_miss_ttft,
            "loops": len(loops),
            "raw": data
        }

    out = {
        "cache_disabled": summarize("Cache Disabled", cond_a),
        "cache_20": summarize("Cache Enabled (20-token)", cond_b),
        "cache_200": summarize("Cache Enabled (200-token)", cond_c)
    }
    
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    main()
