import json
import torch
import gc
from batch_engine import BatchEngine, MAX_SLOTS, TOKEN_BUDGET

def main():
    engine = BatchEngine()
    
    # Mix of short and long prompts
    prompts = [
        {"id": "req-1", "prompt": "What is the capital of France? Answer in one word."},
        {"id": "req-2", "prompt": "Explain the theory of relativity in simple terms. Be very descriptive and use analogies."},
        {"id": "req-3", "prompt": "Write a python script to reverse a string."},
        {"id": "req-4", "prompt": "Summarize the plot of Romeo and Juliet."},
        {"id": "req-5", "prompt": "Translate 'Hello world' to Spanish."},
        {"id": "req-6", "prompt": "What are the primary colors? List them."}
    ]
    
    # Stagger arrivals: req-1 at 0s, req-2 at 0.5s, req-3 at 1.0s, etc.
    arrival_delays = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    print("Running batch engine with staggered arrivals...")
    outputs = engine.run_queue(prompts, arrival_delays)
    
    print("\n--- GENERATED OUTPUTS ---")
    for req_id, text in outputs.items():
        print(f"[{req_id}] -> {text.strip()[:100]}...")
        
    print("\n--- MEASURING VRAM ---")
    vram_alloc = torch.cuda.memory_allocated() / (1024**2)
    vram_res = torch.cuda.memory_reserved() / (1024**2)
    print(f"VRAM Allocated: {vram_alloc:.2f} MB")
    print(f"VRAM Reserved: {vram_res:.2f} MB")
    
    print("\n--- PARSING BATCH EVENTS ---")
    events = []
    with open("batch_events.jsonl", "r") as f:
        for line in f:
            events.append(json.loads(line))
            
    active_slot_states = set()
    max_active_seen = 0
    max_budget_seen = 0
    
    for event in events:
        active = tuple(event["active_requests"])
        active_slot_states.add(active)
        
        num_active = len(active)
        if num_active > max_active_seen:
            max_active_seen = num_active
            
        budget = event["current_total_live_tokens"]
        if budget > max_budget_seen:
            max_budget_seen = budget
            
        if num_active > MAX_SLOTS:
            print(f"FAILED: Step exceeded MAX_SLOTS ({num_active} > {MAX_SLOTS})")
            
        if budget > TOKEN_BUDGET:
            print(f"FAILED: Step exceeded TOKEN_BUDGET ({budget} > {TOKEN_BUDGET})")
            
    print(f"Unique active slot compositions seen: {len(active_slot_states)}")
    print(f"Max active slots used: {max_active_seen}")
    print(f"Max token budget used: {max_budget_seen}")
    
    if len(active_slot_states) >= 3:
        print("PASS: Active slot compositions changed across at least 3 steps.")
    else:
        print("FAIL: Active slot compositions did not change enough.")
        
    if len(outputs) == len(prompts) and all(len(v.strip()) > 0 for v in outputs.values()):
        print("PASS: Every prompt produced a complete, non-empty output.")
    else:
        print("FAIL: Some prompts failed to produce output.")
        
    print("\nSample batch events excerpt showing composition changes:")
    last_state = None
    for event in events:
        active = tuple(event["active_requests"])
        if active != last_state:
            print(f"Time: {event['timestamp']:.2f}, Active: {event['active_requests']}, "
                  f"Admitted: {event['requests_admitted']}, Freed: {event['slots_freed']}")
            last_state = active

if __name__ == "__main__":
    main()
