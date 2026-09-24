import json
import os
import sys

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from tests.test_phase6d_6_benchmark import generate_system_prompt, run_condition

def main():
    print("Setting up prompts...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    
    sys500, len500 = generate_system_prompt(tokenizer, 500)
    print(f"Generated sys_prompt_500: actual length {len500} tokens (target 500)")
    
    # Verification
    print(f"Verification: 500 tokens requires ~32 blocks (block_size=16). Total KV cache pool is 2000 blocks. 32 << 2000. Fits safely.")
    
    with open("test_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["turns"][0]["content"] for item in dataset[:20]]
    
    # Run Condition A: Cache Disabled
    cond_a = run_condition("Cache Disabled (500-token prompt)", False, sys500, questions, max_tokens=40)
    
    # Run Condition B: Cache Enabled
    cond_b = run_condition("Cache Enabled (500-token prompt)", True, sys500, questions, max_tokens=40)
    
    results = {
        "cache_disabled_500": cond_a,
        "cache_enabled_500": cond_b
    }
    
    with open("supplementary_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Saved supplementary_results.json")

if __name__ == "__main__":
    main()
