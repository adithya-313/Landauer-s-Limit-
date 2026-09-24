import sys
import os
import time
import json
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

def generate_system_prompt(tokenizer, target_length):
    text = "You are a helpful and highly capable AI assistant. " * (target_length // 2 + 5)
    tokens = tokenizer.encode(text)
    trimmed = tokenizer.decode(tokens[:target_length])
    return trimmed

async def run_condition(enable_cache, sys_prompt_len):
    # Setup vLLM Engine
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_prefix_caching=enable_cache,
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        enforce_eager=True,
        disable_log_stats=True
    )
    print("Starting vLLM engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Load dataset
    with open("test_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = [item["turns"][0]["content"] for item in dataset[:20]]
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    sys_prompt = generate_system_prompt(tokenizer, sys_prompt_len)
    
    sampling_params = SamplingParams(max_tokens=40, temperature=0.0)
    
    results = []
    
    for i, q_text in enumerate(questions):
        full_prompt = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n{q_text}<|im_end|>\n<|im_start|>assistant\n"
        
        req_id = f"req_{i}"
        
        start_time = time.time()
        ttft = None
        num_tokens = 0
        
        generator = engine.generate(full_prompt, sampling_params, req_id)
        async for request_output in generator:
            text = request_output.outputs[0].text
            if ttft is None and text != "":
                ttft = time.time() - start_time
            num_tokens = len(request_output.outputs[0].token_ids)
                
        is_loop = False
        if num_tokens >= 40:
            # We don't have direct access to the tokens in this simple async for loop text extraction easily,
            # but we can check if the generated text ends in a highly repetitive pattern.
            words = text.split()
            if len(words) >= 4 and len(set(words[-4:])) <= 1:
                is_loop = True
        
        # but we can deduce hit rate from TTFT!
        # If TTFT < 0.2s, it's a hit, otherwise it's a miss (a cold miss takes ~0.5s - 1.0s)
        # We will log the raw TTFT and deduce it later.
        
        results.append({
            "request_id": req_id,
            "ttft": ttft,
            "possible_loop": is_loop
        })
        
        print(f"Req {i+1}/20 - TTFT: {ttft:.4f}s - Loop: {is_loop}")
        
    return results

if __name__ == "__main__":
    enable_cache = sys.argv[1].lower() == "true"
    sys_prompt_len = int(sys.argv[2])
    
    out_file = f"vllm_results_{sys_prompt_len}_{enable_cache}.json"
    
    results = asyncio.run(run_condition(enable_cache, sys_prompt_len))
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
