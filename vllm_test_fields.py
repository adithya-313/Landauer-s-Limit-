import sys
from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct", 
        enable_prefix_caching=True, 
        gpu_memory_utilization=0.75, 
        max_model_len=1024,
        enforce_eager=True
    )
    sampling_params = SamplingParams(max_tokens=15)
    
    # Send a prompt twice to trigger prefix caching
    prompt = "You are a helpful assistant. " * 50 + "What is 2+2?"
    print("--- First Request ---")
    outputs1 = llm.generate([prompt], sampling_params)
    
    print("--- Second Request ---")
    outputs2 = llm.generate([prompt], sampling_params)
    
    for i, outputs in enumerate([outputs1, outputs2]):
        out = outputs[0]
        print(f"--- Request {i+1} ---")
        print("Metrics:", dir(out.metrics))
        print("arrival_time:", out.metrics.arrival_time)
        print("first_token_time:", out.metrics.first_token_time)
        if out.metrics.first_token_time is not None and out.metrics.arrival_time is not None:
            print("ttft:", out.metrics.first_token_time - out.metrics.arrival_time)

if __name__ == "__main__":
    main()
