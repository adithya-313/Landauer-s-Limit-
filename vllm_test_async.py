import asyncio
import time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

async def main():
    engine_args = AsyncEngineArgs(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.75,
        max_model_len=1024,
        enforce_eager=True,
        disable_log_requests=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams(max_tokens=15)
    
    prompt = "You are a helpful assistant. " * 50 + "What is 2+2?"
    
    for i in range(2):
        print(f"--- Request {i+1} ---")
        start_time = time.time()
        ttft = None
        
        request_id = f"req_{i}"
        generator = engine.generate(prompt, sampling_params, request_id)
        
        async for request_output in generator:
            if ttft is None and request_output.outputs[0].text != "":
                ttft = time.time() - start_time
                print(f"TTFT: {ttft:.4f} seconds")
        
        print(f"Total time: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
