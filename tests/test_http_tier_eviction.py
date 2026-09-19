import asyncio
import httpx
import json

async def send_req(client, prompt, tier, engine):
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "engine": engine,
        "tier": tier,
        "max_tokens": 100
    }
    print(f"[{engine}] Sending {tier} request: {prompt}")
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if "error" in line.lower():
                    print(f"[{engine}] ERROR in stream: {line}")
    except Exception as e:
        print(f"[{engine}] Failed: {e}")

async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8003", timeout=300.0) as client:
        # Test vLLM (will likely fail with connection error, but shouldn't throw TypeError)
        print("Testing vLLM Adapter...")
        await send_req(client, "Hello vLLM", "free", "vllm_local")
        
        # Test Llama.cpp (will likely fail with connection error, but shouldn't throw TypeError)
        print("Testing Llama.cpp Adapter...")
        await send_req(client, "Hello Llama", "free", "llamacpp_local")
        
        print("\nTesting Custom Runtime Exhaustion via HTTP...")
        
        # Fire 5 concurrent requests: 3 premium, 2 free
        prompts = [
            ("premium", "Explain quantum mechanics in very long detail."),
            ("free", "Write a very long essay about the history of Rome."),
            ("premium", "Write a 5 page story about a dragon."),
            ("free", "Tell me everything about the industrial revolution."),
            ("premium", "Explain the theory of relativity comprehensively.")
        ]
        
        tasks = []
        for tier, prompt in prompts:
            tasks.append(send_req(client, prompt, tier, "custom_runtime"))
            await asyncio.sleep(0.5) # stagger slightly
            
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
