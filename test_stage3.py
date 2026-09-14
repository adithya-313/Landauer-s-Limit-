import asyncio
import httpx
import time

async def send_request(engine: str, prompt: str, delay: float = 0):
    await asyncio.sleep(delay)
    print(f"[{time.time():.2f}] Sending request to {engine}: {prompt}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8001/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": prompt}],
                    "engine": engine
                },
                timeout=60.0
            )
            # SSE stream comes back, we just read the raw text to see if it succeeded
            content = response.text
            print(f"\n--- {engine} RESPONSE ---\n{content[:200]}...\n------------------")
            return content
        except Exception as e:
            print(f"Error on {engine}: {e}")
            return str(e)

async def main():
    # 4 concurrent requests to custom_runtime to prove continuous batching
    tasks = [
        send_request("custom_runtime", "Write a short poem about the ocean.", 0.0),
        send_request("custom_runtime", "What is the capital of France?", 0.5),
        send_request("custom_runtime", "Explain relativity in one sentence.", 1.0),
        send_request("custom_runtime", "Count from 1 to 5.", 1.5),
        
        # 1 to vllm_local to prove it's unaffected
        send_request("vllm_local", "vLLM test prompt", 2.0),
        
        # 1 to llamacpp_local to prove it's unaffected
        send_request("llamacpp_local", "llama test prompt", 2.5),
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
