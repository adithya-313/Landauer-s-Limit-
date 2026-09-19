import asyncio
import httpx
import time

async def send_request(prompt: str, delay: float = 0):
    await asyncio.sleep(delay)
    print(f"[{time.time():.2f}] Sending request: {prompt}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://127.0.0.1:8001/v1/chat/completions",
                json={
                    "model": "test",
                    "messages": [{"role": "user", "content": prompt}],
                    "engine": "custom_runtime"
                },
                timeout=120.0
            )
            content = response.text
            print(f"\n--- RESPONSE ---\n{content[:200]}...\n------------------")
            return content
        except Exception as e:
            print(f"Error: {e}")
            return str(e)

async def main():
    print("Running staggered traffic test (Phase 6c)")
    tasks = [
        send_request("Write a short poem about the ocean.", 0.0),
        send_request("What is the capital of France?", 0.5),
        send_request("Explain relativity in one sentence.", 1.0),
        send_request("Count from 1 to 5.", 1.5),
        send_request("Write a detailed essay about artificial intelligence.", 2.0),
        send_request("Tell me a joke.", 2.5),
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
