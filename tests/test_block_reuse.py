import asyncio
import httpx

async def send_req(client, prompt, tier, max_tokens, engine):
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "engine": engine,
        "tier": tier,
        "max_tokens": max_tokens
    }
    print(f"[{engine}] Sending {tier} request: {prompt}")
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if "error" in line.lower():
                    print(f"[{engine}] ERROR in stream: {line}")
        print(f"[{engine}] Finished {tier} request: {prompt[:10]}...")
    except Exception as e:
        print(f"[{engine}] Failed: {e}")

async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8004", timeout=300.0) as client:
        print("\nTesting Custom Runtime Block Reuse...")
        
        # Fire 1 short request and 1 long request concurrently.
        # Short request hits EOS quickly.
        prompts = [
            ("premium", "Say exactly the word Hello and nothing else.", 10),
            ("premium", "Write a very long and detailed essay about the history of Rome. Write at least 200 words.", 200),
        ]
        
        tasks = []
        for tier, prompt, mt in prompts:
            tasks.append(send_req(client, prompt, tier, mt, "custom_runtime"))
            await asyncio.sleep(0.5) # stagger slightly
            
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
