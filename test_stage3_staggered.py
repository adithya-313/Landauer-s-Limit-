import asyncio
import httpx
import time

async def send_request(engine: str, prompt: str, delay: float = 0):
    """
    Waits for a specific delay, then sends the question to the AI server.
    """
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
            # We just read the raw text stream to prove it worked independently
            content = response.text
            print(f"\n--- {engine} RESPONSE (Delay {delay}s) ---\n{content[:200]}...\n------------------")
            return content
        except Exception as e:
            print(f"Error on {engine} (Delay {delay}s): {e}")
            return str(e)

async def main():
    """
    Sends 6 staggered requests to recreate a 'revolving door' scenario, 
    where people arrive at different times and leave at different times.
    Because MAX_SLOTS is 4, requests 5 and 6 should be queued or admitted 
    as soon as others finish.
    """
    print("Sending warmup request to initialize engine...")
    await send_request("custom_runtime", "Warmup", 0.0)
    print("Warmup complete. Starting staggered test in 2 seconds...")
    await asyncio.sleep(2.0)
    
    # Clear the old batch_events.jsonl so we get a clean read for this test run
    open("batch_events.jsonl", "w").close()
    
    tasks = [
        # Req 1 arrives at 0s
        send_request("custom_runtime", "Write a short poem about the ocean.", 0.0),
        # Req 2 arrives at 4s
        send_request("custom_runtime", "What is the capital of France?", 4.0),
        # Req 3 arrives at 8s
        send_request("custom_runtime", "Explain relativity in one sentence.", 8.0),
        # Req 4 arrives at 12s
        send_request("custom_runtime", "Count from 1 to 5.", 12.0),
        
        # Since generating 200 tokens takes ~15-20 seconds, Req 1 should finish 
        # around 15-20s. Req 5 arrives at 22s to grab Req 1's newly freed slot 
        # while Reqs 2, 3, 4 are STILL running!
        send_request("custom_runtime", "Why is the sky blue?", 22.0),
        
        # Req 6 arrives at 26s, taking another freed slot.
        send_request("custom_runtime", "Write a haiku about computers.", 26.0),
    ]
    
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
