import asyncio
import httpx
import time

async def send_request(engine: str, prompt: str):
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
            
            if response.status_code != 200:
                print(f"--- ERROR RESPONSE FOR '{prompt}' ---\nHTTP {response.status_code}: {response.text}\n------------------")
            else:
                content = response.text
                print(f"--- SUCCESS RESPONSE FOR '{prompt}' ---\n{content[:100]}...\n------------------")
        except Exception as e:
            print(f"Connection error for '{prompt}': {e}")

async def main():
    # 1. Send normal request
    t1 = asyncio.create_task(send_request("custom_runtime", "What is 2+2?"))
    await asyncio.sleep(1.0)
    
    # 2. Send forced error request
    t2 = asyncio.create_task(send_request("custom_runtime", "crash_test"))
    await asyncio.sleep(1.0)
    
    # 3. Send another normal request to prove server survived
    t3 = asyncio.create_task(send_request("custom_runtime", "What is 3+3?"))
    
    await asyncio.gather(t1, t2, t3)

if __name__ == "__main__":
    asyncio.run(main())
