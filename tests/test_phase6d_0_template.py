import asyncio
import httpx
import sys

async def send_req(client, messages, engine, tier="free", max_tokens=10):
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": messages,
        "engine": engine,
        "tier": tier,
        "max_tokens": max_tokens
    }
    print(f"[{engine}] Sending request...")
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if "error" in line.lower():
                    print(f"[{engine}] ERROR in stream: {line}")
        print(f"[{engine}] Finished request.")
    except Exception as e:
        print(f"[{engine}] Failed: {e}")

async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8004", timeout=300.0) as client:
        print("\n--- Test 1: Explicit System Message (USE_CHAT_TEMPLATE=True) ---")
        msgs_1 = [
            {"role": "system", "content": "You are a Pirate."},
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Ahoy!"},
            {"role": "user", "content": "What is 2+2?"}
        ]
        await send_req(client, msgs_1, "custom_runtime")
        await asyncio.sleep(2)
        
        print("\n--- Test 2: Implicit System Message via Env Var ---")
        msgs_2 = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        await send_req(client, msgs_2, "custom_runtime")
        await asyncio.sleep(2)

        print("\n--- Test 3: Raw Prompt Fallback (requires manual change of USE_CHAT_TEMPLATE=False in batch_engine.py) ---")
        msgs_3 = [
            {"role": "user", "content": "I should be raw text."}
        ]
        await send_req(client, msgs_3, "custom_runtime")
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
