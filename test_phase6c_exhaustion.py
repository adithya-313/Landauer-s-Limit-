import asyncio
import queue
import time
import threading
import torch

# We import from runtime_core to start the engine directly for this test
from runtime_core.batch_engine import BatchEngine
from runtime_core.kv_cache_manager import KVCacheManager

def patch_kv_cache_manager():
    # Patch KVCacheManager to have a very small pool to force eviction
    original_init = KVCacheManager.__init__
    def new_init(self, device, max_blocks=20, dtype=torch.bfloat16): # Very small! 20 blocks = 320 tokens
        original_init(self, device, max_blocks=max_blocks, dtype=dtype)
    KVCacheManager.__init__ = new_init

async def main():
    patch_kv_cache_manager()
    engine = BatchEngine()
    engine.start()
    
    # Send a bunch of requests all at once
    prompts = [
        "Write a very long and detailed essay about the complete history of the Roman Empire.",
        "Write a very long and detailed essay about the complete history of the Roman Empire.",
        "Write a very long and detailed essay about the complete history of the Roman Empire.",
        "Write a very long and detailed essay about the complete history of the Roman Empire.",
        "Write a very long and detailed essay about the complete history of the Roman Empire."
    ]
    
    qs = []
    # Submit first 3 as premium, next 2 as free
    for i, p in enumerate(prompts):
        tier = "premium" if i < 3 else "free"
        req_id, q = engine.submit(p, tier=tier)
        qs.append((req_id, q))
        
    print("Submitted 5 long requests to engine with only 20 blocks available.")
    print("Waiting for eviction to occur...")
    
    # Read queues
    start = time.time()
    evictions = 0
    while time.time() - start < 60:
        all_done = True
        for req_id, q in qs:
            try:
                # read all available messages in queue
                while True:
                    msg = q.get_nowait()
                    if msg["type"] == "error" and "OOM" in msg["content"]:
                        print(f"[{time.time():.2f}] SUCCESS: Request {req_id} was evicted with OOM error.")
                        evictions += 1
                    elif msg["type"] == "done":
                        print(f"[{time.time():.2f}] Request {req_id} finished naturally.")
            except queue.Empty:
                pass
                
        if evictions >= 2: # At least a couple evictions expected
            print("Successfully verified eviction logic.")
            break
        await asyncio.sleep(0.5)
        
    engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
