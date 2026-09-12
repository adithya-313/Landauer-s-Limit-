import asyncio
import os
import json
from request_queue import RequestQueue

async def main():
    print("=" * 60)
    print("PHASE 6a — QUEUE TEST")
    print("=" * 60)
    
    # Remove existing log file to have a clean test output
    if os.path.exists("queue_events.jsonl"):
        os.remove("queue_events.jsonl")
        
    # Initialize the queue
    queue = RequestQueue(max_size=50)
    
    # Create dummy response queues for the bridge
    q1, q2, q3, q4 = [asyncio.Queue() for _ in range(4)]

    print("Enqueueing 2 free and 2 premium requests concurrently...")
    
    # Enqueue them concurrently
    await asyncio.gather(
        queue.enqueue("mock_engine", "free prompt 1", "free", q1),
        queue.enqueue("mock_engine", "free prompt 2", "free", q2),
        queue.enqueue("mock_engine", "premium prompt 1", "premium", q3),
        queue.enqueue("mock_engine", "premium prompt 2", "premium", q4)
    )
    
    print("Dequeuing requests (expecting premium to come out first)...")
    dequeued_tiers = []
    
    # Poll the worker method 4 times
    for _ in range(4):
        item = await queue._get_next()
        dequeued_tiers.append(item.tier)
        print(f"Dequeued request ID: {item.request_id} | Tier: {item.tier} | Prompt: '{item.prompt}'")
        
    print(f"\nDequeue order: {dequeued_tiers}")
    if dequeued_tiers[:2] == ["premium", "premium"] and dequeued_tiers[2:] == ["free", "free"]:
        print("[PASS] Premium requests were prioritized correctly.")
    else:
        print("[FAIL] Unexpected dequeue order.")

    print("\nContents of queue_events.jsonl:")
    if os.path.exists("queue_events.jsonl"):
        with open("queue_events.jsonl", "r") as f:
            for line in f:
                print(line.strip())
    else:
        print("No log file found!")

if __name__ == "__main__":
    asyncio.run(main())
