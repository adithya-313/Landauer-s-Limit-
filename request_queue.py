import asyncio
import logging
import uuid
import json
import time
from datetime import datetime, timezone
import dataclasses

logger = logging.getLogger("RequestQueue")

@dataclasses.dataclass(order=True)
class QueueItem:
    """
    Represents a single request in the priority queue.
    Sorted primarily by priority (0 for premium, 1 for free),
    and secondarily by enqueue_time for strict FIFO within tiers.
    """
    priority: int      # 0 for premium, 1 for free
    enqueue_time: float
    request_id: str = dataclasses.field(compare=False)
    engine: str = dataclasses.field(compare=False)
    prompt: str = dataclasses.field(compare=False)
    tier: str = dataclasses.field(compare=False)
    response_queue: asyncio.Queue = dataclasses.field(compare=False)

class RequestQueue:
    """
    Tier-aware request queue that prioritises 'premium' over 'free' requests.
    Strict FIFO order within a tier based on arrival time.
    """
    def __init__(self, max_size: int = 50):
        # Set a sane default for max queue size to prevent memory explosion on a solo laptop
        self.max_size = max_size
        self.queue = asyncio.PriorityQueue()
        
    async def enqueue(self, engine: str, prompt: str, tier: str, response_queue: asyncio.Queue) -> str:
        """
        Add a request to the queue. Returns request_id.
        NOTE: request_id is strictly for log correlation, NOT for idempotency or deduplication.
        """
        # Reject immediately if full to gracefully degrade
        if self.queue.qsize() >= self.max_size:
            raise RuntimeError("Queue is full, cannot accept more requests.")
            
        request_id = str(uuid.uuid4())
        priority = 0 if tier == "premium" else 1
        
        item = QueueItem(
            priority=priority,
            enqueue_time=time.time(),
            request_id=request_id,
            engine=engine,
            prompt=prompt,
            tier=tier,
            response_queue=response_queue
        )
        
        await self.queue.put(item)
        return request_id
        
    async def _get_next(self) -> QueueItem:
        """
        Internal method for the background worker to fetch the next highest-priority request.
        Blocks until an item is available.
        """
        item = await self.queue.get()
        wait_seconds = round(time.time() - item.enqueue_time, 4)
        return item
