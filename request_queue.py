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
        
    def _log_event(self, event_type: str, request_id: str, tier: str, wait_seconds: float = None):
        """
        Appends a structured JSON event to queue_events.jsonl.
        Using plain file append for simplicity and to match the absence of complex
        centralised logging infra in the repo.
        """
        event = {
            "event": event_type,
            "request_id": request_id,
            "tier": tier,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if wait_seconds is not None:
            event["wait_seconds"] = wait_seconds
            
        try:
            with open("queue_events.jsonl", "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error("Failed to log queue event: %s", e)

    async def enqueue(self, engine: str, prompt: str, tier: str, response_queue: asyncio.Queue) -> str:
        """
        Add a request to the queue. Returns request_id.
        NOTE: request_id is strictly for log correlation, NOT for idempotency or deduplication.
        """
        # Reject immediately if full to gracefully degrade
        if self.queue.qsize() >= self.max_size:
            self._log_event("rejected", "N/A", tier)
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
        self._log_event("enqueued", request_id, tier)
        return request_id
        
    async def _get_next(self) -> QueueItem:
        """
        Internal method for the background worker to fetch the next highest-priority request.
        Blocks until an item is available.
        """
        item = await self.queue.get()
        wait_seconds = round(time.time() - item.enqueue_time, 4)
        self._log_event("dequeued", item.request_id, item.tier, wait_seconds)
        return item
