import logging
import asyncio
import queue
from typing import AsyncIterator, Dict, Any
from runtime_core.batch_engine import BatchEngine

logger = logging.getLogger("CustomRuntimeAdapter")

# Single shared instance across all requests for continuous batching
_shared_engine = None

def get_shared_engine() -> BatchEngine:
    """
    Gets or creates the shared BatchEngine instance on first use.
    """
    global _shared_engine
    if _shared_engine is None:
        logger.info("Initializing and starting shared BatchEngine...")
        _shared_engine = BatchEngine()
        _shared_engine.start()
    return _shared_engine

class CustomRuntimeAdapter:
    """
    Adapter for the Phase 6 Custom Runtime.
    Wires incoming requests to the shared continuous batching engine.
    """

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        engine = get_shared_engine()
        req_id, resp_q = engine.submit(prompt)
        
        while True:
            try:
                # Use asyncio.to_thread to safely poll the queue.Queue 
                # from the asyncio event loop. The queue is populated
                # by the background BatchEngine thread.
                msg = await asyncio.to_thread(resp_q.get, True, 0.5)
                if msg["type"] == "done":
                    break
                elif msg["type"] == "token":
                    yield msg["content"]
                elif msg["type"] == "error":
                    raise RuntimeError(msg["error"])
            except queue.Empty:
                # Timeout on get, loop back and try again to stay responsive to asyncio
                continue
            except Exception as e:
                logger.error("Error fetching from batch engine queue: %s", e)
                raise RuntimeError(str(e))

    async def health_check(self) -> Dict[str, Any]:
        engine = get_shared_engine()
        stats = engine.get_stats()
        return {
            "status": "ok" if stats["status"] == "running" else "degraded",
            "engine": "custom_runtime",
            "note": "Connected to shared BatchEngine",
            "stats": stats
        }
