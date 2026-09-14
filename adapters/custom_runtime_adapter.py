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
    Retrieves the one central brain (BatchEngine) that manages all conversations.
    If it hasn't been turned on yet, it starts it up.
    
    Returns:
    - The active BatchEngine instance, so we can send it new questions to answer.
    """
    global _shared_engine
    if _shared_engine is None:
        logger.info("Initializing and starting shared BatchEngine...")
        _shared_engine = BatchEngine()
        _shared_engine.start()
    return _shared_engine

class CustomRuntimeAdapter:
    """
    This is the bridge between the web server and the continuous batching engine.
    It takes an incoming web request, hands the question to the engine, and then
    waits to catch the answer words one-by-one so it can send them back to the user.
    """

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        """
        Takes a single user's question, hands it to the background AI engine,
        and returns the answer piece by piece as a stream.
        
        Inputs:
        - prompt: The text of the user's question.
        
        Returns:
        - A stream of text (words) that can be sent directly over the web to the user's browser.
        """
        engine = get_shared_engine()
        
        # We hand the question to the engine and get back a personal "mailbox" (resp_q).
        # The engine will drop the generated words into this mailbox as it thinks of them.
        req_id, resp_q = engine.submit(prompt)
        
        while True:
            try:
                # The engine is running in a different background thread. 
                # If we just tried to open the mailbox normally, our web server would 
                # freeze completely while waiting for the next word. 
                # Instead, we use a special tool (asyncio.to_thread) to have an assistant 
                # check the mailbox for us. If there's no word after 0.5 seconds, the 
                # assistant comes back empty-handed so our web server doesn't freeze.
                msg = await asyncio.to_thread(resp_q.get, True, 0.5)
                
                if msg["type"] == "done":
                    break # The engine finished answering this question.
                elif msg["type"] == "token":
                    yield msg["content"] # Hand the newly generated word back to the web server.
                elif msg["type"] == "error":
                    raise RuntimeError(msg["error"]) # Something went wrong inside the engine.
                    
            except queue.Empty:
                # The assistant checked the mailbox but the engine hasn't generated the next word yet.
                # We just loop back and ask the assistant to check again.
                continue
            except Exception as e:
                logger.error("Error fetching from batch engine queue: %s", e)
                raise RuntimeError(str(e))

    async def health_check(self) -> Dict[str, Any]:
        """
        Checks if the AI engine is awake and ready to answer questions.
        
        Returns:
        - A small dictionary of statistics (e.g., how many questions it's currently answering).
        """
        engine = get_shared_engine()
        stats = engine.get_stats()
        return {
            "status": "ok" if stats["status"] == "running" else "degraded",
            "engine": "custom_runtime",
            "note": "Connected to shared BatchEngine",
            "stats": stats
        }
