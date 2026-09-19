import time
from runtime_core.batch_engine import BatchEngine
import queue

engine = BatchEngine()
engine.start()

req_id, q = engine.submit("What is the capital of France?")
try:
    while True:
        msg = q.get(timeout=10.0)
        if msg["type"] == "done":
            break
except queue.Empty:
    pass
    
engine.stop()
