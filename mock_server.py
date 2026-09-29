from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

active_requests = 0
max_active_requests = 0

async def mock_stream():
    global active_requests, max_active_requests
    active_requests += 1
    if active_requests > max_active_requests:
        max_active_requests = active_requests
    print(f"Active requests: {active_requests}")
    
    await asyncio.sleep(0.5)
    
    yield f"data: {json.dumps({'choices': [{'delta': {'content': 'mock'}}]})}\n\n"
    
    active_requests -= 1

@app.post("/v1/chat/completions")
async def mock_endpoint(req: dict):
    return StreamingResponse(mock_stream(), media_type="text/event-stream")

@app.get("/stats")
def get_stats():
    return {"max_active": max_active_requests}
