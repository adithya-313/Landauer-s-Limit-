"""
gateway.py
==========
PHASE 2/3/4/5 — FastAPI Gateway with 64KB Memory Protection + CPU Guardrail
                 + Semantic Cache + Multi-Engine Router.

This script defines a FastAPI web server that acts as the API Gateway for the
Landauer's Limit project. It serves an OpenAI-compatible /v1/chat/completions
endpoint.

Execution layers (in order):
  1. Security middleware — reject payloads > 64 KB before reading a byte.
  2. Guardrail (Phase 3) — ONNX + FAISS prompt-injection detector with
     fail-open degradation on timeout/error.
  3. Semantic Cache (Phase 4) — FAISS dual-lock cache (entity + vector).
  4. Multi-Engine Router (Phase 5) — dispatches to vLLM, llama.cpp, Colab
     T4, or the custom runtime placeholder.  Uses StreamingResponse for
     token-by-token SSE output.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# FastAPI is our web framework. Pydantic defines the data shapes.
# StreamingResponse sends OpenAI-compatible SSE tokens to the client.
# JSONResponse lets us return custom payloads from the middleware.
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import time
import os
from typing import AsyncIterator, List, Optional
import time
import uuid
from typing import List, Optional, Literal, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# The CPU Guardrail (Phase 3) — ONNX + FAISS prompt-injection detector.
import guardrail as _guardrail_module
from guardrail import guardrail as _guardrail_instance

# The Semantic Cache (Phase 4) — FAISS IndexIDMap + LRU eviction cache.
import semantic_cache as _cache_module
from semantic_cache import semantic_cache as _cache_instance

# The Multi-Engine Router (Phase 5) — dispatches to vLLM, llama.cpp, Colab, etc.
from adapters.router import EngineRouter
from request_queue import RequestQueue

# Create the global router instance that all requests share.
_router = EngineRouter()

# Create the global request queue (Phase 6a)
_request_queue = RequestQueue(max_size=50)

# Set up a logger so we can log guardrail degradation events.
logger = logging.getLogger("gateway")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
# MAX_PAYLOAD_SIZE_BYTES defines the upper limit for any request body.
# If the client sends more than 64 KB we reject it before reading a byte.
# ---------------------------------------------------------------------------
MAX_PAYLOAD_SIZE_BYTES: int = 65536


# ---------------------------------------------------------------------------
# DATA MODELS (Domain-Driven Design — Layer 1)
# ---------------------------------------------------------------------------
# These Pydantic classes define the exact shape of the OpenAI-compatible
# chat completion request and response. Every field is typed so FastAPI
# can validate incoming data automatically and return clear error messages.
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """
    A single turn in a conversation.

    Attributes
    ----------
    role : str
        Either "user", "assistant", or "system".
    content : str
        The text of this turn.
    """
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    """
    The expected shape of a POST /v1/chat/completions request body.

    This matches the OpenAI API specification so any OpenAI-compatible
    client library can talk to our gateway without modification.

    Attributes
    ----------
    model : str
        The model identifier (e.g. "gpt-4", "llama-3").
    messages : list[Message]
        The conversation history plus the latest user prompt.
    temperature : float | None
        Sampling temperature (0-2). Optional, defaults to 1.0.
    max_tokens : int | None
        Maximum tokens to generate. Optional.
    """
    model: str
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    engine: Optional[str] = Field(
        default="vllm_local",
        description=(
            "Target engine: vllm_local, llamacpp_local, colab_cloud, "
            "or custom_runtime.  Defaults to vllm_local."
        ),
    )
    tier: Literal["free", "premium"] = Field(
        default="free",
        description="Priority tier for request execution."
    )


class Choice(BaseModel):
    """
    A single completion choice returned by the model.

    Attributes
    ----------
    index : int
        The position of this choice in the list.
    message : Message
        The assistant's response message.
    finish_reason : str
        Why the model stopped generating ("stop", "length", etc.).
    """
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """
    The full response returned by POST /v1/chat/completions.

    Follows the OpenAI schema so downstream consumers (like chatbots,
    evaluation scripts, etc.) can parse it without adaptation.
    """
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[dict] = None


# ---------------------------------------------------------------------------
# SECURITY MIDDLEWARE (Domain-Driven Design — Layer 2)
# ---------------------------------------------------------------------------
# This middleware runs on EVERY incoming request before FastAPI processes
# the route. It checks the Content-Length header and rejects oversized
# payloads WITHOUT reading the body, keeping memory usage predictable.
# ---------------------------------------------------------------------------

async def check_payload_size(request: Request, call_next):
    """
    ASGI middleware that enforces a maximum request body size.

    How it works:
    1. Read the Content-Length header from the request.
    2. If the header is missing, return 411 Length Required immediately.
    3. If the value exceeds MAX_PAYLOAD_SIZE_BYTES, return 413.
    4. Otherwise, pass the request through to the normal route handler.

    Parameters
    ----------
    request : Request
        The incoming HTTP request.
    call_next : callable
        The next middleware or route handler in the chain.

    Returns
    -------
    Response
        Either an error JSONResponse or the normal handler response.
    """
    # Only enforce Content-Length rules on methods that carry a body.
    # GET, DELETE, HEAD, OPTIONS typically have no body so we skip the check.
    if request.method in ("GET", "DELETE", "HEAD", "OPTIONS"):
        return await call_next(request)

    # Try to read the Content-Length header.
    content_length_header = request.headers.get("content-length")

    # If no Content-Length was sent, we cannot safely size the payload.
    # Reject with HTTP 411 Length Required.
    if content_length_header is None:
        return JSONResponse(
            status_code=411,
            content={
                "error": {
                    "message": (
                        "Content-Length header is required. "
                        "Please include it with your request."
                    ),
                    "type": "length_required",
                    "code": 411,
                }
            },
        )

    # Parse the header value to an integer. If it's malformed, reject.
    try:
        content_length = int(content_length_header)
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Content-Length header is not a valid integer.",
                    "type": "bad_request",
                    "code": 400,
                }
            },
        )

    # If the payload exceeds our limit, return 413 BEFORE reading the body.
    if content_length > MAX_PAYLOAD_SIZE_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": (
                        f"Payload too large. Maximum allowed size is "
                        f"{MAX_PAYLOAD_SIZE_BYTES} bytes, but received "
                        f"{content_length} bytes."
                    ),
                    "type": "payload_too_large",
                    "code": 413,
                }
            },
        )

    # Payload is within limits. Hand off to the route handler.
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# APPLICATION SETUP
# ---------------------------------------------------------------------------
# Create the FastAPI app instance and register the security middleware
# so that every request passes through it.
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Landauer's Limit — API Gateway",
    description=(
        "OpenAI-compatible chat completion gateway with "
        "built-in 64KB payload size protection."
    ),
    version="0.1.0",
)

# Register the payload-size middleware.
app.middleware("http")(check_payload_size)


# ---------------------------------------------------------------------------
# LIFESPAN EVENT — Startup / Shutdown
# ---------------------------------------------------------------------------
# We use the modern lifespan context manager (replaces deprecated
# on_event("startup")). The guardrail is already initialised at import time;
# here we just log confirmation.
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    """
    Confirm the guardrail is ready when the server starts.
    The guardrail initialises itself at module import time, so this is
    mostly a logging convenience.
    """
    logger.info(
        "Guardrail loaded. Timeout = %.0f ms, Threshold = %.2f",
        _guardrail_module.GUARDRAIL_TIMEOUT * 1000,
        _guardrail_module.SIMILARITY_THRESHOLD,
    )
    
    # Start the single background queue worker
    asyncio.create_task(queue_worker())

async def _handle_request(item):
    """
    Takes exactly one user's question from the waiting line and manages its entire lifecycle.
    
    Inputs:
    - item: A bundle of information about the user's request (their question, which model they want,
            and the specific mailbox where we should drop the answers).
            
    Returns:
    - Nothing directly. Instead, it drops the answer words into the user's mailbox as they are generated.
    """
    try:
        # Ask the router to find the right AI engine to answer this specific question.
        # As the engine generates words one-by-one, we loop over them here.
        async for token in _router.route_request(item.engine, item.prompt, item.tier, item.max_tokens, getattr(item, "messages", None)):
            # Drop the new word into the user's personal mailbox so the web server can send it to them.
            await item.response_queue.put({"type": "token", "content": token})
            
        # The engine finished the whole answer, so we drop a special "done" message into the mailbox.
        await item.response_queue.put({"type": "done"})
        
    except Exception as e:
        # If the engine crashes while answering, we don't want the web server to wait forever.
        # We drop a special error message into the mailbox so the user knows something broke.
        logger.error("Error processing request %s: %s", getattr(item, "request_id", "unknown"), e)
        _request_queue._log_event("error", getattr(item, "request_id", "unknown"), getattr(item, "tier", "unknown"))
        if 'item' in locals() and hasattr(item, 'response_queue'):
            await item.response_queue.put({"type": "error", "error": str(e)})

async def queue_worker():
    """
    The main background worker that acts like a bouncer at a club.
    It runs endlessly, pulling the next person out of the waiting line and 
    assigning an assistant (_handle_request) to help them immediately, 
    before turning right back to pull the next person.
    """
    while True:
        try:
            # Pull the next person's question out of the waiting line. 
            # If the line is empty, it just waits here patiently until someone arrives.
            item = await _request_queue._get_next()
            
            # Instead of helping this person from start to finish itself (which would block the line),
            # it spawns an independent "assistant" task to handle this specific person concurrently.
            # This allows the bouncer to immediately turn back and pull the next person from the line.
            asyncio.create_task(_handle_request(item))
            
        except Exception as e:
            logger.error("Queue worker dispatch error: %s", e)


# ---------------------------------------------------------------------------
# API ROUTES (Domain-Driven Design — Layer 3)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check():
    """
    Simple health-check endpoint. Useful for load balancers and monitoring.
    """
    return {"status": "ok", "timestamp": int(time.time())}


@app.post("/v1/chat/completions", tags=["chat"])
async def create_chat_completion(
    request_body: ChatCompletionRequest,
    request: Request,
):
    """
    Chat completion endpoint with guardrail + semantic cache + multi-engine router.

    Execution order:
      1. Parse request body.
      2. guardrail.check(prompt) with timeout.
      3. If malicious -> HTTP 400 (Hard Halt).
      4. If TimeoutError/Exception -> degraded = True, proceed (fail-open).
      5. cache.check_cache(messages).
      6. Cache Hit -> HTTP 200 + X-Cache: HIT [+ X-Guardrail-Degraded].
      7. Cache Miss -> route to selected engine via StreamingResponse.
         Tokens are streamed as SSE.  After stream completes, the full
         response is inserted into the cache.
         Headers: X-Cache: MISS [+ X-Guardrail-Degraded].
    """
    # --- Step 1: Parse / extract the last user message ---
    messages_dicts = [
        {"role": msg.role, "content": msg.content}
        for msg in request_body.messages
    ]

    last_user_message = ""
    has_system = False
    for msg in reversed(messages_dicts):
        if msg["role"] == "user" and not last_user_message:
            last_user_message = msg["content"]
        if msg["role"] == "system":
            has_system = True

    if not has_system:
        system_prompt = os.environ.get("SYSTEM_PROMPT", "")
        if system_prompt:
            messages_dicts.insert(0, {"role": "system", "content": system_prompt})

    if not last_user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found in the request."},
        )

    # --- Step 2: Guardrail with fail-open ---
    degraded = False
    try:
        is_malicious, score = await asyncio.wait_for(
            asyncio.to_thread(_guardrail_instance.check, last_user_message),
            timeout=_guardrail_module.GUARDRAIL_TIMEOUT,
        )

        # Step 3: Malicious -> Hard Halt.
        if is_malicious:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Prompt Injection Detected",
                    "similarity_score": round(score, 4),
                },
            )

    except asyncio.TimeoutError:
        # Step 4: Fail-open — guardrail timed out, proceed degraded.
        logger.warning(
            "Guardrail timeout (%.0f ms) — degrading for prompt: %.50s",
            _guardrail_module.GUARDRAIL_TIMEOUT * 1000,
            last_user_message,
        )
        degraded = True
    except Exception:
        # Fail-open — any unexpected guardrail error, proceed degraded.
        logger.exception("Guardrail error — degrading.")
        degraded = True

    # --- Step 5: Check the semantic cache ---
    cached_response = await _cache_instance.check_cache(messages_dicts)

    # --- Step 6: Cache Hit — return immediately ---
    if cached_response is not None:
        headers = {"X-Cache": "HIT"}
        if degraded:
            headers["X-Guardrail-Degraded"] = "true"
        logger.info("Cache HIT for: %.50s", last_user_message)
        return JSONResponse(
            status_code=200,
            content=cached_response,
            headers=headers,
        )

    # --- Step 7: Cache Miss — route to selected engine, stream, cache ---
    # We build an async generator that:
    #   1. Streams SSE tokens to the client via StreamingResponse.
    #   2. Accumulates the full response text.
    #   3. After streaming completes, inserts the full response into the cache.
    # This keeps the cache populated for future hits without buffering the
    # entire response before sending the first token.

    # Build a per-request asyncio.Queue to receive tokens from the background worker.
    # This bridges the gap between the single background worker and this specific HTTP client.
    response_queue = asyncio.Queue()
    
    try:
        req_id = await _request_queue.enqueue(
            engine=request_body.engine,
            prompt=last_user_message,
            tier=request_body.tier,
            max_tokens=request_body.max_tokens,
            response_queue=response_queue,
            messages=messages_dicts
        )
    except RuntimeError as e:
        # The queue is full, gracefully reject the request immediately rather than blocking
        return JSONResponse(status_code=429, content={"error": str(e)})

    async def streaming_generator() -> AsyncIterator[str]:
        """
        Async generator that yields SSE-encoded tokens from the engine router
        and caches the completed response after the stream ends.
        """
        full_content = ""
        stream_failed = False

        try:
            while True:
                msg = await response_queue.get()
                
                if msg["type"] == "done":
                    break
                elif msg["type"] == "error":
                    # Engine failure — stream an error token so the client sees it.
                    stream_failed = True
                    error_text = f" [Engine error: {msg['error']}] "
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': error_text}, 'index': 0}]})}\n\n"
                    logger.error("Stream failed for engine '%s': %s", request_body.engine, msg["error"])
                    break
                else:
                    token = msg["content"]
                    full_content += token
                    # Yield an OpenAI-compatible SSE delta chunk.
                    chunk = {
                        "choices": [
                            {
                                "delta": {"content": token},
                                "index": 0,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

        finally:
            # Signal the end of the SSE stream.
            yield "data: [DONE]\n\n"

            # Cache the full response only if streaming succeeded.
            if not stream_failed and full_content:
                response = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    object="chat.completion",
                    created=int(time.time()),
                    model=request_body.model,
                    choices=[
                        Choice(
                            index=0,
                            message=Message(role="assistant", content=full_content),
                            finish_reason="stop",
                        )
                    ],
                    usage={
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                )
                await _cache_instance.insert(messages_dicts, response.model_dump())
                logger.info(
                    "Cache inserted for engine='%s' (%d chars).",
                    request_body.engine, len(full_content),
                )

    # Build the response headers.
    stream_headers = {"X-Cache": "MISS"}
    if degraded:
        stream_headers["X-Guardrail-Degraded"] = "true"

    return StreamingResponse(
        streaming_generator(),
        media_type="text/event-stream",
        headers=stream_headers,
    )
