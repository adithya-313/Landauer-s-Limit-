"""
gateway.py
==========
PHASE 2/3/4 — FastAPI Gateway with 64KB Memory Protection + CPU Guardrail
              + Semantic Cache.

This script defines a FastAPI web server that acts as the API Gateway for the
Landauer's Limit project. It currently serves a mocked /v1/chat/completions
endpoint that follows the OpenAI API schema. The real AI engines will be
plugged in during Phase 5.

Security middleware intercepts every incoming request BEFORE the body is read.
If the Content-Length exceeds 64KB the request is rejected with HTTP 413,
preventing large payloads from consuming server memory.

The CPU Guardrail (Phase 3) inspects every user message for prompt-injection
patterns via ONNX Runtime + FAISS HNSW. It runs with a strict timeout; if the
guardrail times out, the request is allowed through (fail-open) and the
X-Guardrail-Degraded response header is set.

The Semantic Cache (Phase 4) checks whether a semantically similar question
has already been answered before calling the guardrail. On a cache hit the
cached response is returned immediately with X-Cache: HIT, bypassing both the
guardrail and (in the future) the LLM call.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# FastAPI is our web framework. Pydantic defines the data shapes.
# We use `requests`-style models for the OpenAI-compatible schema.
# JSONResponse lets us return custom payloads from the middleware.
# ---------------------------------------------------------------------------
import asyncio
import logging
import time
import uuid
from typing import List, Optional, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# The CPU Guardrail (Phase 3) — ONNX + FAISS prompt-injection detector.
import guardrail as _guardrail_module
from guardrail import guardrail as _guardrail_instance

# The Semantic Cache (Phase 4) — FAISS IndexIDMap + LRU eviction cache.
import semantic_cache as _cache_module
from semantic_cache import semantic_cache as _cache_instance

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
    Chat completion endpoint with semantic cache + prompt-injection guardrail.

    Request lifecycle:
    1. Extract the last user message and (for multi-turn) the last assistant
       response to build context.
    2. Check the semantic cache. If a semantically similar Q&A pair is found,
       return the cached response immediately with X-Cache: HIT.
    3. Otherwise, run the guardrail ONNX inference offloaded to a thread pool
       with a hard timeout. If the guardrail times out, the request proceeds
       with X-Guardrail-Degraded: true (fail-open).
    4. If the guardrail flags the message as malicious, return HTTP 400.
    5. Insert the response into the semantic cache and return it with
       X-Cache: MISS.
    """
    # --- Step 1: Extract messages for cache lookup and guardrail ---
    last_user_message = ""
    last_assistant_response = ""
    for msg in reversed(request_body.messages):
        if msg.role == "assistant" and not last_assistant_response:
            last_assistant_response = msg.content
        if msg.role == "user" and not last_user_message:
            last_user_message = msg.content

    # Bail out early if there is no user message at all.
    if not last_user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "No user message found in the request."},
        )

    # --- Step 2: Check the semantic cache ---
    # Use the last assistant response as context for multi-turn disambiguation.
    cache_context = last_assistant_response

    cache_hit, cached_response, cache_similarity = await asyncio.to_thread(
        _cache_instance.check_cache, last_user_message, cache_context,
    )

    if cache_hit:
        logger.info(
            "Cache HIT (sim=%.4f) for: %.50s",
            cache_similarity,
            last_user_message,
        )
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=request_body.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=cached_response),
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        )
        return JSONResponse(
            status_code=200,
            content=response.model_dump(),
            headers={"X-Cache": "HIT"},
        )

    # --- Step 3: Run the guardrail with a timeout ---
    degraded = False
    guardrail_passed = False
    try:
        # Offload to thread pool so the ONNX inference does not block.
        is_malicious, score = await asyncio.wait_for(
            asyncio.to_thread(_guardrail_instance.check, last_user_message),
            timeout=_guardrail_module.GUARDRAIL_TIMEOUT,
        )

        if is_malicious:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Prompt Injection Detected",
                    "similarity_score": round(score, 4),
                },
            )

        guardrail_passed = True

    except asyncio.TimeoutError:
        # Fail-open: guardrail took too long, let the request through.
        logger.warning(
            "Guardrail timeout (%.0f ms) — degrading for prompt: %.50s",
            _guardrail_module.GUARDRAIL_TIMEOUT * 1000,
            last_user_message,
        )
        degraded = True
    except Exception:
        # Also fail-open on any unexpected guardrail error.
        logger.exception("Guardrail error — degrading.")
        degraded = True

    # --- Step 4: Build the mock response ---
    mock_reply = (
        f"This is a mock response from the Landauer's Limit gateway. "
        f"You said: \"{last_user_message[:50]}{'...' if len(last_user_message) > 50 else ''}\""
    )

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        object="chat.completion",
        created=int(time.time()),
        model=request_body.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=mock_reply),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )

    # --- Step 5: Insert into cache if guardrail passed (not malicious) ---
    if guardrail_passed or degraded:
        # Offload the cache insert to a thread so it does not block.
        await asyncio.to_thread(
            _cache_instance.insert, last_user_message,
            response.choices[0].message.content, cache_context,
        )

    # --- Step 6: Attach headers ---
    response_headers = {"X-Cache": "MISS"}
    if degraded:
        response_headers["X-Guardrail-Degraded"] = "true"

    response_dict = response.model_dump()
    return JSONResponse(
        status_code=200,
        content=response_dict,
        headers=response_headers,
    )
