"""
gateway.py
==========
PHASE 2 — FastAPI Gateway with 64KB Memory Protection.

This script defines a FastAPI web server that acts as the API Gateway for the
Landauer's Limit project. It currently serves a mocked /v1/chat/completions
endpoint that follows the OpenAI API schema. The real AI engines will be
plugged in during Phase 5.

Security middleware intercepts every incoming request BEFORE the body is read.
If the Content-Length exceeds 64KB the request is rejected with HTTP 413,
preventing large payloads from consuming server memory.
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
# FastAPI is our web framework. Pydantic defines the data shapes.
# We use `requests`-style models for the OpenAI-compatible schema.
# JSONResponse lets us return custom payloads from the middleware.
# ---------------------------------------------------------------------------
import time
import uuid
from typing import List, Optional, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


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
# API ROUTES (Domain-Driven Design — Layer 3)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check():
    """
    Simple health-check endpoint. Useful for load balancers and monitoring.
    """
    return {"status": "ok", "timestamp": int(time.time())}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, tags=["chat"])
async def create_chat_completion(request_body: ChatCompletionRequest):
    """
    Dummy chat completion endpoint (to be connected to real AI in Phase 5).

    This endpoint currently returns a hard-coded mock response so the API
    contract can be tested end-to-end before the AI engines are integrated.
    """
    # Build a simple echo-like response so tests can verify the schema.
    last_user_message = ""
    for msg in reversed(request_body.messages):
        if msg.role == "user":
            last_user_message = msg.content
            break

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

    return response
