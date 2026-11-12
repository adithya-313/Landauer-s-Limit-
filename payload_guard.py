from __future__ import annotations

from typing import Any, Callable

from fastapi import Request
from fastapi.responses import JSONResponse

# MAX_PAYLOAD_SIZE_BYTES defines the upper limit for any request body (64 KB).
MAX_PAYLOAD_SIZE_BYTES: int = 65536


async def check_payload_size(request: Request, call_next: Callable[[Request], Any]) -> Any:
    """
    ASGI middleware that enforces a maximum request body size.

    Execution order:
    1. Only enforce Content-Length rules on methods that carry a body (skip GET, DELETE, HEAD, OPTIONS).
    2. If Content-Length header is missing, return 411 Length Required immediately.
    3. If malformed, return 400 Bad Request.
    4. If value exceeds MAX_PAYLOAD_SIZE_BYTES, return 413 Payload Too Large.
    5. Otherwise, pass the request to call_next.
    """
    if request.method in ("GET", "DELETE", "HEAD", "OPTIONS"):
        return await call_next(request)

    content_length_header = request.headers.get("content-length")

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

    return await call_next(request)
