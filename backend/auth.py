"""
Authentication middleware for the Deepfake Detection System.

Provides a simple API‑key authentication scheme.  When the
``API_KEY`` environment variable is set, every request must include
a matching ``X-API-Key`` header.  When the variable is unset,
authentication is disabled (open access — suitable for development).
"""

import os
from typing import Optional

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# Public paths that never require authentication.
PUBLIC_PATHS: set[str] = {"/health", "/docs", "/openapi.json", "/redoc"}


def get_api_key() -> Optional[str]:
    """Return the configured API key or *None* when auth is disabled."""
    return os.environ.get("API_KEY")


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that checks for a valid ``X-API-Key`` header.

    If ``API_KEY`` is not set in the environment, all requests are
    allowed through without authentication.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Validate the API key on every non‑public request."""
        api_key = get_api_key()

        # Auth disabled — pass through
        if api_key is None:
            return await call_next(request)

        # Allow public endpoints
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        # Allow OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Check header
        request_key = request.headers.get("X-API-Key")
        if request_key != api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide a valid X-API-Key header.",
            )

        return await call_next(request)
