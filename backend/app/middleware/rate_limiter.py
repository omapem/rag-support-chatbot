"""
Rate limiting middleware using slowapi.
Prevents API abuse and manages request quotas.
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
import os


def get_rate_limit_key(request: Request) -> str:
    """
    Get rate limit key for the request.
    Uses IP address for anonymous requests or user ID for authenticated requests.

    Args:
        request: FastAPI request object

    Returns:
        Rate limit key string
    """
    # For future authentication: check for user ID in request state
    # For now, use IP address
    return get_remote_address(request)


# Initialize rate limiter
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[
        # Global limits
        "100/minute",  # Maximum 100 requests per minute per IP
        "1000/hour",   # Maximum 1000 requests per hour per IP
    ],
    storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
    strategy="fixed-window",  # or "moving-window" for more accurate limiting
)


# Custom rate limit exceeded handler
def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors.

    Args:
        request: FastAPI request object
        exc: RateLimitExceeded exception

    Returns:
        JSON response with rate limit error
    """
    from datetime import datetime
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "message": "Too many requests. Please try again later.",
            "details": {
                "retry_after": getattr(exc, "retry_after", None),
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
        headers={
            "Retry-After": str(getattr(exc, "retry_after", 60)),
        },
    )
