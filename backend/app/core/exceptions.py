"""
Custom exceptions and error handlers for the API.
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
from typing import Union
import logging

logger = logging.getLogger(__name__)


class APIException(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: dict = None,
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class SessionNotFoundException(APIException):
    """Raised when a session is not found."""

    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session {session_id} not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"session_id": session_id},
        )


class VectorStoreException(APIException):
    """Raised when vector store operations fail."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {},
        )


class LLMException(APIException):
    """Raised when LLM operations fail."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details or {},
        )


class RateLimitException(APIException):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """
    Handle custom API exceptions.

    Args:
        request: FastAPI request object
        exc: APIException instance

    Returns:
        JSON response with error details
    """
    logger.error(
        f"API Exception: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details,
            "path": request.url.path,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: RequestValidationError instance

    Returns:
        JSON response with validation error details
    """
    logger.warning(
        f"Validation error on {request.url.path}",
        extra={"errors": exc.errors()},
    )

    # Serialize validation errors properly (convert non-JSON types to strings)
    def serialize_error(error_dict):
        """Convert validation error dict to JSON-serializable format."""
        serialized = {}
        for key, value in error_dict.items():
            if key == "ctx" and isinstance(value, dict):
                # Convert context values to strings
                serialized[key] = {k: str(v) for k, v in value.items()}
            elif isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, (list, tuple)):
                serialized[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in value]
            else:
                serialized[key] = str(value)
        return serialized

    validation_errors = [serialize_error(err) for err in exc.errors()]

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": {"validation_errors": validation_errors},
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def generic_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Exception instance

    Returns:
        JSON response with generic error message
    """
    logger.error(
        f"Unexpected error: {str(exc)}",
        exc_info=True,
        extra={"path": request.url.path},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Please try again later.",
            "details": {},
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
