"""
Health check endpoints for monitoring and status.
"""

from fastapi import APIRouter, status, Request
from datetime import datetime
import logging

from app.models.schemas import HealthResponse
from app.middleware.rate_limiter import limiter
from app.services.conversation_memory import conversation_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])

API_VERSION = "1.0.0"


@limiter.limit("60/minute")  # Allow frequent health checks
@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check(request: Request) -> HealthResponse:
    """
    Basic health check endpoint.

    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        components={},
    )


@limiter.limit("30/minute")
@router.get("/detailed", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def detailed_health_check(request: Request) -> HealthResponse:
    """
    Detailed health check with component status.

    Returns:
        HealthResponse with detailed component information

    Note:
        In production, this should check:
        - Vector database connectivity
        - LLM API availability
        - Memory usage
        - Active sessions
    """
    components = {}

    # Check conversation memory
    try:
        session_count = conversation_memory.get_session_count()
        components["conversation_memory"] = f"healthy ({session_count} active sessions)"
    except Exception as e:
        logger.error(f"Conversation memory check failed: {e}")
        components["conversation_memory"] = "unhealthy"

    # Check vector store (basic check - generator initialization)
    try:
        from app.api.routes.chat import get_generator
        generator = get_generator()
        components["vector_store"] = "healthy"
        components["rag_pipeline"] = "healthy"
    except Exception as e:
        logger.error(f"RAG pipeline check failed: {e}")
        components["vector_store"] = "unhealthy"
        components["rag_pipeline"] = "unhealthy"

    # Determine overall status
    overall_status = "healthy" if all(
        "healthy" in status for status in components.values()
    ) else "degraded"

    return HealthResponse(
        status=overall_status,
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        components=components,
    )


@limiter.limit("60/minute")
@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check(request: Request):
    """
    Kubernetes-style readiness probe.

    Returns:
        Simple OK response if service is ready to accept traffic

    Raises:
        HTTPException: If service is not ready
    """
    try:
        # Basic readiness checks
        from app.api.routes.chat import get_generator
        get_generator()  # Verify RAG pipeline is initialized
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )


@limiter.limit("60/minute")
@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check(request: Request):
    """
    Kubernetes-style liveness probe.

    Returns:
        Simple OK response if service is alive
    """
    return {"status": "alive"}
