"""
Main FastAPI application with RAG chatbot backend.
Includes CORS, rate limiting, error handling, and API routes.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded
import logging
import sys
from contextlib import asynccontextmanager

from app.api.routes import chat_router, health_router
from app.middleware.rate_limiter import limiter, custom_rate_limit_handler
from app.core.exceptions import (
    APIException,
    api_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.services.conversation_memory import conversation_memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.

    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting RAG Support Chatbot API...")

    # Initialize RAG components
    try:
        from app.api.routes.chat import get_generator
        get_generator()
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Don't fail startup - allow health checks to report unhealthy state

    # Cleanup expired sessions periodically could be done here
    # For now, it's handled on-demand

    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down RAG Support Chatbot API...")
    # Cleanup resources if needed
    logger.info("API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="RAG Support Chatbot API",
    description="Apache Kafka support chatbot powered by RAG and Claude AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add rate limiter state
app.state.limiter = limiter

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js default development server
        "http://localhost:8501",  # Streamlit default port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register exception handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include routers
app.include_router(health_router)
app.include_router(chat_router)


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        API welcome message and documentation links
    """
    return {
        "message": "RAG Support Chatbot API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health_check": "/health",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "detailed_health": "/health/detailed",
        },
    }


@app.get("/stats", tags=["monitoring"])
@limiter.limit("10/minute")
async def get_stats(request: Request):
    """
    Get API statistics.

    Returns:
        Current API statistics including active sessions
    """
    return {
        "active_sessions": conversation_memory.get_session_count(),
        "api_version": "1.0.0",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
