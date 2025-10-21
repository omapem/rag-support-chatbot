"""
Chat API endpoints with RAG integration.
Handles chat requests, conversation management, and response generation.
"""

from fastapi import APIRouter, HTTPException, status, Request
from typing import Optional
import logging
import uuid

from app.models.schemas import ChatRequest, ChatResponse
from app.services.conversation_memory import conversation_memory
from app.core.exceptions import VectorStoreException, LLMException
from app.middleware.rate_limiter import limiter
from src.generation import ResponseGenerator
from slowapi import Limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize RAG components (singleton)
_generator: Optional[ResponseGenerator] = None


def get_generator() -> ResponseGenerator:
    """
    Get or create ResponseGenerator instance.

    Returns:
        ResponseGenerator instance
    """
    global _generator
    if _generator is None:
        try:
            _generator = ResponseGenerator()
            logger.info("ResponseGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ResponseGenerator: {e}")
            raise VectorStoreException(
                message="Failed to initialize RAG system",
                details={"error": str(e)},
            )
    return _generator


@limiter.limit("20/minute")  # More restrictive limit for chat endpoint
@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: Request, chat_request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint with RAG-powered responses.

    Args:
        request: ChatRequest with query and optional session_id

    Returns:
        ChatResponse with answer, sources, and metadata

    Raises:
        HTTPException: On RAG pipeline failures
    """
    try:
        # Generate or validate session ID
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Add user message to conversation history
        conversation_memory.add_message(
            session_id=session_id,
            role="user",
            content=chat_request.query,
        )

        # Get conversation history for context
        history = conversation_memory.get_formatted_history(
            session_id=session_id,
            max_messages=10,  # Keep last 10 messages for context
        )

        # Generate response using RAG
        generator = get_generator()
        result = generator.generate_response(
            query=chat_request.query,
            conversation_history=history if history else None,
            top_k=chat_request.top_k,
        )

        # Add assistant response to conversation history
        conversation_memory.add_message(
            session_id=session_id,
            role="assistant",
            content=result["answer"],
        )

        # Build response
        response = ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            num_sources=result["num_sources"],
            has_context=result["has_context"],
            session_id=session_id,
        )

        # Add debug info if requested
        if chat_request.debug:
            response.debug_info = {
                "retrieval_method": "hybrid" if hasattr(generator.retriever, "enable_hybrid") else "semantic",
                "top_k": chat_request.top_k or 4,
                "conversation_history_length": len(conversation_memory.get_history(session_id)),
            }

        logger.info(
            f"Chat request processed successfully",
            extra={
                "session_id": session_id,
                "query_length": len(chat_request.query),
                "num_sources": result["num_sources"],
                "has_context": result["has_context"],
            },
        )

        return response

    except VectorStoreException:
        raise
    except LLMException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request",
        )


@limiter.limit("30/minute")
@router.get("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def get_session_info(request: Request, session_id: str):
    """
    Get information about a conversation session.

    Args:
        session_id: Session identifier

    Returns:
        Session metadata and statistics

    Raises:
        HTTPException: If session not found
    """
    info = conversation_memory.get_session_info(session_id)

    if info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )

    return info


@limiter.limit("10/minute")
@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def clear_session(request: Request, session_id: str):
    """
    Clear a conversation session.

    Args:
        session_id: Session identifier

    Returns:
        No content on success
    """
    conversation_memory.clear_session(session_id)
    logger.info(f"Session {session_id} cleared")


@limiter.limit("10/minute")
@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(request: Request):
    """
    Create a new conversation session.

    Returns:
        New session information
    """
    session_id = conversation_memory.create_session()
    info = conversation_memory.get_session_info(session_id)

    logger.info(f"New session created: {session_id}")

    return info
