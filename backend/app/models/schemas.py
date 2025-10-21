"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Represents a single message in a conversation."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate that role is either 'user' or 'assistant'."""
        if v not in ["user", "assistant"]:
            raise ValueError("Role must be either 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    query: str = Field(..., min_length=1, max_length=2000, description="User's question")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Number of documents to retrieve")
    debug: Optional[bool] = Field(default=False, description="Enable debug mode for retrieval details")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class Source(BaseModel):
    """Source document information."""

    name: str = Field(..., description="Document name")
    page: Optional[int] = Field(default=None, description="Page number")
    relevance_score: Optional[float] = Field(default=None, description="Relevance score")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    answer: str = Field(..., description="Generated answer")
    sources: List[str] = Field(default_factory=list, description="List of source document names")
    num_sources: int = Field(..., description="Number of sources used")
    has_context: bool = Field(..., description="Whether relevant context was found")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation tracking")
    debug_info: Optional[Dict[str, Any]] = Field(default=None, description="Debug information if enabled")


class ConversationHistory(BaseModel):
    """Conversation history model."""

    session_id: str = Field(..., description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="List of messages in conversation")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Session creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current timestamp")
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
