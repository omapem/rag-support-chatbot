"""
Unit tests for Pydantic models and schemas.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConversationHistory,
    HealthResponse,
    ErrorResponse,
    Source,
)


class TestChatMessage:
    """Tests for ChatMessage model."""

    def test_valid_user_message(self):
        """Test creating a valid user message."""
        message = ChatMessage(role="user", content="Test message")
        assert message.role == "user"
        assert message.content == "Test message"
        assert message.timestamp is None

    def test_valid_assistant_message(self):
        """Test creating a valid assistant message."""
        message = ChatMessage(role="assistant", content="Response message")
        assert message.role == "assistant"
        assert message.content == "Response message"

    def test_message_with_timestamp(self):
        """Test message with explicit timestamp."""
        now = datetime.utcnow()
        message = ChatMessage(role="user", content="Test", timestamp=now)
        assert message.timestamp == now

    def test_invalid_role(self):
        """Test that invalid role raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role="invalid", content="Test")
        assert "Role must be either 'user' or 'assistant'" in str(exc_info.value)

    def test_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError):
            ChatMessage(role="user")  # Missing content

        with pytest.raises(ValidationError):
            ChatMessage(content="Test")  # Missing role


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_basic_request(self):
        """Test creating a valid basic chat request."""
        request = ChatRequest(query="How do I create a Kafka topic?")
        assert request.query == "How do I create a Kafka topic?"
        assert request.session_id is None
        assert request.top_k is None
        assert request.debug is False

    def test_valid_request_with_all_fields(self):
        """Test request with all optional fields."""
        request = ChatRequest(
            query="Test query",
            session_id="test-session-123",
            top_k=5,
            debug=True,
        )
        assert request.query == "Test query"
        assert request.session_id == "test-session-123"
        assert request.top_k == 5
        assert request.debug is True

    def test_query_whitespace_stripping(self):
        """Test that query whitespace is stripped."""
        request = ChatRequest(query="  Test query  ")
        assert request.query == "Test query"

    def test_empty_query_validation(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="")
        # Pydantic v2 has different error messages
        assert "query" in str(exc_info.value).lower()

    def test_whitespace_only_query_validation(self):
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="   ")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_query_length_validation(self):
        """Test query length limits."""
        # Valid length
        valid_query = "x" * 2000
        request = ChatRequest(query=valid_query)
        assert len(request.query) == 2000

        # Too long
        with pytest.raises(ValidationError):
            ChatRequest(query="x" * 2001)

    def test_top_k_range_validation(self):
        """Test top_k range validation."""
        # Valid values
        for k in [1, 5, 10]:
            request = ChatRequest(query="test", top_k=k)
            assert request.top_k == k

        # Too small
        with pytest.raises(ValidationError):
            ChatRequest(query="test", top_k=0)

        # Too large
        with pytest.raises(ValidationError):
            ChatRequest(query="test", top_k=11)


class TestChatResponse:
    """Tests for ChatResponse model."""

    def test_valid_response(self):
        """Test creating a valid chat response."""
        response = ChatResponse(
            answer="Here's how to create a topic...",
            sources=["kafka_docs.pdf", "best_practices.pdf"],
            num_sources=2,
            has_context=True,
        )
        assert response.answer == "Here's how to create a topic..."
        assert len(response.sources) == 2
        assert response.num_sources == 2
        assert response.has_context is True
        assert response.session_id is None
        assert response.debug_info is None

    def test_response_with_session_and_debug(self):
        """Test response with session_id and debug info."""
        debug_data = {"retrieval_method": "hybrid", "top_k": 4}
        response = ChatResponse(
            answer="Test answer",
            sources=["doc1.pdf"],
            num_sources=1,
            has_context=True,
            session_id="session-123",
            debug_info=debug_data,
        )
        assert response.session_id == "session-123"
        assert response.debug_info == debug_data

    def test_response_no_context(self):
        """Test response when no context found."""
        response = ChatResponse(
            answer="I couldn't find relevant information.",
            sources=[],
            num_sources=0,
            has_context=False,
        )
        assert response.has_context is False
        assert len(response.sources) == 0
        assert response.num_sources == 0


class TestConversationHistory:
    """Tests for ConversationHistory model."""

    def test_new_conversation(self):
        """Test creating a new conversation history."""
        history = ConversationHistory(
            session_id="test-session",
            messages=[],
        )
        assert history.session_id == "test-session"
        assert len(history.messages) == 0
        assert isinstance(history.created_at, datetime)
        assert isinstance(history.updated_at, datetime)

    def test_conversation_with_messages(self):
        """Test conversation with message history."""
        messages = [
            ChatMessage(role="user", content="Question 1"),
            ChatMessage(role="assistant", content="Answer 1"),
            ChatMessage(role="user", content="Question 2"),
        ]
        history = ConversationHistory(
            session_id="test-session",
            messages=messages,
        )
        assert len(history.messages) == 3
        assert history.messages[0].role == "user"
        assert history.messages[1].role == "assistant"


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_basic_health_response(self):
        """Test basic health response."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            components={},
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert isinstance(response.timestamp, datetime)
        assert response.components == {}

    def test_health_with_components(self):
        """Test health response with component status."""
        components = {
            "database": "healthy",
            "vector_store": "healthy",
            "llm_api": "degraded",
        }
        response = HealthResponse(
            status="degraded",
            version="1.0.0",
            components=components,
        )
        assert response.status == "degraded"
        assert len(response.components) == 3
        assert response.components["llm_api"] == "degraded"


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    def test_basic_error_response(self):
        """Test basic error response."""
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid request data",
        )
        assert response.error == "ValidationError"
        assert response.message == "Invalid request data"
        assert response.details is None
        assert isinstance(response.timestamp, datetime)

    def test_error_with_details(self):
        """Test error response with additional details."""
        details = {"field": "query", "issue": "too short"}
        response = ErrorResponse(
            error="ValidationError",
            message="Invalid query",
            details=details,
        )
        assert response.details == details
        assert response.details["field"] == "query"


class TestSource:
    """Tests for Source model."""

    def test_source_basic(self):
        """Test basic source creation."""
        source = Source(name="kafka_docs.pdf")
        assert source.name == "kafka_docs.pdf"
        assert source.page is None
        assert source.relevance_score is None

    def test_source_with_metadata(self):
        """Test source with page and relevance score."""
        source = Source(
            name="kafka_docs.pdf",
            page=42,
            relevance_score=0.95,
        )
        assert source.name == "kafka_docs.pdf"
        assert source.page == 42
        assert source.relevance_score == 0.95
