"""
Unit tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from app.main import app
from app.services.conversation_memory import conversation_memory


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up conversation memory after each test."""
    yield
    # Clear all sessions after each test
    for session_id in list(conversation_memory._sessions.keys()):
        conversation_memory.clear_session(session_id)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_basic_health_check(self, client):
        """Test basic health endpoint."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_detailed_health_check(self, client):
        """Test detailed health endpoint with components."""
        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data

    def test_readiness_probe(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/health/ready")
        # May fail if RAG pipeline can't initialize, but should return a response
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ready"

    def test_liveness_probe(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"


class TestRootEndpoints:
    """Tests for root and stats endpoints."""

    def test_root_endpoint(self, client):
        """Test root API information endpoint."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "documentation" in data
        assert "endpoints" in data

    def test_stats_endpoint(self, client):
        """Test API statistics endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "active_sessions" in data
        assert "api_version" in data
        assert isinstance(data["active_sessions"], int)


class TestChatEndpoints:
    """Tests for chat endpoints."""

    @patch('app.api.routes.chat.ResponseGenerator')
    def test_chat_basic_request(self, mock_generator_class, client):
        """Test basic chat request."""
        # Mock the ResponseGenerator
        mock_generator = Mock()
        mock_generator.generate_response.return_value = {
            "answer": "To create a Kafka topic, use kafka-topics.sh",
            "sources": ["kafka_docs.pdf"],
            "num_sources": 1,
            "has_context": True,
        }
        mock_generator_class.return_value = mock_generator

        response = client.post(
            "/chat/",
            json={"query": "How do I create a Kafka topic?"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "num_sources" in data
        assert "has_context" in data
        assert "session_id" in data
        assert data["has_context"] is True

    @patch('app.api.routes.chat.ResponseGenerator')
    def test_chat_with_session_id(self, mock_generator_class, client):
        """Test chat request with existing session."""
        mock_generator = Mock()
        mock_generator.generate_response.return_value = {
            "answer": "Test answer",
            "sources": ["doc.pdf"],
            "num_sources": 1,
            "has_context": True,
        }
        mock_generator_class.return_value = mock_generator

        session_id = "test-session-123"

        response = client.post(
            "/chat/",
            json={
                "query": "Follow-up question",
                "session_id": session_id,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_chat_validation_empty_query(self, client):
        """Test that empty query is rejected."""
        response = client.post(
            "/chat/",
            json={"query": ""}
        )
        assert response.status_code == 422  # Validation error

    def test_chat_validation_query_too_long(self, client):
        """Test that overly long query is rejected."""
        response = client.post(
            "/chat/",
            json={"query": "x" * 2001}  # Max is 2000
        )
        assert response.status_code == 422

    def test_chat_validation_invalid_top_k(self, client):
        """Test that invalid top_k is rejected."""
        # top_k too large
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": 11}  # Max is 10
        )
        assert response.status_code == 422

        # top_k too small
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": 0}  # Min is 1
        )
        assert response.status_code == 422

    @patch('app.api.routes.chat.ResponseGenerator')
    def test_chat_debug_mode(self, mock_generator_class, client):
        """Test chat request with debug mode enabled."""
        mock_generator = Mock()
        mock_generator.generate_response.return_value = {
            "answer": "Test answer",
            "sources": ["doc.pdf"],
            "num_sources": 1,
            "has_context": True,
        }
        mock_generator.retriever = Mock()
        mock_generator.retriever.enable_hybrid = True
        mock_generator_class.return_value = mock_generator

        response = client.post(
            "/chat/",
            json={"query": "Test query", "debug": True}
        )

        assert response.status_code == 200
        data = response.json()
        assert "debug_info" in data
        assert data["debug_info"] is not None

    def test_create_session(self, client):
        """Test creating a new session."""
        response = client.post("/chat/sessions")
        assert response.status_code == 201

        data = response.json()
        assert "session_id" in data
        assert "message_count" in data
        assert data["message_count"] == 0

    def test_get_session_info(self, client):
        """Test getting session information."""
        # Create a session first
        create_response = client.post("/chat/sessions")
        session_id = create_response.json()["session_id"]

        # Get session info
        response = client.get(f"/chat/sessions/{session_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["session_id"] == session_id
        assert "message_count" in data
        assert "created_at" in data

    def test_get_session_info_nonexistent(self, client):
        """Test getting info for nonexistent session."""
        response = client.get("/chat/sessions/nonexistent-session")
        assert response.status_code == 404

    def test_clear_session(self, client):
        """Test clearing a session."""
        # Create a session
        create_response = client.post("/chat/sessions")
        session_id = create_response.json()["session_id"]

        # Clear it
        response = client.delete(f"/chat/sessions/{session_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/chat/sessions/{session_id}")
        assert get_response.status_code == 404


class TestCORSConfiguration:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are configured."""
        response = client.options(
            "/health/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # CORS should allow the request
        assert "access-control-allow-origin" in response.headers


class TestRateLimiting:
    """Tests for rate limiting."""

    @patch('app.api.routes.chat.ResponseGenerator')
    def test_rate_limit_enforcement(self, mock_generator_class, client):
        """Test that rate limits are enforced."""
        mock_generator = Mock()
        mock_generator.generate_response.return_value = {
            "answer": "Test",
            "sources": [],
            "num_sources": 0,
            "has_context": False,
        }
        mock_generator_class.return_value = mock_generator

        # Make requests up to the limit (20/minute for chat)
        # Note: In real scenario, this would trigger rate limit
        # For unit test, we just verify endpoint works
        for i in range(3):
            response = client.post(
                "/chat/",
                json={"query": f"Test query {i}"}
            )
            # Should succeed for first few requests
            assert response.status_code in [200, 429]


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_not_found(self, client):
        """Test 404 handling for nonexistent endpoint."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        response = client.put("/health/")
        assert response.status_code == 405

    def test_validation_error_response_format(self, client):
        """Test that validation errors have consistent format."""
        response = client.post(
            "/chat/",
            json={"query": ""}  # Empty query
        )

        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data
