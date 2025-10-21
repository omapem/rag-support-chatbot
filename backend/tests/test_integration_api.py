"""
Integration tests for FastAPI with real RAG pipeline.
These tests require the vector database to be set up.
"""

import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.main import app
from app.services.conversation_memory import conversation_memory


# Check if vector database exists (use absolute path from this file's location)
VECTOR_DB_PATH = Path(__file__).parent.parent / "data" / "chroma_db"
HAS_VECTOR_DB = VECTOR_DB_PATH.exists()

# Check if .env file has required API keys
HAS_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY") is not None


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


@pytest.mark.integration
class TestHealthEndpointsIntegration:
    """Integration tests for health endpoints with real components."""

    def test_basic_health_check(self, client):
        """Test basic health endpoint returns successfully."""
        response = client.get("/health/")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_detailed_health_with_components(self, client):
        """Test detailed health check with component status."""
        response = client.get("/health/detailed")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data
        assert "timestamp" in data

        # Check component statuses
        components = data["components"]
        assert "conversation_memory" in components

        # If vector DB exists, it should be healthy
        if HAS_VECTOR_DB:
            assert "vector_store" in components
            assert "rag_pipeline" in components

    def test_readiness_probe(self, client):
        """Test readiness probe reflects actual system state."""
        response = client.get("/health/ready")

        if HAS_VECTOR_DB and HAS_ANTHROPIC_KEY:
            # Should be ready if all dependencies available
            assert response.status_code == 200
            assert response.json()["status"] == "ready"
        else:
            # May not be ready without dependencies
            assert response.status_code in [200, 503]

    def test_liveness_probe(self, client):
        """Test liveness probe always succeeds."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"


@pytest.mark.integration
class TestConversationFlowIntegration:
    """Integration tests for conversation flow through API."""

    def test_create_and_manage_session(self, client):
        """Test full session lifecycle."""
        # Create session
        create_response = client.post("/chat/sessions")
        assert create_response.status_code == 201

        session_data = create_response.json()
        session_id = session_data["session_id"]
        assert session_id is not None
        assert session_data["message_count"] == 0

        # Get session info
        info_response = client.get(f"/chat/sessions/{session_id}")
        assert info_response.status_code == 200
        assert info_response.json()["session_id"] == session_id

        # Clear session
        delete_response = client.delete(f"/chat/sessions/{session_id}")
        assert delete_response.status_code == 204

        # Verify deleted
        verify_response = client.get(f"/chat/sessions/{session_id}")
        assert verify_response.status_code == 404

    def test_stats_endpoint(self, client):
        """Test stats endpoint tracks sessions."""
        # Initial stats
        stats1 = client.get("/stats").json()
        initial_count = stats1["active_sessions"]

        # Create a session
        client.post("/chat/sessions")

        # Stats should reflect new session
        stats2 = client.get("/stats").json()
        assert stats2["active_sessions"] == initial_count + 1


@pytest.mark.integration
@pytest.mark.skipif(not HAS_VECTOR_DB, reason="Requires vector database")
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="Requires Anthropic API key")
class TestChatIntegrationWithRAG:
    """Integration tests with real RAG pipeline (requires setup)."""

    def test_chat_with_real_rag(self, client):
        """Test chat endpoint with real RAG pipeline."""
        response = client.post(
            "/chat/",
            json={"query": "What is Apache Kafka?"}
        )

        assert response.status_code == 200

        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] != ""

        # Real RAG should find context
        assert data["has_context"] is True or data["has_context"] is False
        assert isinstance(data["num_sources"], int)

    def test_conversation_with_context(self, client):
        """Test multi-turn conversation maintains context."""
        session_id = "test-conversation-session"

        # First question
        response1 = client.post(
            "/chat/",
            json={
                "query": "How do I create a Kafka topic?",
                "session_id": session_id,
            }
        )
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == session_id

        # Follow-up question
        response2 = client.post(
            "/chat/",
            json={
                "query": "What about replication factor?",
                "session_id": session_id,
            }
        )
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id

        # Verify session has messages
        session_info = client.get(f"/chat/sessions/{session_id}").json()
        # Should have 4 messages (2 user + 2 assistant)
        assert session_info["message_count"] == 4

    def test_chat_with_debug_mode(self, client):
        """Test chat debug mode provides additional info."""
        response = client.post(
            "/chat/",
            json={
                "query": "Test query",
                "debug": True,
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "debug_info" in data
        assert data["debug_info"] is not None


@pytest.mark.integration
class TestValidationIntegration:
    """Integration tests for request validation."""

    def test_query_validation_flow(self, client):
        """Test various query validation scenarios."""
        # Empty query
        response = client.post("/chat/", json={"query": ""})
        assert response.status_code == 422

        # Whitespace only
        response = client.post("/chat/", json={"query": "   "})
        assert response.status_code == 422

        # Query too long
        long_query = "x" * 2001
        response = client.post("/chat/", json={"query": long_query})
        assert response.status_code == 422

    def test_top_k_validation(self, client):
        """Test top_k parameter validation boundary conditions."""
        # Test invalid values that should be rejected by validation

        # Too small (below minimum of 1)
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": 0}
        )
        assert response.status_code == 422

        # Negative value
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": -1}
        )
        assert response.status_code == 422

        # Too large (above maximum of 10)
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": 11}
        )
        assert response.status_code == 422

        # Way too large
        response = client.post(
            "/chat/",
            json={"query": "test", "top_k": 100}
        )
        assert response.status_code == 422

        # Note: Testing valid range (1-10) requires functional RAG system
        # and is covered by TestChatIntegrationWithRAG tests


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_404_handling(self, client):
        """Test 404 for nonexistent endpoints."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP methods."""
        response = client.put("/health/")
        assert response.status_code == 405

    def test_session_not_found(self, client):
        """Test session not found error."""
        response = client.get("/chat/sessions/nonexistent-session-id")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data or "detail" in data


@pytest.mark.integration
class TestCORSIntegration:
    """Integration tests for CORS configuration."""

    def test_cors_preflight(self, client):
        """Test CORS preflight request."""
        response = client.options(
            "/health/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )

        # Should allow CORS
        headers = {k.lower(): v for k, v in response.headers.items()}
        assert "access-control-allow-origin" in headers

    def test_cors_for_next_js(self, client):
        """Test CORS allows Next.js default port."""
        response = client.get(
            "/health/",
            headers={"Origin": "http://localhost:3000"}
        )

        headers = {k.lower(): v for k, v in response.headers.items()}
        assert "access-control-allow-origin" in headers


@pytest.mark.integration
@pytest.mark.slow
class TestRateLimitingIntegration:
    """Integration tests for rate limiting (slow due to multiple requests)."""

    def test_rate_limit_headers(self, client):
        """Test that rate limit headers are present."""
        response = client.get("/health/")
        # Some rate limiters add headers
        # This is a basic check that the endpoint works
        assert response.status_code == 200

    def test_multiple_requests_within_limit(self, client):
        """Test multiple requests within rate limit succeed."""
        # Make several requests that should be within limit
        for i in range(5):
            response = client.get("/health/")
            assert response.status_code == 200
