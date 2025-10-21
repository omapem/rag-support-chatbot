"""
Unit tests for conversation memory service.
"""

import pytest
import time
from datetime import datetime, timedelta

from app.services.conversation_memory import ConversationMemory
from app.models.schemas import ChatMessage


class TestConversationMemory:
    """Tests for ConversationMemory service."""

    @pytest.fixture
    def memory(self):
        """Create a fresh ConversationMemory instance for each test."""
        return ConversationMemory(
            max_sessions=100,
            max_messages_per_session=20,
            session_ttl_minutes=1,  # Short TTL for testing
        )

    def test_create_session(self, memory):
        """Test creating a new session."""
        session_id = memory.create_session()
        assert session_id is not None
        assert memory.get_session_count() == 1

    def test_create_session_with_custom_id(self, memory):
        """Test creating a session with custom ID."""
        custom_id = "my-custom-session-id"
        session_id = memory.create_session(session_id=custom_id)
        assert session_id == custom_id
        assert memory.get_session_count() == 1

    def test_add_message_new_session(self, memory):
        """Test adding message creates session automatically."""
        session_id = "test-session"
        memory.add_message(session_id, "user", "Hello")

        history = memory.get_history(session_id)
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "Hello"

    def test_add_message_no_autocreate(self, memory):
        """Test that add_message raises error when auto_create=False."""
        with pytest.raises(ValueError, match="Session .* not found"):
            memory.add_message("nonexistent", "user", "Hello", auto_create=False)

    def test_conversation_flow(self, memory):
        """Test a complete conversation flow."""
        session_id = "conversation-1"

        # User question
        memory.add_message(session_id, "user", "How do I create a Kafka topic?")
        # Assistant response
        memory.add_message(session_id, "assistant", "Use kafka-topics.sh --create...")
        # Follow-up question
        memory.add_message(session_id, "user", "What about replication factor?")
        # Follow-up response
        memory.add_message(session_id, "assistant", "Replication factor determines...")

        history = memory.get_history(session_id)
        assert len(history) == 4
        assert history[0].role == "user"
        assert history[1].role == "assistant"
        assert history[2].role == "user"
        assert history[3].role == "assistant"

    def test_max_messages_trimming(self, memory):
        """Test that old messages are trimmed when exceeding max."""
        session_id = "trim-test"

        # Add more messages than max (20)
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            memory.add_message(session_id, role, f"Message {i}")

        history = memory.get_history(session_id)
        # Should only keep the most recent 20 messages
        assert len(history) == 20
        # First message should be "Message 5" (trimmed first 5)
        assert history[0].content == "Message 5"
        # Last message should be "Message 24"
        assert history[-1].content == "Message 24"

    def test_get_history_limited(self, memory):
        """Test retrieving limited number of messages."""
        session_id = "limit-test"

        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            memory.add_message(session_id, role, f"Message {i}")

        # Get only last 3 messages
        history = memory.get_history(session_id, max_messages=3)
        assert len(history) == 3
        assert history[0].content == "Message 7"
        assert history[-1].content == "Message 9"

    def test_get_formatted_history(self, memory):
        """Test formatted history output."""
        session_id = "format-test"

        memory.add_message(session_id, "user", "Question 1")
        memory.add_message(session_id, "assistant", "Answer 1")
        memory.add_message(session_id, "user", "Question 2")

        formatted = memory.get_formatted_history(session_id)

        assert "User: Question 1" in formatted
        assert "Assistant: Answer 1" in formatted
        assert "User: Question 2" in formatted
        # Check messages are separated
        assert "\n\n" in formatted

    def test_get_formatted_history_empty(self, memory):
        """Test formatted history for nonexistent session."""
        formatted = memory.get_formatted_history("nonexistent")
        assert formatted == ""

    def test_clear_session(self, memory):
        """Test clearing a session."""
        session_id = "clear-test"

        memory.add_message(session_id, "user", "Test message")
        assert memory.get_session_count() == 1

        memory.clear_session(session_id)
        assert memory.get_session_count() == 0

        # Verify history is gone
        history = memory.get_history(session_id)
        assert len(history) == 0

    def test_max_sessions_lru_eviction(self, memory):
        """Test LRU eviction when max sessions reached."""
        # Create max sessions (100)
        for i in range(100):
            session_id = f"session-{i}"
            memory.create_session(session_id)

        assert memory.get_session_count() == 100

        # Create one more - should evict oldest (session-0)
        memory.create_session("session-100")
        assert memory.get_session_count() == 100

        # session-0 should be gone
        info = memory.get_session_info("session-0")
        assert info is None

        # session-100 should exist
        info = memory.get_session_info("session-100")
        assert info is not None

    def test_session_info(self, memory):
        """Test getting session metadata."""
        session_id = "info-test"

        memory.add_message(session_id, "user", "Message 1")
        memory.add_message(session_id, "assistant", "Message 2")

        info = memory.get_session_info(session_id)

        assert info is not None
        assert info["session_id"] == session_id
        assert info["message_count"] == 2
        assert "created_at" in info
        assert "updated_at" in info
        assert "is_expired" in info

    def test_session_info_nonexistent(self, memory):
        """Test getting info for nonexistent session."""
        info = memory.get_session_info("nonexistent")
        assert info is None

    def test_session_expiration(self, memory):
        """Test that sessions expire after TTL."""
        session_id = "expire-test"

        # Create session and add message
        memory.add_message(session_id, "user", "Test message")

        # Verify session exists
        history = memory.get_history(session_id)
        assert len(history) == 1

        # Wait for session to expire (TTL is 1 minute in fixture)
        # For testing, we'll manipulate the session directly
        session = memory._sessions[session_id]
        # Set updated_at to 2 minutes ago
        session.updated_at = datetime.utcnow() - timedelta(minutes=2)

        # Now history should be empty (expired)
        history = memory.get_history(session_id)
        assert len(history) == 0

        # Session info should show expired
        info = memory.get_session_info(session_id)
        assert info["is_expired"] is True

    def test_cleanup_expired_sessions(self, memory):
        """Test cleanup of expired sessions."""
        # Create multiple sessions
        for i in range(5):
            session_id = f"session-{i}"
            memory.add_message(session_id, "user", f"Message {i}")

        assert memory.get_session_count() == 5

        # Expire sessions 0-2 by manipulating timestamps
        for i in range(3):
            session_id = f"session-{i}"
            session = memory._sessions[session_id]
            session.updated_at = datetime.utcnow() - timedelta(minutes=2)

        # Run cleanup
        removed = memory.cleanup_expired_sessions()

        assert removed == 3
        assert memory.get_session_count() == 2

        # Verify correct sessions remain
        assert memory.get_session_info("session-3") is not None
        assert memory.get_session_info("session-4") is not None
        assert memory.get_session_info("session-0") is None

    def test_thread_safety(self, memory):
        """Test basic thread safety of operations."""
        import threading

        session_id = "thread-test"
        num_threads = 10
        messages_per_thread = 5

        def add_messages(thread_id):
            for i in range(messages_per_thread):
                memory.add_message(
                    session_id,
                    "user",
                    f"Thread {thread_id}, Message {i}"
                )

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=add_messages, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have all messages (up to max_messages_per_session limit)
        history = memory.get_history(session_id)
        # With max_messages_per_session=20, we'll get the last 20
        assert len(history) == 20

    def test_lru_update_on_access(self, memory):
        """Test that accessing a session updates LRU order."""
        # Create sessions in order
        for i in range(5):
            memory.create_session(f"session-{i}")

        # Access session-0 (should move to end)
        memory.add_message("session-0", "user", "New message")

        # Now create sessions until we hit max (100)
        for i in range(5, 100):
            memory.create_session(f"session-{i}")

        # Create one more to trigger eviction
        memory.create_session("session-100")

        # session-1 should be evicted (oldest that wasn't accessed)
        assert memory.get_session_info("session-1") is None
        # session-0 should still exist (was recently accessed)
        assert memory.get_session_info("session-0") is not None
