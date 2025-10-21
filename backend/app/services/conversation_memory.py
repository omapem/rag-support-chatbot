"""
In-memory conversation history management.
Handles storing and retrieving conversation sessions with TTL support.
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import threading
from collections import OrderedDict
import uuid

from app.models.schemas import ChatMessage, ConversationHistory


class ConversationMemory:
    """
    Thread-safe in-memory conversation storage with TTL and size limits.

    Features:
    - Automatic session expiration (TTL)
    - Maximum message history per session
    - Thread-safe operations
    - LRU eviction when max sessions reached
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        max_messages_per_session: int = 50,
        session_ttl_minutes: int = 60,
    ):
        """
        Initialize conversation memory.

        Args:
            max_sessions: Maximum number of active sessions
            max_messages_per_session: Maximum messages per session
            session_ttl_minutes: Session time-to-live in minutes
        """
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self.session_ttl = timedelta(minutes=session_ttl_minutes)

        # Thread-safe storage
        self._sessions: OrderedDict[str, ConversationHistory] = OrderedDict()
        self._lock = threading.Lock()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional session ID (generated if not provided)

        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        with self._lock:
            # Check if we need to evict old sessions
            if len(self._sessions) >= self.max_sessions:
                # Remove oldest session (LRU)
                self._sessions.popitem(last=False)

            # Create new session
            self._sessions[session_id] = ConversationHistory(
                session_id=session_id,
                messages=[],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Move to end (mark as recently used)
            self._sessions.move_to_end(session_id)

        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        auto_create: bool = True,
    ) -> None:
        """
        Add a message to a conversation session.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            auto_create: Create session if it doesn't exist

        Raises:
            ValueError: If session doesn't exist and auto_create is False
        """
        with self._lock:
            # Get or create session
            if session_id not in self._sessions:
                if auto_create:
                    self.create_session(session_id)
                else:
                    raise ValueError(f"Session {session_id} not found")

            session = self._sessions[session_id]

            # Check session expiration
            if self._is_expired(session):
                # Reset expired session
                session.messages = []
                session.created_at = datetime.utcnow()

            # Add message
            message = ChatMessage(
                role=role,
                content=content,
                timestamp=datetime.utcnow(),
            )
            session.messages.append(message)

            # Trim if exceeds max messages
            if len(session.messages) > self.max_messages_per_session:
                # Keep most recent messages
                session.messages = session.messages[-self.max_messages_per_session:]

            # Update timestamp and mark as recently used
            session.updated_at = datetime.utcnow()
            self._sessions.move_to_end(session_id)

    def get_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Retrieve conversation history for a session.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages to retrieve

        Returns:
            List of ChatMessage objects (most recent first if limited)
        """
        with self._lock:
            if session_id not in self._sessions:
                return []

            session = self._sessions[session_id]

            # Check expiration
            if self._is_expired(session):
                return []

            messages = session.messages

            # Return most recent messages if limited
            if max_messages and len(messages) > max_messages:
                return messages[-max_messages:]

            return messages

    def get_formatted_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> str:
        """
        Get conversation history formatted as a string.

        Args:
            session_id: Session identifier
            max_messages: Maximum number of recent messages

        Returns:
            Formatted conversation history string
        """
        messages = self.get_history(session_id, max_messages)

        if not messages:
            return ""

        formatted = []
        for msg in messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            formatted.append(f"{role_label}: {msg.content}")

        return "\n\n".join(formatted)

    def clear_session(self, session_id: str) -> None:
        """
        Clear a conversation session.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        with self._lock:
            expired_sessions = [
                sid for sid, session in self._sessions.items()
                if self._is_expired(session)
            ]

            for sid in expired_sessions:
                del self._sessions[sid]

            return len(expired_sessions)

    def _is_expired(self, session: ConversationHistory) -> bool:
        """
        Check if a session has expired.

        Args:
            session: ConversationHistory object

        Returns:
            True if expired, False otherwise
        """
        age = datetime.utcnow() - session.updated_at
        return age > self.session_ttl

    def get_session_count(self) -> int:
        """
        Get number of active sessions.

        Returns:
            Number of sessions in memory
        """
        with self._lock:
            return len(self._sessions)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get session metadata.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session info or None
        """
        with self._lock:
            if session_id not in self._sessions:
                return None

            session = self._sessions[session_id]
            return {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "is_expired": self._is_expired(session),
            }


# Singleton instance
conversation_memory = ConversationMemory(
    max_sessions=1000,
    max_messages_per_session=50,
    session_ttl_minutes=60,
)
