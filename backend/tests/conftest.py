"""
Pytest configuration and shared fixtures.
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import patch

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings."""
    return {
        "test_mode": True,
        "log_level": "ERROR",  # Reduce logging noise during tests
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "How do I create a Kafka topic?"},
        {"role": "assistant", "content": "To create a Kafka topic, use kafka-topics.sh"},
        {"role": "user", "content": "What about replication factor?"},
        {"role": "assistant", "content": "Replication factor determines the number of replicas"},
    ]


@pytest.fixture
def sample_rag_response():
    """Sample RAG response for mocking."""
    return {
        "answer": "To create a Kafka topic, use the kafka-topics.sh script with the --create flag.",
        "sources": ["kafka_docs.pdf", "kafka_operations.pdf"],
        "num_sources": 2,
        "has_context": True,
        "context": "Sample context from documents...",
    }


@pytest.fixture(scope="session")
def test_vectordb_path():
    """Path to test vector database."""
    return backend_dir / "tests" / "test_vectordb"


@pytest.fixture(scope="session")
def use_test_vectordb(test_vectordb_path):
    """
    Configure tests to use test vector database instead of production database.
    This is a session-scoped fixture that patches the config before any tests run.
    """
    # Check if test vectordb exists
    if not test_vectordb_path.exists():
        pytest.skip("Test vector database not found. Run tests/setup_test_vectordb.py first.")

    # Patch the config to use test database
    with patch.dict(os.environ, {
        "CHROMA_PERSIST_DIRECTORY": str(test_vectordb_path),
        "CHROMA_COLLECTION_NAME": "kafka_test_docs",
    }):
        yield test_vectordb_path


@pytest.fixture
def test_vectordb(use_test_vectordb):
    """
    Provides access to the test vector database for integration tests.
    Automatically uses the test database instead of production.
    """
    from src.embeddings import EmbeddingManager

    manager = EmbeddingManager()
    manager.load_vector_store()

    yield manager

    # Cleanup if needed
    manager = None
