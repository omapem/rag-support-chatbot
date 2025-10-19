"""
Configuration management for RAG Support Chatbot.
Loads environment variables and provides application settings.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Environment
    environment: str = "development"

    # API Keys
    anthropic_api_key: str
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None

    # RAG Configuration
    chunk_size: int = 1000  # Increased from 800 to preserve more context
    chunk_overlap: int = 200  # Increased from 100 to prevent splitting related content
    top_k: int = 6  # Increased from 4 to improve retrieval coverage
    similarity_threshold: float = 0.7

    # Model Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "claude-3-5-sonnet-20241022"
    llm_temperature: float = 0.3
    max_tokens: int = 1024

    # Vector Database (Chroma - Local)
    chroma_persist_directory: str = "./data/chroma_db"
    chroma_collection_name: str = "kafka_docs"
    anonymized_telemetry: bool = False

    # Application Settings
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env


# Singleton instance
settings = Settings()
