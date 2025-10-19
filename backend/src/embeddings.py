"""
Embedding generation and vector database management.
Handles creating embeddings and storing them in Chroma.
"""

from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import settings


class EmbeddingManager:
    """Manages embedding generation and vector database operations."""

    def __init__(self, collection_name: str = None):
        """
        Initialize the embedding manager.

        Args:
            collection_name: Name of the Chroma collection (defaults to settings)
        """
        self.collection_name = collection_name or settings.chroma_collection_name

        # Initialize embedding model (Sentence Transformers - FREE!)
        print(f"Loading embedding model: {settings.embedding_model}")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},  # Use 'cuda' if you have GPU
            encode_kwargs={"normalize_embeddings": True},
        )

        # Initialize or load Chroma vector store
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create a new vector store from documents.

        Args:
            documents: List of documents to embed and store

        Returns:
            Chroma vector store instance
        """
        print(f"Creating embeddings for {len(documents)} documents...")

        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=settings.chroma_persist_directory,
        )

        print(f"Vector store created and persisted to {settings.chroma_persist_directory}")
        return self.vector_store

    def load_vector_store(self) -> Chroma:
        """
        Load an existing vector store from disk.

        Returns:
            Chroma vector store instance
        """
        print(f"Loading existing vector store from {settings.chroma_persist_directory}")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=settings.chroma_persist_directory,
        )

        print("Vector store loaded successfully")
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to existing vector store.

        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store() first.")

        print(f"Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("Documents added successfully")

    def get_retriever(self, top_k: int = None, search_type: str = "similarity"):
        """
        Get a retriever for searching the vector store.

        Args:
            top_k: Number of documents to retrieve (defaults to settings)
            search_type: Type of search ('similarity' or 'mmr')

        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load_vector_store() first.")

        top_k = top_k or settings.top_k

        # Use similarity search by default (MMR was filtering out relevant results)
        search_kwargs = {"k": top_k}

        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )


# Example usage (for testing)
if __name__ == "__main__":
    from src.ingestion import DocumentIngester

    # Example workflow:
    # 1. Load and chunk documents
    # ingester = DocumentIngester()
    # chunks = ingester.process_pdf_directory("data/raw/pdfs")

    # 2. Create embeddings and vector store
    # embedding_manager = EmbeddingManager()
    # vector_store = embedding_manager.create_vector_store(chunks)

    # 3. Later, load existing vector store
    # embedding_manager = EmbeddingManager()
    # vector_store = embedding_manager.load_vector_store()

    # 4. Test retrieval
    # retriever = embedding_manager.get_retriever(top_k=3)
    # results = retriever.get_relevant_documents("How do I create a Kafka topic?")
    # for doc in results:
    #     print(f"Source: {doc.metadata['source']}")
    #     print(f"Content: {doc.page_content[:200]}...\n")

    print("Embedding module ready. Uncomment examples to test.")
