"""
Script to set up test vector database for integration tests.
Creates a small Chroma database with test Kafka documentation.
"""

import os
import sys
from pathlib import Path

# Add backend to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def setup_test_vectordb():
    """Set up test vector database with sample Kafka documents."""

    print("=" * 60)
    print("Setting up test vector database for integration tests")
    print("=" * 60)

    # Paths
    test_data_dir = backend_dir / "tests" / "test_data"
    test_db_dir = backend_dir / "tests" / "test_vectordb"

    # Clean up existing test database
    if test_db_dir.exists():
        print(f"\nCleaning up existing test database at {test_db_dir}")
        import shutil
        shutil.rmtree(test_db_dir)

    test_db_dir.mkdir(parents=True, exist_ok=True)

    # Load test documents
    print(f"\nLoading test documents from {test_data_dir}")
    documents = []

    for txt_file in test_data_dir.glob("*.txt"):
        print(f"  Loading: {txt_file.name}")
        loader = TextLoader(str(txt_file))
        docs = loader.load()

        # Add metadata
        for doc in docs:
            doc.metadata["source"] = txt_file.name
            doc.metadata["doc_name"] = txt_file.stem

        documents.extend(docs)

    print(f"\nLoaded {len(documents)} documents")

    # Split documents into chunks
    print("\nSplitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Smaller chunks for test data
        chunk_overlap=50,
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Initialize embeddings
    print("\nInitializing embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # Create vector store
    print(f"\nCreating Chroma vector store at {test_db_dir}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(test_db_dir),
        collection_name="kafka_test_docs",
    )

    print(f"\n✅ Test vector database created successfully!")
    print(f"   Location: {test_db_dir}")
    print(f"   Documents: {len(documents)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Collection: kafka_test_docs")

    # Test a query
    print("\nTesting retrieval...")
    results = vectorstore.similarity_search("How do I create a Kafka topic?", k=2)
    print(f"  Found {len(results)} relevant chunks")
    if results:
        print(f"  Top result: {results[0].page_content[:100]}...")

    print("\n" + "=" * 60)
    print("Setup complete! Integration tests can now use this database.")
    print("=" * 60)

    return vectorstore


if __name__ == "__main__":
    try:
        setup_test_vectordb()
    except Exception as e:
        print(f"\n❌ Error setting up test vector database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
