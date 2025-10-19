"""
Script to rebuild vector database with better chunking and deduplication.
Run this to fix retrieval quality issues.
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentIngester
from src.embeddings import EmbeddingManager
from src.config import settings


def main():
    print("🔄 Rebuilding Vector Database")
    print("=" * 70)

    # Delete old database
    db_path = Path(settings.chroma_persist_directory)
    if db_path.exists():
        print(f"\n🗑️  Deleting old database: {db_path}")
        shutil.rmtree(db_path)

    # Initialize ingester with updated settings
    ingester = DocumentIngester(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )

    # Process PDFs
    pdf_dir = Path("data/raw/pdfs")
    print(f"\n📄 Processing PDFs from: {pdf_dir}")
    chunks = ingester.process_pdf_directory(str(pdf_dir))

    # Deduplicate chunks
    print(f"\n🔍 Deduplicating chunks...")
    unique_chunks = []
    seen_content = set()

    for chunk in chunks:
        content_hash = hash(chunk.page_content)
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_chunks.append(chunk)
        else:
            print(f"  Skipping duplicate from page {chunk.metadata.get('page', 'N/A')}")

    print(f"✅ Reduced {len(chunks)} → {len(unique_chunks)} chunks (removed {len(chunks) - len(unique_chunks)} duplicates)")

    # Create new vector database
    print(f"\n🔄 Creating new vector database...")
    embedding_manager = EmbeddingManager()
    vector_store = embedding_manager.create_vector_store(unique_chunks)

    print(f"\n✅ Vector database rebuilt successfully!")
    print(f"📍 Location: {settings.chroma_persist_directory}")
    print(f"🔢 Total unique documents: {len(unique_chunks)}")

    # Test retrieval
    print(f"\n🧪 Testing retrieval...")
    test_query = "How do I create a Kafka topic?"
    retriever = embedding_manager.get_retriever(top_k=8)
    results = retriever.invoke(test_query)

    print(f"\nQuery: '{test_query}'")
    print(f"Retrieved {len(results)} documents:")

    for i, doc in enumerate(results, 1):
        has_command = '✅ HAS kafka-topics' if 'kafka-topics' in doc.page_content else ''
        print(f"  {i}. Page {doc.metadata.get('page', 'N/A')} {has_command}")
        if 'kafka-topics.sh --create' in doc.page_content:
            print(f"     🎯 FOUND THE ANSWER!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
