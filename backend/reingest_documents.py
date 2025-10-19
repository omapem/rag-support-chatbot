#!/usr/bin/env python3
"""
Re-ingest documents with updated chunking parameters.
Run this script after changing chunk_size or chunk_overlap in config.py.

This will rebuild the vector database with the new settings.
"""

import os
import shutil
from pathlib import Path

from src.ingestion import DocumentIngester
from src.embeddings import EmbeddingManager
from src.config import settings


def main():
    """Re-ingest all documents with updated chunking parameters."""
    print("=" * 70)
    print("DOCUMENT RE-INGESTION SCRIPT")
    print("=" * 70)
    print(f"\nCurrent chunking settings:")
    print(f"  - Chunk size: {settings.chunk_size} characters")
    print(f"  - Chunk overlap: {settings.chunk_overlap} characters")
    print(f"  - Top-k retrieval: {settings.top_k} documents")
    print()

    # Check if vector store exists
    chroma_dir = Path(settings.chroma_persist_directory)
    if chroma_dir.exists():
        print(f"⚠️  Existing vector store found at: {chroma_dir}")
        response = input("Do you want to DELETE and rebuild it? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted. No changes made.")
            return

        print(f"Deleting existing vector store...")
        shutil.rmtree(chroma_dir)
        print("✅ Deleted old vector store")
    else:
        print("No existing vector store found. Creating new one...")

    # Find data directory
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Please create 'data/raw/' and add your PDF files.")
        return

    # Find PDF files
    pdf_files = list(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {data_dir}")
        print("Please add PDF files to the data/raw/ directory.")
        return

    print(f"\n📁 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Process documents
    print(f"\n🔄 Processing documents with new chunking settings...")
    ingester = DocumentIngester(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            documents = ingester.load_pdf(str(pdf_file))
            chunks = ingester.chunk_documents(documents)
            all_chunks.extend(chunks)
            print(f"  ✅ Created {len(chunks)} chunks")
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file.name}: {e}")

    if not all_chunks:
        print("\n❌ Error: No chunks created. Check your PDF files.")
        return

    print(f"\n📊 Total chunks created: {len(all_chunks)}")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in all_chunks) / len(all_chunks):.0f} characters")

    # Create vector store
    print(f"\n🔄 Creating vector embeddings and storing in Chroma...")
    embedding_manager = EmbeddingManager()
    try:
        vector_store = embedding_manager.create_vector_store(all_chunks)
        print(f"✅ Vector store created successfully!")
        print(f"   Location: {settings.chroma_persist_directory}")
        print(f"   Collection: {settings.chroma_collection_name}")
        print(f"   Documents: {len(all_chunks)} chunks")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        return

    # Test retrieval
    print(f"\n🧪 Testing retrieval with sample query...")
    try:
        from src.retrieval import Retriever

        retriever = Retriever(enable_hybrid=True)
        test_query = "How do I create a topic?"

        print(f"\nQuery: '{test_query}'")
        results = retriever.retrieve_and_format(test_query, debug=True)

        print(f"\n✅ Retrieval test successful!")
        print(f"   Retrieved {results['num_results']} documents")
        print(f"   Sources: {', '.join(results['sources'])}")

        # Show first result preview
        if results['documents']:
            first_doc = results['documents'][0]
            print(f"\n📄 First result preview:")
            print(f"   Source: {first_doc.metadata.get('doc_name', 'Unknown')}")
            print(f"   Page: {first_doc.metadata.get('page', '?')}")
            print(f"   Content: {first_doc.page_content[:200]}...")

    except Exception as e:
        print(f"⚠️  Warning: Retrieval test failed: {e}")
        print(f"   Vector store was created, but you may need to restart the application.")

    print("\n" + "=" * 70)
    print("✅ RE-INGESTION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Restart your Streamlit app: streamlit run streamlit_app.py")
    print("2. Test with the query: 'How do I create a topic?'")
    print("3. You should now see CLI commands from page 33!")
    print()


if __name__ == "__main__":
    main()
