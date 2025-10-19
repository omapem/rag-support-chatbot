"""
One-time script to create the vector database from your documents.
Run this after you've added PDFs to data/raw/pdfs/
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentIngester
from src.embeddings import EmbeddingManager
from src.config import settings


def main():
    print("🚀 RAG Vector Database Creation Script")
    print("=" * 50)

    # Initialize ingester
    ingester = DocumentIngester()

    # Check if PDFs exist
    pdf_dir = Path("data/raw/pdfs")
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print("\n⚠️  No PDF files found in data/raw/pdfs/")
        print("\nPlease add your Kafka documentation PDFs to:")
        print(f"  {pdf_dir.absolute()}")
        print("\nThen run this script again.")
        return

    print(f"\n📄 Found {len(pdf_files)} PDF file(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    # Process PDFs
    print("\n🔄 Processing PDFs and creating chunks...")
    chunks = ingester.process_pdf_directory(str(pdf_dir))

    if not chunks:
        print("\n❌ No chunks created. Check if PDFs are readable.")
        return

    print(f"\n✅ Created {len(chunks)} chunks from PDFs")

    # Optional: Add web pages
    print("\n🌐 Do you want to add Kafka documentation from web pages?")
    print("   (Recommended URLs: kafka.apache.org/documentation, quickstart)")
    add_web = input("\nAdd web pages? (y/n): ").lower().strip()

    if add_web == 'y':
        print("\nEnter URLs (one per line, empty line to finish):")
        urls = []
        while True:
            url = input("URL: ").strip()
            if not url:
                break
            urls.append(url)

        if urls:
            print(f"\n🔄 Processing {len(urls)} web page(s)...")
            web_chunks = ingester.process_urls(urls)
            chunks.extend(web_chunks)
            print(f"✅ Added {len(web_chunks)} chunks from web pages")

    print(f"\n📊 Total chunks: {len(chunks)}")

    # Show sample chunk
    print("\n📖 Sample chunk preview:")
    print("-" * 50)
    print(f"Source: {chunks[0].metadata.get('doc_name', 'Unknown')}")
    print(f"Content: {chunks[0].page_content[:300]}...")
    print("-" * 50)

    # Create vector database
    print("\n🔄 Creating vector database (this may take a few minutes)...")
    print("   First run will download the embedding model (~100MB)")

    embedding_manager = EmbeddingManager()
    vector_store = embedding_manager.create_vector_store(chunks)

    print("\n🎉 Vector database created successfully!")
    print(f"\n📍 Database location: {settings.chroma_persist_directory}")
    print(f"📚 Collection name: {embedding_manager.collection_name}")
    print(f"🔢 Total documents: {len(chunks)}")

    # Test retrieval
    print("\n🧪 Testing retrieval with sample query...")
    test_query = "What is Apache Kafka?"
    retriever = embedding_manager.get_retriever(top_k=3)
    results = retriever.get_relevant_documents(test_query)

    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n  {i}. Source: {doc.metadata.get('doc_name', 'Unknown')}")
        print(f"     Preview: {doc.page_content[:150]}...")

    print("\n" + "=" * 50)
    print("✨ Setup complete! You can now run the chatbot:")
    print("   streamlit run streamlit_app.py")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you're in the backend/ directory")
        print("  2. Check that .env file exists with ANTHROPIC_API_KEY")
        print("  3. Verify PDFs are in data/raw/pdfs/")
        print("  4. Make sure dependencies are installed: pip install -r requirements.txt")
