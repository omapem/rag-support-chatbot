"""
Document ingestion and chunking module.
Handles loading PDFs and web pages, then chunking them for embedding.
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

from src.config import settings


class DocumentIngester:
    """Handles document loading and chunking for RAG pipeline."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        """
        Initialize the document ingester.

        Args:
            chunk_size: Size of text chunks (defaults to settings)
            chunk_overlap: Overlap between chunks (defaults to settings)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Initialize text splitter with optimized separators to preserve code blocks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Preserve code blocks and command examples together
            separators=[
                "\n\n\n",  # Triple newlines (major section breaks)
                "\n\n",    # Double newlines (paragraph breaks)
                "\n#",     # Command prompts (preserve shell commands)
                "\n",      # Single newlines
                ". ",      # Sentences
                " ",       # Words
                "",        # Characters
            ],
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return documents.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata["source"] = pdf_path
            doc.metadata["source_type"] = "pdf"
            doc.metadata["doc_name"] = Path(pdf_path).name

        return documents

    def load_web_page(self, url: str) -> List[Document]:
        """
        Load a web page and return documents.

        Args:
            url: URL of the web page

        Returns:
            List of Document objects
        """
        loader = WebBaseLoader(url)
        documents = loader.load()

        # Add metadata
        for doc in documents:
            doc.metadata["source"] = url
            doc.metadata["source_type"] = "web"
            doc.metadata["doc_name"] = url

        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents
        """
        chunks = self.text_splitter.split_documents(documents)

        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks

    def process_pdf_directory(self, directory_path: str) -> List[Document]:
        """
        Process all PDFs in a directory.

        Args:
            directory_path: Path to directory containing PDFs

        Returns:
            List of chunked documents from all PDFs
        """
        all_chunks = []
        pdf_files = Path(directory_path).glob("*.pdf")

        for pdf_file in pdf_files:
            print(f"Processing {pdf_file.name}...")
            documents = self.load_pdf(str(pdf_file))
            chunks = self.chunk_documents(documents)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks from {pdf_file.name}")

        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks

    def process_urls(self, urls: List[str]) -> List[Document]:
        """
        Process multiple web URLs.

        Args:
            urls: List of URLs to process

        Returns:
            List of chunked documents from all URLs
        """
        all_chunks = []

        for url in urls:
            print(f"Processing {url}...")
            try:
                documents = self.load_web_page(url)
                chunks = self.chunk_documents(documents)
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks from {url}")
            except Exception as e:
                print(f"  Error processing {url}: {e}")

        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


# Example usage (for testing)
if __name__ == "__main__":
    # Initialize ingester
    ingester = DocumentIngester()

    # Example: Process a single PDF
    # chunks = ingester.load_pdf("data/raw/pdfs/kafka-guide.pdf")
    # chunks = ingester.chunk_documents(chunks)
    # print(f"Created {len(chunks)} chunks")

    # Example: Process all PDFs in directory
    # chunks = ingester.process_pdf_directory("data/raw/pdfs")

    # Example: Process web pages
    # kafka_urls = [
    #     "https://kafka.apache.org/documentation/",
    #     "https://kafka.apache.org/quickstart",
    # ]
    # chunks = ingester.process_urls(kafka_urls)

    print("Ingestion module ready. Uncomment examples to test.")
