"""
Hybrid retrieval combining semantic search with BM25 keyword search.
This significantly improves retrieval quality for technical "how-to" questions.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

from src.embeddings import EmbeddingManager
from src.config import settings


class HybridRetriever:
    """
    Combines semantic search (embeddings) with keyword search (BM25)
    for better retrieval quality on technical documentation.
    """

    def __init__(self):
        """Initialize hybrid retriever with both semantic and keyword components."""
        # Semantic component
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.load_vector_store()

        # Get all documents for BM25 indexing
        all_data = self.embedding_manager.vector_store.get()
        self.all_documents = []

        for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
            self.all_documents.append(
                Document(page_content=doc_text, metadata=metadata)
            )

        # Build BM25 index
        print(f"Building BM25 index for {len(self.all_documents)} documents...")
        tokenized_docs = [doc.page_content.lower().split() for doc in self.all_documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("BM25 index ready")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: User's question
            top_k: Number of documents to retrieve (defaults to settings)
            semantic_weight: Weight for semantic similarity (0-1)
            bm25_weight: Weight for BM25 keyword matching (0-1)

        Returns:
            List of relevant Document objects
        """
        top_k = top_k or settings.top_k

        # Semantic search
        semantic_results = self.embedding_manager.vector_store.similarity_search_with_score(
            query, k=top_k * 2  # Get more candidates
        )

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize scores to 0-1 range
        semantic_scores_dict = {
            id(doc): 1.0 / (1.0 + score)  # Lower distance = higher score
            for doc, score in semantic_results
        }

        bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_scores_normalized = bm25_scores / bm25_max

        # Combine scores
        final_scores = []
        for i, doc in enumerate(self.all_documents):
            doc_id = id(doc)
            semantic_score = semantic_scores_dict.get(doc_id, 0.0)
            bm25_score = bm25_scores_normalized[i]

            # Weighted combination
            combined_score = (
                semantic_weight * semantic_score +
                bm25_weight * bm25_score
            )

            final_scores.append((doc, combined_score))

        # Sort by combined score and return top k
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in final_scores[:top_k]]

    def retrieve_and_format(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        Retrieve documents and format them for the LLM.

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with 'documents', 'context', and 'sources'
        """
        documents = self.retrieve(query, top_k=top_k)

        # Format context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("doc_name", "Unknown source")
            content = doc.page_content.strip()
            context_parts.append(
                f"[Document {i}] (Source: {source})\n{content}\n"
            )

        context = "\n---\n".join(context_parts)

        # Extract unique sources
        sources = list(set(doc.metadata.get("doc_name", "Unknown") for doc in documents))

        return {
            "documents": documents,
            "context": context,
            "sources": sources,
            "num_results": len(documents),
        }


# Test hybrid retrieval
if __name__ == "__main__":
    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    print("Testing Hybrid Retrieval")
    print("=" * 70)

    retriever = HybridRetriever()

    test_query = "How do I create a Kafka topic?"
    results = retriever.retrieve(test_query, top_k=10)

    print(f"\nQuery: '{test_query}'")
    print(f"Top {len(results)} results:\n")

    for i, doc in enumerate(results, 1):
        page = doc.metadata.get('page', 'N/A')
        has_cmd = '🎯 CLI' if 'kafka-topics.sh --create' in doc.page_content else ''
        has_text = '📝' if any(term in doc.page_content.lower() for term in ['create topic', 'creating topic']) else ''

        print(f"{i:2}. Page {page:3} {has_cmd} {has_text}")

        if i == 1:
            print(f"    Preview: {doc.page_content[:150]}...")
