"""
Retrieval module for semantic search.
Handles querying the vector database and retrieving relevant chunks.
Supports hybrid search combining BM25 (keyword) and semantic search.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from src.embeddings import EmbeddingManager
from src.config import settings


# Query expansion mappings for common Kafka questions
QUERY_EXPANSIONS = {
    "create topic": ["kafka-topics.sh", "topic creation", "new topic", "--create"],
    "delete topic": ["kafka-topics.sh", "topic deletion", "remove topic", "--delete"],
    "list topics": ["kafka-topics.sh", "show topics", "describe", "--list"],
    "consumer group": ["kafka-consumer-groups.sh", "consumer offset", "group management"],
    "producer": ["kafka-console-producer.sh", "produce messages", "send messages"],
    "consumer": ["kafka-console-consumer.sh", "consume messages", "read messages"],
    "configuration": ["broker config", "server.properties", "configure"],
    "retention": ["log retention", "retention policy", "log.retention"],
    "partition": ["partitioning", "partition assignment", "num.partitions"],
    "replication": ["replication factor", "replica", "replicas"],
}


class Retriever:
    """Handles semantic search and document retrieval with hybrid search support."""

    def __init__(self, enable_hybrid: bool = True):
        """
        Initialize the retriever with vector store.

        Args:
            enable_hybrid: If True, use hybrid search (BM25 + semantic)
        """
        self.embedding_manager = EmbeddingManager()
        self.embedding_manager.load_vector_store()
        self.retriever = self.embedding_manager.get_retriever(top_k=settings.top_k)
        self.enable_hybrid = enable_hybrid
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[EnsembleRetriever] = None

        # Initialize hybrid search if enabled
        if enable_hybrid:
            self._initialize_hybrid_retriever()

    def _initialize_hybrid_retriever(self) -> None:
        """Initialize BM25 and hybrid retrievers for keyword + semantic search."""
        try:
            # Get all documents from vector store for BM25
            all_docs = self.embedding_manager.vector_store.get()
            if all_docs and 'documents' in all_docs:
                # Convert to Document objects
                documents = [
                    Document(
                        page_content=doc,
                        metadata=meta
                    )
                    for doc, meta in zip(
                        all_docs['documents'],
                        all_docs['metadatas']
                    )
                ]

                # Initialize BM25 retriever
                self.bm25_retriever = BM25Retriever.from_documents(documents)
                self.bm25_retriever.k = settings.top_k

                # Create ensemble retriever (70% semantic, 30% keyword)
                self.hybrid_retriever = EnsembleRetriever(
                    retrievers=[self.retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
                print("Hybrid search (BM25 + semantic) initialized successfully")
            else:
                print("Warning: Could not initialize BM25 - no documents in vector store")
                self.enable_hybrid = False
        except Exception as e:
            print(f"Warning: Hybrid search initialization failed: {e}")
            print("Falling back to semantic search only")
            self.enable_hybrid = False

    def expand_query(self, query: str) -> str:
        """
        Expand query with related technical terms to improve retrieval.

        Args:
            query: Original user query

        Returns:
            Expanded query with additional relevant terms
        """
        query_lower = query.lower()
        expanded_terms = []

        # Check for matching patterns
        for pattern, expansions in QUERY_EXPANSIONS.items():
            if pattern in query_lower:
                expanded_terms.extend(expansions)

        # Return expanded query if we found matches
        if expanded_terms:
            # Add top 2-3 most relevant expansion terms
            expansion_str = " ".join(expanded_terms[:3])
            expanded = f"{query} {expansion_str}"
            return expanded

        return query

    def retrieve(self, query: str, top_k: int = None, use_hybrid: bool = None, expand_query: bool = True) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User's question
            top_k: Number of documents to retrieve (defaults to settings)
            use_hybrid: Override to force hybrid or semantic-only search
            expand_query: If True, expand query with related technical terms

        Returns:
            List of relevant Document objects
        """
        # Expand query if enabled
        search_query = self.expand_query(query) if expand_query else query
        if search_query != query:
            print(f"Query expanded: '{query}' → '{search_query}'")

        # Determine whether to use hybrid search
        should_use_hybrid = use_hybrid if use_hybrid is not None else self.enable_hybrid

        if top_k and top_k != settings.top_k:
            # Update BM25 retriever k if needed
            if should_use_hybrid and self.bm25_retriever:
                self.bm25_retriever.k = top_k

            # Create a new retriever with custom top_k
            retriever = self.embedding_manager.get_retriever(top_k=top_k)

            if should_use_hybrid and self.bm25_retriever:
                # Use hybrid with updated k
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[retriever, self.bm25_retriever],
                    weights=[0.7, 0.3]
                )
                documents = hybrid_retriever.invoke(search_query)
            else:
                documents = retriever.invoke(search_query)
        else:
            # Use default retrievers
            if should_use_hybrid and self.hybrid_retriever:
                documents = self.hybrid_retriever.invoke(search_query)
            else:
                documents = self.retriever.invoke(search_query)

        return documents

    def retrieve_with_scores(self, query: str, top_k: int = None, apply_reranking: bool = True) -> List[tuple]:
        """
        Retrieve documents with similarity scores and optional re-ranking.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            apply_reranking: If True, apply relevance-based re-ranking

        Returns:
            List of (Document, score) tuples sorted by relevance
        """
        top_k = top_k or settings.top_k

        # Retrieve more candidates for re-ranking
        fetch_k = top_k * 2 if apply_reranking else top_k

        results = self.embedding_manager.vector_store.similarity_search_with_score(
            query, k=fetch_k
        )

        if apply_reranking and len(results) > 0:
            results = self._rerank_results(query, results, top_k)

        return results

    def _rerank_results(self, query: str, results: List[tuple], top_k: int) -> List[tuple]:
        """
        Re-rank results based on multiple relevance signals.

        Args:
            query: Original query
            results: Initial (Document, score) tuples
            top_k: Number of results to return

        Returns:
            Re-ranked (Document, score) tuples
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        scored_results = []
        for doc, base_score in results:
            content_lower = doc.page_content.lower()

            # Relevance signals
            relevance_score = 0.0

            # 1. Base semantic similarity (normalized to 0-1, inverted for distance)
            semantic_score = 1.0 / (1.0 + base_score)
            relevance_score += semantic_score * 0.5

            # 2. Query term coverage (how many query terms appear in doc)
            term_matches = sum(1 for term in query_terms if term in content_lower)
            term_coverage = term_matches / len(query_terms) if query_terms else 0
            relevance_score += term_coverage * 0.2

            # 3. Document length penalty (prefer moderate lengths)
            doc_length = len(doc.page_content)
            ideal_length = 800
            length_score = 1.0 - min(abs(doc_length - ideal_length) / ideal_length, 1.0)
            relevance_score += length_score * 0.1

            # 4. Position of query terms (boost if terms appear early)
            first_match_pos = min(
                (content_lower.find(term) for term in query_terms if term in content_lower),
                default=len(content_lower)
            )
            position_score = 1.0 - (first_match_pos / len(content_lower))
            relevance_score += position_score * 0.1

            # 5. Metadata relevance (boost for certain document types)
            doc_type = doc.metadata.get('source_type', '')
            if doc_type == 'pdf':  # Official docs are higher quality
                relevance_score += 0.1

            scored_results.append((doc, relevance_score))

        # Sort by relevance score (descending) and return top_k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]

    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            documents: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("doc_name", "Unknown source")
            content = doc.page_content.strip()

            context_parts.append(
                f"[Document {i}] (Source: {source})\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def retrieve_and_format(self, query: str, top_k: int = None, debug: bool = False) -> Dict[str, Any]:
        """
        Retrieve documents and format them for the LLM.

        Args:
            query: User's question
            top_k: Number of documents to retrieve
            debug: If True, include similarity scores and ranking info

        Returns:
            Dictionary with 'documents', 'context', 'sources', and optional 'debug_info'
        """
        # Get documents with scores for debugging
        if debug:
            results_with_scores = self.retrieve_with_scores(query, top_k=top_k)
            documents = [doc for doc, score in results_with_scores]
            scores = [float(score) for doc, score in results_with_scores]

            # Log retrieval diagnostics
            print(f"\n=== RETRIEVAL DEBUG ===")
            print(f"Query: {query}")
            print(f"Top-k: {top_k or settings.top_k}")
            print(f"\nRetrieved {len(documents)} documents:")
            for i, (doc, score) in enumerate(results_with_scores, 1):
                source = doc.metadata.get("doc_name", "Unknown")
                page = doc.metadata.get("page", "?")
                print(f"{i}. Score: {score:.4f} | Source: {source} (page {page})")
                print(f"   Preview: {doc.page_content[:100]}...")
            print("=" * 50)
        else:
            documents = self.retrieve(query, top_k=top_k)
            scores = []

        context = self.format_context(documents)

        # Extract unique sources
        sources = list(set(doc.metadata.get("doc_name", "Unknown") for doc in documents))

        result = {
            "documents": documents,
            "context": context,
            "sources": sources,
            "num_results": len(documents),
        }

        if debug:
            result["debug_info"] = {
                "scores": scores,
                "query": query,
                "top_k": top_k or settings.top_k,
            }

        return result


# Example usage (for testing)
if __name__ == "__main__":
    # Initialize retriever
    # retriever = Retriever()

    # Test query
    # query = "How do I configure Kafka retention policies?"
    # results = retriever.retrieve_and_format(query, top_k=3)

    # print(f"Query: {query}")
    # print(f"\nFound {results['num_results']} relevant documents")
    # print(f"Sources: {', '.join(results['sources'])}")
    # print(f"\nFormatted context:\n{results['context']}")

    print("Retrieval module ready. Uncomment examples to test.")
