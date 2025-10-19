"""
Evaluation metrics for RAG system quality assessment.
Tracks retrieval accuracy, response quality, and system performance.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from langchain_core.documents import Document

from src.config import settings


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality assessment."""
    query: str
    num_retrieved: int
    avg_score: float
    top_score: float
    sources: List[str]
    retrieval_time_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResponseMetrics:
    """Metrics for response generation quality."""
    query: str
    response_length: int
    num_sources_cited: int
    generation_time_ms: float
    total_time_ms: float
    model: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a query."""
    query: str
    retrieved_docs: int
    response: str
    sources: List[str]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    relevance_scores: List[float]
    avg_relevance: float
    timestamp: str

    # Optional ground truth comparison
    expected_answer: Optional[str] = None
    human_rating: Optional[int] = None  # 1-5 scale
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RAGEvaluator:
    """Evaluates RAG system performance across multiple dimensions."""

    def __init__(self, eval_data_dir: str = "./data/eval"):
        """
        Initialize evaluator with data directory.

        Args:
            eval_data_dir: Directory to store evaluation results
        """
        self.eval_data_dir = Path(eval_data_dir)
        self.eval_data_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.eval_data_dir / "evaluation_results.jsonl"
        self.metrics_file = self.eval_data_dir / "metrics_summary.json"

    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Document],
        scores: List[float],
        retrieval_time: float
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality for a single query.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            scores: Relevance scores for each document
            retrieval_time: Time taken for retrieval in seconds

        Returns:
            RetrievalMetrics object
        """
        sources = list(set(doc.metadata.get("doc_name", "Unknown") for doc in retrieved_docs))

        metrics = RetrievalMetrics(
            query=query,
            num_retrieved=len(retrieved_docs),
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            top_score=max(scores) if scores else 0.0,
            sources=sources,
            retrieval_time_ms=retrieval_time * 1000,
            timestamp=datetime.now().isoformat()
        )

        return metrics

    def evaluate_response(
        self,
        query: str,
        response: str,
        sources_cited: List[str],
        generation_time: float,
        total_time: float,
        model: str = None
    ) -> ResponseMetrics:
        """
        Evaluate response generation quality.

        Args:
            query: User query
            response: Generated response
            sources_cited: List of sources cited in response
            generation_time: Time for LLM generation in seconds
            total_time: Total end-to-end time in seconds
            model: LLM model name

        Returns:
            ResponseMetrics object
        """
        metrics = ResponseMetrics(
            query=query,
            response_length=len(response),
            num_sources_cited=len(sources_cited),
            generation_time_ms=generation_time * 1000,
            total_time_ms=total_time * 1000,
            model=model or settings.llm_model,
            timestamp=datetime.now().isoformat()
        )

        return metrics

    def evaluate_end_to_end(
        self,
        query: str,
        retrieved_docs: List[Document],
        relevance_scores: List[float],
        response: str,
        sources: List[str],
        retrieval_time: float,
        generation_time: float,
        expected_answer: Optional[str] = None,
        human_rating: Optional[int] = None,
        notes: Optional[str] = None
    ) -> EvaluationResult:
        """
        Complete end-to-end evaluation of a RAG interaction.

        Args:
            query: User query
            retrieved_docs: Retrieved documents
            relevance_scores: Document relevance scores
            response: Generated response
            sources: Source documents used
            retrieval_time: Retrieval time in seconds
            generation_time: Generation time in seconds
            expected_answer: Optional ground truth answer
            human_rating: Optional 1-5 rating
            notes: Optional evaluation notes

        Returns:
            EvaluationResult object
        """
        total_time = retrieval_time + generation_time
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0

        result = EvaluationResult(
            query=query,
            retrieved_docs=len(retrieved_docs),
            response=response,
            sources=sources,
            retrieval_time_ms=retrieval_time * 1000,
            generation_time_ms=generation_time * 1000,
            total_time_ms=total_time * 1000,
            relevance_scores=relevance_scores,
            avg_relevance=avg_relevance,
            timestamp=datetime.now().isoformat(),
            expected_answer=expected_answer,
            human_rating=human_rating,
            notes=notes
        )

        # Save result
        self._save_result(result)

        return result

    def _save_result(self, result: EvaluationResult) -> None:
        """Save evaluation result to JSONL file."""
        with open(self.results_file, 'a') as f:
            json.dump(result.to_dict(), f)
            f.write('\n')

    def calculate_precision_at_k(
        self,
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = None
    ) -> float:
        """
        Calculate Precision@K - fraction of top-k retrieved docs that are relevant.

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of IDs for documents known to be relevant
            k: Number of top documents to consider (defaults to all)

        Returns:
            Precision@K score (0.0 to 1.0)
        """
        k = k or len(retrieved_docs)
        top_k_docs = retrieved_docs[:k]

        relevant_count = sum(
            1 for doc in top_k_docs
            if doc.metadata.get('doc_name') in relevant_doc_ids
        )

        return relevant_count / k if k > 0 else 0.0

    def calculate_recall_at_k(
        self,
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str],
        k: int = None
    ) -> float:
        """
        Calculate Recall@K - fraction of relevant docs found in top-k results.

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of IDs for documents known to be relevant
            k: Number of top documents to consider

        Returns:
            Recall@K score (0.0 to 1.0)
        """
        k = k or len(retrieved_docs)
        top_k_docs = retrieved_docs[:k]

        found_relevant = sum(
            1 for doc in top_k_docs
            if doc.metadata.get('doc_name') in relevant_doc_ids
        )

        total_relevant = len(relevant_doc_ids)
        return found_relevant / total_relevant if total_relevant > 0 else 0.0

    def calculate_mrr(
        self,
        retrieved_docs: List[Document],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank - position of first relevant document.

        Args:
            retrieved_docs: Retrieved documents
            relevant_doc_ids: List of IDs for documents known to be relevant

        Returns:
            MRR score (0.0 to 1.0)
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc.metadata.get('doc_name') in relevant_doc_ids:
                return 1.0 / i
        return 0.0

    def generate_metrics_report(self) -> Dict[str, Any]:
        """
        Generate summary report from all evaluation results.

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.results_file.exists():
            return {"error": "No evaluation results found"}

        results = []
        with open(self.results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))

        if not results:
            return {"error": "No evaluation results found"}

        # Aggregate metrics
        total_queries = len(results)
        avg_retrieval_time = sum(r['retrieval_time_ms'] for r in results) / total_queries
        avg_generation_time = sum(r['generation_time_ms'] for r in results) / total_queries
        avg_total_time = sum(r['total_time_ms'] for r in results) / total_queries
        avg_relevance = sum(r['avg_relevance'] for r in results) / total_queries
        avg_docs_retrieved = sum(r['retrieved_docs'] for r in results) / total_queries

        # Human ratings (if available)
        rated_results = [r for r in results if r.get('human_rating')]
        avg_human_rating = sum(r['human_rating'] for r in rated_results) / len(rated_results) if rated_results else None

        report = {
            "summary": {
                "total_queries_evaluated": total_queries,
                "evaluation_period": {
                    "first": results[0]['timestamp'],
                    "last": results[-1]['timestamp']
                }
            },
            "retrieval_metrics": {
                "avg_docs_retrieved": round(avg_docs_retrieved, 2),
                "avg_relevance_score": round(avg_relevance, 4),
                "avg_retrieval_time_ms": round(avg_retrieval_time, 2)
            },
            "generation_metrics": {
                "avg_generation_time_ms": round(avg_generation_time, 2),
                "avg_total_time_ms": round(avg_total_time, 2)
            },
            "quality_metrics": {
                "avg_human_rating": round(avg_human_rating, 2) if avg_human_rating else None,
                "num_human_ratings": len(rated_results)
            }
        }

        # Save report
        with open(self.metrics_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def load_test_queries(self, test_file: str = None) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        Args:
            test_file: Path to test queries file

        Returns:
            List of test query dictionaries
        """
        test_file = test_file or str(self.eval_data_dir / "test_queries.json")
        test_path = Path(test_file)

        if not test_path.exists():
            # Create example test file
            example_queries = [
                {
                    "query": "How do I create a new Kafka topic?",
                    "expected_concepts": ["kafka-topics.sh", "--create", "partition", "replication-factor"],
                    "relevant_docs": ["kafka-guide.pdf"],
                    "category": "operations"
                },
                {
                    "query": "What are the key producer configuration properties?",
                    "expected_concepts": ["acks", "batch.size", "linger.ms", "buffer.memory"],
                    "relevant_docs": ["kafka-guide.pdf"],
                    "category": "development"
                },
                {
                    "query": "How do I monitor Kafka cluster health?",
                    "expected_concepts": ["JMX", "metrics", "broker", "under-replicated"],
                    "relevant_docs": ["kafka-guide.pdf"],
                    "category": "monitoring"
                }
            ]

            with open(test_path, 'w') as f:
                json.dump(example_queries, f, indent=2)

            return example_queries

        with open(test_path, 'r') as f:
            return json.load(f)


# Example usage
if __name__ == "__main__":
    evaluator = RAGEvaluator()

    # Generate report from existing results
    report = evaluator.generate_metrics_report()
    print("Evaluation Report:")
    print(json.dumps(report, indent=2))
