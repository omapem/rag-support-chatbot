"""
Evaluation script for RAG system.
Runs test queries and generates comprehensive quality metrics.
"""

import os
import sys
import time
from pathlib import Path

# Disable tokenizers parallelism warning (safe for this script)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import RAGEvaluator
from src.retrieval import Retriever
from src.generation import ResponseGenerator
from src.config import settings


def run_evaluation_suite(test_file: str = None, debug: bool = True):
    """
    Run complete evaluation suite on test queries.

    Args:
        test_file: Path to test queries JSON file (optional)
        debug: Enable debug output
    """
    print("=" * 60)
    print("RAG System Evaluation Suite")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing RAG components...")
    evaluator = RAGEvaluator()
    retriever = Retriever(enable_hybrid=True)
    generator = ResponseGenerator()

    # Load test queries
    print("\n2. Loading test queries...")
    test_queries = evaluator.load_test_queries(test_file)
    print(f"   Loaded {len(test_queries)} test queries")

    # Run evaluations
    print("\n3. Running evaluations...")
    results = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        expected_concepts = test_case.get("expected_concepts", [])
        relevant_docs = test_case.get("relevant_docs", [])
        category = test_case.get("category", "general")

        print(f"\n   [{i}/{len(test_queries)}] Category: {category}")
        print(f"   Query: {query}")

        # Retrieval
        retrieval_start = time.time()
        retrieved_results = retriever.retrieve_with_scores(query, apply_reranking=True)
        retrieval_time = time.time() - retrieval_start

        retrieved_docs = [doc for doc, score in retrieved_results]
        scores = [score for doc, score in retrieved_results]

        if debug:
            print(f"   Retrieved {len(retrieved_docs)} docs in {retrieval_time*1000:.2f}ms")
            print(f"   Avg relevance: {sum(scores)/len(scores):.4f}")

        # Generation (using ResponseGenerator which retrieves internally)
        generation_start = time.time()
        response_data = generator.generate_response(query, top_k=len(retrieved_docs))
        generation_time = time.time() - generation_start

        response = response_data["answer"]
        sources = response_data["sources"]

        if debug:
            print(f"   Generated response in {generation_time*1000:.2f}ms")
            print(f"   Sources: {', '.join(sources)}")

        # Calculate retrieval metrics
        precision = evaluator.calculate_precision_at_k(retrieved_docs, relevant_docs, k=4)
        recall = evaluator.calculate_recall_at_k(retrieved_docs, relevant_docs, k=4)
        mrr = evaluator.calculate_mrr(retrieved_docs, relevant_docs)

        if debug and relevant_docs:
            print(f"   Precision@4: {precision:.2f}, Recall@4: {recall:.2f}, MRR: {mrr:.2f}")

        # Check if expected concepts are in response
        concepts_found = sum(1 for concept in expected_concepts if concept.lower() in response.lower())
        concept_coverage = concepts_found / len(expected_concepts) if expected_concepts else None

        if debug and expected_concepts:
            print(f"   Concept coverage: {concepts_found}/{len(expected_concepts)} ({concept_coverage*100:.1f}%)")

        # Save evaluation result
        coverage_str = f"{concept_coverage:.2f}" if concept_coverage is not None else "0.00"
        eval_result = evaluator.evaluate_end_to_end(
            query=query,
            retrieved_docs=retrieved_docs,
            relevance_scores=scores,
            response=response,
            sources=sources,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            notes=f"Category: {category}, Precision@4: {precision:.2f}, Concept coverage: {coverage_str}"
        )

        results.append({
            "query": query,
            "category": category,
            "precision": precision,
            "recall": recall,
            "mrr": mrr,
            "concept_coverage": concept_coverage,
            "retrieval_time_ms": retrieval_time * 1000,
            "generation_time_ms": generation_time * 1000
        })

    # Generate summary report
    print("\n4. Generating metrics report...")
    report = evaluator.generate_metrics_report()

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nQueries Evaluated: {report['summary']['total_queries_evaluated']}")

    print("\nRetrieval Performance:")
    retrieval = report['retrieval_metrics']
    print(f"  Avg Documents Retrieved: {retrieval['avg_docs_retrieved']}")
    print(f"  Avg Relevance Score: {retrieval['avg_relevance_score']:.4f}")
    print(f"  Avg Retrieval Time: {retrieval['avg_retrieval_time_ms']:.2f}ms")

    print("\nGeneration Performance:")
    generation = report['generation_metrics']
    print(f"  Avg Generation Time: {generation['avg_generation_time_ms']:.2f}ms")
    print(f"  Avg Total Time: {generation['avg_total_time_ms']:.2f}ms")

    # Category-specific metrics
    print("\nMetrics by Category:")
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result)

    for cat, cat_results in categories.items():
        avg_precision = sum(r['precision'] for r in cat_results) / len(cat_results)
        avg_recall = sum(r['recall'] for r in cat_results) / len(cat_results)
        avg_mrr = sum(r['mrr'] for r in cat_results) / len(cat_results)
        avg_time = sum(r['retrieval_time_ms'] + r['generation_time_ms'] for r in cat_results) / len(cat_results)

        print(f"\n  {cat.upper()}:")
        print(f"    Precision@4: {avg_precision:.2f}")
        print(f"    Recall@4: {avg_recall:.2f}")
        print(f"    MRR: {avg_mrr:.2f}")
        print(f"    Avg Total Time: {avg_time:.2f}ms")

    print("\n" + "=" * 60)
    print(f"Results saved to: {evaluator.results_file}")
    print(f"Report saved to: {evaluator.metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to test queries JSON file (optional)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable debug output"
    )

    args = parser.parse_args()

    try:
        run_evaluation_suite(
            test_file=args.test_file,
            debug=not args.quiet
        )
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
