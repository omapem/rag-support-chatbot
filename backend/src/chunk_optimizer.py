"""
Chunking optimization utilities for RAG pipeline.
Provides tools to analyze and optimize chunking parameters.
"""

import statistics
from typing import List, Dict, Any, Tuple
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion import DocumentIngester


class ChunkOptimizer:
    """Analyzes documents and suggests optimal chunking parameters."""

    def __init__(self):
        """Initialize chunk optimizer."""
        self.analysis_results: Dict[str, Any] = {}

    def analyze_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Analyze document characteristics to inform chunking strategy.

        Args:
            documents: List of documents to analyze

        Returns:
            Dictionary with document statistics
        """
        if not documents:
            return {"error": "No documents provided"}

        # Document lengths
        doc_lengths = [len(doc.page_content) for doc in documents]

        # Content patterns
        has_code_blocks = sum(1 for doc in documents if "```" in doc.page_content or "kafka-" in doc.page_content)
        has_lists = sum(1 for doc in documents if any(marker in doc.page_content for marker in ["\n- ", "\n* ", "\n1. "]))
        has_tables = sum(1 for doc in documents if "|" in doc.page_content and "\n" in doc.page_content)

        # Paragraph structure
        avg_paragraphs_per_doc = statistics.mean(
            doc.page_content.count("\n\n") for doc in documents
        )

        analysis = {
            "total_documents": len(documents),
            "length_stats": {
                "min": min(doc_lengths),
                "max": max(doc_lengths),
                "mean": round(statistics.mean(doc_lengths), 2),
                "median": statistics.median(doc_lengths),
                "stdev": round(statistics.stdev(doc_lengths), 2) if len(doc_lengths) > 1 else 0
            },
            "content_patterns": {
                "docs_with_code": has_code_blocks,
                "docs_with_lists": has_lists,
                "docs_with_tables": has_tables,
                "avg_paragraphs_per_doc": round(avg_paragraphs_per_doc, 2)
            }
        }

        self.analysis_results = analysis
        return analysis

    def suggest_chunk_parameters(self, documents: List[Document] = None) -> Dict[str, Any]:
        """
        Suggest optimal chunking parameters based on document analysis.

        Args:
            documents: Optional documents to analyze (uses cached analysis if None)

        Returns:
            Dictionary with recommended chunking parameters
        """
        if documents:
            self.analyze_documents(documents)

        if not self.analysis_results:
            return {"error": "No analysis available. Provide documents first."}

        mean_length = self.analysis_results["length_stats"]["mean"]
        has_code = self.analysis_results["content_patterns"]["docs_with_code"] > 0

        # Determine chunk size based on content characteristics
        if mean_length < 500:
            # Short documents - use smaller chunks
            chunk_size = 400
            chunk_overlap = 50
            rationale = "Short documents detected - using smaller chunks to preserve context"
        elif mean_length > 5000:
            # Long documents - use larger chunks
            chunk_size = 1200
            chunk_overlap = 200
            rationale = "Long documents detected - using larger chunks for comprehensive context"
        elif has_code > 0:
            # Technical documents with code - moderate chunks with higher overlap
            chunk_size = 1000
            chunk_overlap = 150
            rationale = "Technical content with code blocks - balanced chunk size with higher overlap to preserve commands"
        else:
            # Standard documents
            chunk_size = 800
            chunk_overlap = 100
            rationale = "Standard document structure - default balanced chunking"

        recommendations = {
            "recommended_chunk_size": chunk_size,
            "recommended_overlap": chunk_overlap,
            "rationale": rationale,
            "additional_notes": self._generate_recommendations_notes()
        }

        return recommendations

    def _generate_recommendations_notes(self) -> List[str]:
        """Generate additional recommendations based on analysis."""
        notes = []

        patterns = self.analysis_results.get("content_patterns", {})

        if patterns.get("docs_with_code", 0) > 0:
            notes.append("Consider custom separators to preserve code blocks (e.g., '\\n#', '\\n$')")

        if patterns.get("docs_with_tables", 0) > 0:
            notes.append("Tables detected - ensure separator includes table row markers")

        if patterns.get("docs_with_lists", 0) > 0:
            notes.append("Lists detected - separator should preserve bullet points")

        length_stats = self.analysis_results.get("length_stats", {})
        if length_stats.get("stdev", 0) > 1000:
            notes.append("High length variance - consider document-specific chunking strategies")

        return notes

    def test_chunking_strategy(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int,
        separators: List[str] = None
    ) -> Dict[str, Any]:
        """
        Test a chunking strategy and return statistics.

        Args:
            documents: Documents to chunk
            chunk_size: Chunk size to test
            chunk_overlap: Overlap size to test
            separators: Custom separators (optional)

        Returns:
            Dictionary with chunking statistics
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators or ["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]

        # Check for split important content
        code_blocks_split = sum(
            1 for chunk in chunks
            if "```" in chunk.page_content and chunk.page_content.count("```") % 2 != 0
        )

        commands_split = sum(
            1 for chunk in chunks
            if any(cmd in chunk.page_content for cmd in ["kafka-topics", "kafka-console"])
            and not any(flag in chunk.page_content for flag in ["--create", "--delete", "--list"])
        )

        results = {
            "total_chunks": len(chunks),
            "chunk_size_stats": {
                "min": min(chunk_lengths),
                "max": max(chunk_lengths),
                "mean": round(statistics.mean(chunk_lengths), 2),
                "median": statistics.median(chunk_lengths),
                "target": chunk_size
            },
            "quality_indicators": {
                "chunks_at_target_size": sum(1 for length in chunk_lengths if abs(length - chunk_size) < 100),
                "incomplete_code_blocks": code_blocks_split,
                "incomplete_commands": commands_split
            },
            "effectiveness_score": self._calculate_effectiveness_score(
                chunks, chunk_size, code_blocks_split, commands_split
            )
        }

        return results

    def _calculate_effectiveness_score(
        self,
        chunks: List[Document],
        target_size: int,
        code_blocks_split: int,
        commands_split: int
    ) -> float:
        """
        Calculate effectiveness score for chunking strategy (0-100).

        Args:
            chunks: Generated chunks
            target_size: Target chunk size
            code_blocks_split: Number of incomplete code blocks
            commands_split: Number of incomplete commands

        Returns:
            Effectiveness score (0-100)
        """
        score = 100.0

        # Penalty for split code blocks (serious issue)
        score -= code_blocks_split * 10

        # Penalty for split commands (moderate issue)
        score -= commands_split * 5

        # Penalty for chunks far from target size
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        size_variance = statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
        if size_variance > target_size * 0.3:  # More than 30% variance
            score -= 10

        # Bonus for balanced chunk sizes
        median_length = statistics.median(chunk_lengths)
        if abs(median_length - target_size) < target_size * 0.1:  # Within 10%
            score += 10

        return max(0.0, min(100.0, score))

    def compare_strategies(
        self,
        documents: List[Document],
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple chunking strategies.

        Args:
            documents: Documents to test
            strategies: List of strategy configs with 'chunk_size', 'chunk_overlap', 'separators'

        Returns:
            Comparison results with best strategy recommendation
        """
        results = []

        for i, strategy in enumerate(strategies):
            test_result = self.test_chunking_strategy(
                documents,
                strategy.get("chunk_size", 800),
                strategy.get("chunk_overlap", 100),
                strategy.get("separators")
            )
            test_result["strategy_name"] = strategy.get("name", f"Strategy {i+1}")
            test_result["parameters"] = strategy
            results.append(test_result)

        # Find best strategy by effectiveness score
        best_strategy = max(results, key=lambda x: x["effectiveness_score"])

        return {
            "strategies_tested": len(strategies),
            "results": results,
            "best_strategy": best_strategy["strategy_name"],
            "best_score": best_strategy["effectiveness_score"],
            "recommendation": f"Use {best_strategy['strategy_name']} with effectiveness score {best_strategy['effectiveness_score']:.1f}/100"
        }


# Example usage
if __name__ == "__main__":
    print("Chunk Optimizer ready. Use with DocumentIngester to analyze chunking strategies.")

    # Example workflow:
    # 1. Load documents
    # ingester = DocumentIngester()
    # docs = ingester.load_pdf("data/raw/pdfs/kafka-guide.pdf")

    # 2. Analyze and get recommendations
    # optimizer = ChunkOptimizer()
    # analysis = optimizer.analyze_documents(docs)
    # recommendations = optimizer.suggest_chunk_parameters()
    # print(f"Recommended chunk size: {recommendations['recommended_chunk_size']}")

    # 3. Test and compare strategies
    # strategies = [
    #     {"name": "Small", "chunk_size": 500, "chunk_overlap": 50},
    #     {"name": "Medium", "chunk_size": 800, "chunk_overlap": 100},
    #     {"name": "Large", "chunk_size": 1200, "chunk_overlap": 200}
    # ]
    # comparison = optimizer.compare_strategies(docs, strategies)
    # print(f"Best strategy: {comparison['best_strategy']}")
