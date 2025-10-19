"""
LLM response generation module.
Handles calling Claude API and generating responses based on retrieved context.
"""

from typing import Dict, Any, Optional
from anthropic import Anthropic

from src.config import settings
from src.retrieval import Retriever
from src.prompts import (
    SYSTEM_PROMPT,
    format_rag_prompt,
    format_fallback_prompt,
    format_conversation_prompt,
)


class ResponseGenerator:
    """Generates responses using Claude API and RAG pipeline."""

    def __init__(self):
        """Initialize the response generator."""
        self.client = Anthropic(api_key=settings.anthropic_api_key)
        self.retriever = Retriever()
        self.model = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.max_tokens

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        top_k: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using RAG.

        Args:
            query: User's question
            conversation_history: Optional conversation history
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with 'answer', 'sources', 'context', and metadata
        """
        # Step 1: Retrieve relevant documents
        retrieval_results = self.retriever.retrieve_and_format(query, top_k=top_k)

        # Step 2: Check if we found relevant context
        if retrieval_results["num_results"] == 0:
            # No relevant documents found - use fallback
            prompt = format_fallback_prompt(query)
            answer = self._call_claude(prompt)

            return {
                "answer": answer,
                "sources": [],
                "context": "",
                "num_sources": 0,
                "has_context": False,
            }

        # Step 3: Format prompt with context
        if conversation_history:
            prompt = format_conversation_prompt(
                query, retrieval_results["context"], conversation_history
            )
        else:
            prompt = format_rag_prompt(query, retrieval_results["context"])

        # Step 4: Generate response
        answer = self._call_claude(prompt)

        return {
            "answer": answer,
            "sources": retrieval_results["sources"],
            "context": retrieval_results["context"],
            "num_sources": retrieval_results["num_results"],
            "has_context": True,
        }

    def _call_claude(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Call Claude API with the given prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt (defaults to SYSTEM_PROMPT)

        Returns:
            Generated response text
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )

            return message.content[0].text

        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return f"I apologize, but I encountered an error generating a response. Please try again. Error: {str(e)}"

    def stream_response(self, query: str, top_k: int = None):
        """
        Generate a streaming response (for future use with UI).

        Args:
            query: User's question
            top_k: Number of documents to retrieve

        Yields:
            Response chunks as they're generated
        """
        # Retrieve context
        retrieval_results = self.retriever.retrieve_and_format(query, top_k=top_k)

        if retrieval_results["num_results"] == 0:
            prompt = format_fallback_prompt(query)
        else:
            prompt = format_rag_prompt(query, retrieval_results["context"])

        # Stream response
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            yield f"Error: {str(e)}"


# Example usage (for testing)
if __name__ == "__main__":
    # Initialize generator
    # generator = ResponseGenerator()

    # Test query
    # query = "How do I create a Kafka topic?"
    # response = generator.generate_response(query)

    # print(f"Query: {query}\n")
    # print(f"Answer: {response['answer']}\n")
    # print(f"Sources ({response['num_sources']}): {', '.join(response['sources'])}")

    # Test streaming
    # print("\nStreaming response:")
    # for chunk in generator.stream_response(query):
    #     print(chunk, end="", flush=True)

    print("Generation module ready. Uncomment examples to test.")
