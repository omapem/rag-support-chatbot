"""
Simple Streamlit UI for testing the RAG chatbot (Week 1).
This provides a basic chat interface to test the RAG pipeline.
"""

import streamlit as st
from src.generation import ResponseGenerator
from src.config import settings

# Page configuration
st.set_page_config(
    page_title="Kafka Support Chatbot",
    page_icon="☕",
    layout="centered",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "generator" not in st.session_state:
    st.session_state.generator = ResponseGenerator()


# App title and description
st.title("☕ Apache Kafka Support Chatbot")
st.markdown(
    """
    Ask me anything about Apache Kafka! I'll search through the documentation
    to find relevant information and provide accurate answers with citations.
    """
)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot uses **Retrieval-Augmented Generation (RAG)** to answer
        questions about Apache Kafka.

        **Tech Stack:**
        - 🤖 Claude 3.5 Sonnet (LLM)
        - 🔍 Sentence Transformers (embeddings)
        - 📚 Chroma (vector database)
        - ⚡ LangChain (RAG framework)
        """
    )

    st.divider()

    st.header("Settings")
    top_k = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=settings.top_k)
    st.caption(f"Default: {settings.top_k} sources • More sources = better coverage but slower")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("**Sample Questions:**")
    sample_questions = [
        "What is Apache Kafka?",
        "How do I create a topic?",
        "How do I configure retention policies?",
        "What is a consumer group?",
        "How do I monitor Kafka performance?",
    ]

    for question in sample_questions:
        if st.button(question, key=f"sample_{question}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()


# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display sources if available
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander(f"📚 Sources ({len(message['sources'])})"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")


# Chat input
if prompt := st.chat_input("Ask a question about Kafka..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            try:
                response = st.session_state.generator.generate_response(
                    prompt, top_k=top_k
                )

                # Display answer
                st.markdown(response["answer"])

                # Display sources
                if response["sources"]:
                    with st.expander(f"📚 Sources ({len(response['sources'])})"):
                        for source in response["sources"]:
                            st.markdown(f"- {source}")

                # Add assistant message to chat
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                    }
                )

            except Exception as e:
                error_message = f"Error: {str(e)}\n\nMake sure you have:\n1. Set up your .env file with ANTHROPIC_API_KEY\n2. Created the vector database (run ingestion first)"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# Footer
st.divider()
st.caption("Built with LangChain, Claude, and Streamlit • RAG Support Chatbot Demo")
