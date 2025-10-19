# RAG Support Chatbot - Backend

Apache Kafka support chatbot using Retrieval-Augmented Generation (RAG).

## Quick Start

### 1. Initial Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 2. Prepare Your Knowledge Base

```bash
# Place your Kafka PDF in data/raw/pdfs/
mkdir -p data/raw/pdfs
# Copy your PDF here

# Or add Kafka documentation URLs to process
```

### 3. Create Vector Database (One-time setup)

```python
# Run this in a Python shell or create a script
from src.ingestion import DocumentIngester
from src.embeddings import EmbeddingManager

# Load and chunk documents
ingester = DocumentIngester()
chunks = ingester.process_pdf_directory("data/raw/pdfs")

# Create embeddings and vector store
embedding_manager = EmbeddingManager()
vector_store = embedding_manager.create_vector_store(chunks)

print(f"✅ Created vector database with {len(chunks)} chunks")
```

### 4. Test the RAG Pipeline

```bash
# Run the Streamlit UI
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` and start asking questions!

## Project Structure

```
backend/
├── src/                      # Core RAG modules
│   ├── config.py            # Configuration management
│   ├── ingestion.py         # Document loading & chunking
│   ├── embeddings.py        # Embedding generation & vector DB
│   ├── retrieval.py         # Semantic search
│   ├── generation.py        # LLM response generation
│   └── prompts.py           # Prompt templates
├── app/                      # FastAPI app (Week 2)
├── tests/                    # Unit tests
├── data/                     # Data storage
│   ├── raw/                 # Original documents
│   ├── processed/           # Chunked data
│   └── chroma_db/           # Vector database
├── streamlit_app.py         # Simple UI for testing
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (don't commit!)
```

## Module Overview

### `src/config.py`
- Loads environment variables
- Provides application settings
- Configuration for RAG parameters

### `src/ingestion.py`
- Load PDFs with `PyPDFLoader`
- Scrape web pages with `WebBaseLoader`
- Chunk documents with `RecursiveCharacterTextSplitter`
- Add metadata to chunks

### `src/embeddings.py`
- Generate embeddings with Sentence Transformers (free!)
- Manage Chroma vector database
- Persist and load vector stores
- Create retrievers for search

### `src/retrieval.py`
- Semantic search over vector database
- Format retrieved context for LLM
- Manage top-k retrieval

### `src/generation.py`
- Call Claude API for response generation
- Implement RAG pipeline (retrieve → generate)
- Handle conversation history
- Streaming support

### `src/prompts.py`
- System prompts defining chatbot persona
- RAG prompt templates
- Fallback prompts
- Conversation prompts

## Common Tasks

### Add More Documents

```python
from src.ingestion import DocumentIngester
from src.embeddings import EmbeddingManager

# Process new PDFs
ingester = DocumentIngester()
new_chunks = ingester.process_pdf_directory("data/raw/pdfs")

# Add to existing vector store
embedding_manager = EmbeddingManager()
embedding_manager.load_vector_store()
embedding_manager.add_documents(new_chunks)
```

### Test Retrieval Quality

```python
from src.retrieval import Retriever

retriever = Retriever()
results = retriever.retrieve_with_scores("How do I create a topic?", top_k=5)

for doc, score in results:
    print(f"Score: {score:.3f} | Source: {doc.metadata['doc_name']}")
    print(f"Content: {doc.page_content[:200]}...\n")
```

### Test Response Generation

```python
from src.generation import ResponseGenerator

generator = ResponseGenerator()
response = generator.generate_response("What is Apache Kafka?")

print(f"Answer: {response['answer']}\n")
print(f"Sources: {', '.join(response['sources'])}")
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_retrieval.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

## Configuration

Key settings in `.env`:

```bash
# Required
ANTHROPIC_API_KEY=your_key_here

# RAG Configuration
CHUNK_SIZE=800              # Size of text chunks
CHUNK_OVERLAP=100           # Overlap between chunks
TOP_K=4                     # Number of documents to retrieve
SIMILARITY_THRESHOLD=0.7    # Minimum similarity score

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=claude-3-5-sonnet-20241022
LLM_TEMPERATURE=0.3
MAX_TOKENS=1024
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Solution**: Run commands from the `backend/` directory

**Issue**: `ValueError: Vector store not initialized`
- **Solution**: Create the vector database first (see step 3 above)

**Issue**: `anthropic.AuthenticationError`
- **Solution**: Check your `ANTHROPIC_API_KEY` in `.env`

**Issue**: Slow embedding generation
- **Solution**: First time loads the model (~100MB). Subsequent runs are faster.

## Next Steps (Week 2)

- Build FastAPI backend in `app/`
- Add conversation memory
- Implement rate limiting
- Write comprehensive tests
- Deploy to Railway/Render

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
