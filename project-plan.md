# Customer Support Chatbot with RAG - Project Plan

Great choice for a learning project! Building a RAG-based chatbot will teach you essential AI engineering skills. Here's a comprehensive plan to guide you.

## Project Scope

**Core Objectives:**

- Build a chatbot that can answer customer questions using your own knowledge base
- Knowledge base will be based on the Apache Kafka Ecosystem. This will consist of Kafka cluster maintenance, development and best practices
- Implement RAG (Retrieval-Augmented Generation) to ground responses in actual documentation
- Apply prompt engineering techniques to control chatbot behavior and output quality
- Create a clean, modern web interface for testing and demonstration

**Key Features:**

- Document ingestion system (PDFs, text files, web pages)
- Vector search to find relevant information
- LLM integration for natural language responses
- Conversation history tracking
- Simple UI for interaction

**Learning Outcomes:**

- Understanding embeddings and vector databases
- Working with LLM APIs
- Implementing semantic search
- Prompt engineering best practices
- Building end-to-end AI applications

## Tech Stack

**LLM & APIs:**

- **OpenAI API** (GPT-4 or GPT-3.5-turbo) - easiest to start with
- Alternative: Anthropic Claude API, or open-source models via Hugging Face

**Vector Database:**

- **Pinecone** (managed, easy setup) or **Chroma** (local, free, great for learning)
- Alternative: Weaviate, Qdrant

**Embeddings:**

- OpenAI embeddings (`text-embedding-ada-002`)
- Alternative: Sentence Transformers (open-source)

**Backend:**

- **Python 3.12+**
- **LangChain** - framework for building LLM applications
- **FastAPI** - for creating REST API endpoints
- **Pydantic** - for any agents or validation

**Frontend:**

- **Next.js**
- **Tailwind CSS**
- **ShadCN**
- **Framer Motion**
- **GSAP**

**Additional Tools:**

- **PyPDF2** or **pypdf** for PDF processing
- **BeautifulSoup4** for web scraping
- **python-dotenv** for environment variables
- **tiktoken** for token counting

## Implementation Steps

### Phase 1: Setup & Environment (Week 1) ✅ COMPLETE

1. **Environment Setup** ✅

   - ✅ Create a Python virtual environment
   - ✅ Install core dependencies: langchain, anthropic, chromadb, sentence-transformers
   - ✅ Set up API keys (Anthropic Claude API)
   - ✅ Create project structure with backend/ directory organization

2. **Learn the Basics** ✅
   - ✅ Read LangChain documentation on RAG
   - ✅ Understand how embeddings work
   - ✅ Test Claude API calls

### Phase 2: Data Ingestion Pipeline (Week 1-2) ✅ COMPLETE (Week 1)

3. **Prepare Knowledge Base** ✅

   - ✅ Set up data/ directory structure (raw/pdfs/, processed/, chroma_db/)
   - ✅ Ready for Kafka documentation PDFs

4. **Build Document Loader** ✅

   - ✅ PDF loading with PyPDF2 (`ingestion.py`)
   - ✅ Text chunking (800 tokens per chunk with 100 token overlap)
   - ✅ Metadata tracking (source, page numbers, document type)

5. **Create Embeddings** ✅
   - ✅ Convert text chunks to vector embeddings using Sentence Transformers
   - ✅ Store embeddings in Chroma vector database
   - ✅ Persistence to disk (`data/chroma_db/`)
   - ✅ Test similarity search with sample queries

### Phase 3: RAG Implementation (Week 1-2) ✅ COMPLETE (Week 1)

6. **Build Retrieval System** ✅

   - ✅ Implement semantic search function (`retrieval.py`)
   - ✅ Configurable retrieval parameters (top-k=4, similarity threshold)
   - ✅ BONUS: Hybrid search implementation (`hybrid_retrieval.py` - semantic + BM25)

7. **Prompt Engineering** ✅

   - ✅ Design system prompts for Kafka support specialist persona (`prompts.py`)
   - ✅ Create RAG prompt templates with retrieved context injection
   - ✅ Implement guardrails (handling off-topic queries, missing information)

8. **Response Generation** ✅
   - ✅ Integrate Claude API with retrieval system (`generation.py`)
   - ✅ Implement citation/source referencing
   - ✅ Handle cases where information isn't found
   - ✅ Error handling and graceful degradation

### Phase 4: Application Development - Week 1 ✅ COMPLETE | Week 2-3 IN PROGRESS

9. **Build Backend API** (Week 2)

   - Create FastAPI endpoints for chat
   - Implement conversation history storage
   - Add rate limiting and error handling

10. **Develop Frontend** ✅ Week 1 Complete (Streamlit)

    - ✅ Build simple chat interface with Streamlit (`streamlit_app.py`)
    - ✅ Display sources/citations for answers
    - Week 3: Next.js professional frontend

11. **Testing & Refinement** ✅ Week 1 Complete
    - ✅ Test with various queries
    - ✅ Measure response quality and relevance
    - ✅ Iterate on prompts and retrieval parameters

### Phase 5: Enhancement & Deployment (Week 4+)

12. **Advanced Features** (Optional)

    - Add conversation memory
    - Implement query classification (greeting, question, complaint)
    - Add multi-turn conversation handling
    - Include fallback responses

13. **Evaluation**

    - Create test question sets
    - Measure accuracy and relevance
    - Track retrieval quality metrics

14. **Documentation**
    - Write README with setup instructions
    - Document your prompt engineering decisions
    - Create architecture diagram

## Sample Project Structure

```
customer-support-chatbot/
├── data/
│   ├── raw/              # Original documents
│   └── processed/        # Chunked and preprocessed
├── src/
│   ├── ingestion.py      # Document loading and chunking
│   ├── embeddings.py     # Embedding generation
│   ├── retrieval.py      # Vector search
│   ├── generation.py     # LLM response generation
│   └── prompts.py        # Prompt templates
├── app/
│   ├── api.py            # FastAPI backend
│   └── streamlit_app.py  # Frontend
├── tests/
├── notebooks/            # Experimentation
├── .env                  # API keys
├── requirements.txt
└── README.md
```

## Key Learning Checkpoints

- **After Phase 2:** You should understand embeddings and vector similarity
- **After Phase 3:** You should grasp RAG architecture and prompt engineering
- **After Phase 4:** You should have a working end-to-end application
- **After Phase 5:** You should be able to explain design tradeoffs and evaluate performance
