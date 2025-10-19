# Apache Kafka RAG Support Chatbot

A production-ready chatbot using Retrieval-Augmented Generation (RAG) to answer questions about Apache Kafka. Built for learning AI engineering concepts and demonstrating RAG implementation.

## 🎯 Project Goals

- **Learning**: Master RAG, embeddings, vector databases, and prompt engineering
- **Portfolio**: Showcase AI engineering skills with a deployed demo
- **Cost-Effective**: Use free/cheap alternatives (Sentence Transformers, Claude API)
- **Production-Ready**: Cloud deployment with professional frontend

## 🚀 Quick Start

**Get running in 10 minutes:**

```bash
# 1. Initialize
chmod +x init_backend.sh && ./init_backend.sh

# 2. Setup environment
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: Add ANTHROPIC_API_KEY=your_key

# 4. Add your Kafka PDF
cp ~/your-kafka-doc.pdf data/raw/pdfs/

# 5. Create vector database
python scripts/create_vectordb.py

# 6. Run chatbot
streamlit run streamlit_app.py
```

Open `http://localhost:8501` and start asking Kafka questions!

## 📚 Documentation

### Getting Started
- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Complete setup walkthrough with detailed explanations
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Commands and code snippets cheat sheet
- **[INITIALIZATION_SUMMARY.md](INITIALIZATION_SUMMARY.md)** - What was created and why

### Technical Details
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture, data flow, and design decisions
- **[backend/README.md](backend/README.md)** - Module documentation and API reference
- **[CLAUDE.md](CLAUDE.md)** - Project overview for Claude Code sessions

### Planning
- **[project-plan.md](project-plan.md)** - Original 4-week development plan

## 🏗️ Tech Stack

### Backend
- **Python 3.12+** - Core language
- **LangChain** - RAG framework
- **Claude API** - LLM (cost-effective, excellent for technical docs)
- **Sentence Transformers** - Embeddings (FREE!)
- **Chroma** - Vector database (local dev)
- **Pinecone** - Vector database (production)
- **FastAPI** - REST API (Week 2)

### Frontend
- **Next.js 14** - React framework with App Router
- **Tailwind CSS** - Styling
- **ShadCN** - UI components
- **Framer Motion** - Animations

## 📁 Project Structure

```
rag-support-chatbot/
├── backend/
│   ├── src/              # Core RAG modules (Week 1)
│   │   ├── config.py     # Configuration management
│   │   ├── prompts.py    # Prompt templates
│   │   ├── ingestion.py  # Document loading & chunking
│   │   ├── embeddings.py # Vector database operations
│   │   ├── retrieval.py  # Semantic search
│   │   └── generation.py # LLM response generation
│   ├── app/              # FastAPI backend (Week 2)
│   ├── tests/            # Unit tests
│   ├── data/             # Document storage
│   ├── scripts/          # Utility scripts
│   └── streamlit_app.py  # Test UI
├── frontend/             # Next.js app (Week 3)
└── docs/                 # Documentation
```

## 🎓 Learning Path (4 Weeks)

### Week 1: RAG Core ✅
- ✅ Environment setup
- ✅ Document ingestion
- ✅ Vector database
- ✅ Retrieval system
- ✅ LLM integration
- ✅ Streamlit UI

### Week 2: Backend API
- FastAPI endpoints
- Conversation memory
- Rate limiting
- Testing
- Documentation

### Week 3: Frontend
- Next.js setup
- Chat interface
- Citation display
- Animations
- Responsive design

### Week 4: Deployment
- Pinecone migration
- Backend deployment (Railway/Render)
- Frontend deployment (Vercel)
- Documentation
- Demo preparation

## 🔑 Key Features

### Current (Week 1)
- ✅ PDF document processing
- ✅ Web page scraping
- ✅ Semantic search with Chroma
- ✅ Claude-powered responses
- ✅ Source citation
- ✅ Simple chat interface

### Planned
- Conversation history tracking
- Multi-turn context awareness
- Hybrid search (semantic + keyword)
- Streaming responses
- Response quality metrics
- Production deployment

## 🧪 Development Workflow

### Testing Individual Components

```python
# Test document loading
from src.ingestion import DocumentIngester
ingester = DocumentIngester()
chunks = ingester.process_pdf_directory("data/raw/pdfs")

# Test retrieval
from src.retrieval import Retriever
retriever = Retriever()
results = retriever.retrieve("How do I create a topic?", top_k=3)

# Test generation
from src.generation import ResponseGenerator
generator = ResponseGenerator()
response = generator.generate_response("What is Kafka?")
```

### Running Tests
```bash
pytest                              # Run all tests
pytest --cov=src --cov-report=html  # With coverage
pytest tests/test_retrieval.py -v   # Specific test file
```

### Code Quality
```bash
black src/ tests/        # Format code
flake8 src/ tests/       # Lint
mypy src/                # Type check
```

## 📊 Performance Targets

- **Response Time**: < 5 seconds (target: 2-3 seconds)
- **Retrieval Quality**: High relevance for Kafka queries
- **Citation Accuracy**: 100% (sources must match content)
- **Uptime**: 99%+ for deployed demo

## 💰 Cost Estimates

### Development (FREE)
- Chroma: Free (local)
- Sentence Transformers: Free
- Streamlit: Free

### Production (~$10-15/month)
- Claude API: ~$5-10/month (light usage)
- Pinecone: Free tier (1M vectors)
- Vercel: Free tier
- Railway/Render: Free tier

## 🎯 Success Criteria

### Functional
- ✅ Answers Kafka questions accurately
- ✅ Cites sources for all answers
- ✅ Handles multi-turn conversations
- ✅ Responds within 5 seconds
- ✅ Graceful handling of unknown topics

### Portfolio
- Professional, modern UI
- Clear RAG architecture demonstration
- Comprehensive documentation
- Live demo with public URL
- Clean GitHub repository

## 🛠️ Common Tasks

### Add New Documents
```bash
# Add PDFs to data/raw/pdfs/
cp ~/new-kafka-doc.pdf backend/data/raw/pdfs/

# Recreate vector database
cd backend
python scripts/create_vectordb.py
```

### Update Configuration
```bash
# Edit .env file
nano backend/.env

# Available settings:
# - CHUNK_SIZE (default: 800)
# - TOP_K (default: 4)
# - LLM_TEMPERATURE (default: 0.3)
```

### Modify Prompts
```python
# Edit backend/src/prompts.py
# Change SYSTEM_PROMPT or RAG_PROMPT_TEMPLATE
# Restart Streamlit to see changes
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run commands from `backend/` directory |
| `.env file not found` | Copy `.env.example` to `.env` and add API key |
| `Vector store not initialized` | Run `python scripts/create_vectordb.py` |
| Slow embedding generation | Normal on first run (downloads model) |
| API authentication error | Check `ANTHROPIC_API_KEY` in `.env` |

## 📖 Resources

### Documentation
- [LangChain Docs](https://python.langchain.com/)
- [Anthropic Claude API](https://docs.anthropic.com/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Learning Materials
- [Anthropic's RAG Guide](https://www.anthropic.com/index/contextual-retrieval)
- [Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

## 🤝 Contributing

This is a personal learning project, but feedback and suggestions are welcome!

## 📝 License

MIT License - Feel free to use this for your own learning projects.

## 🙏 Acknowledgments

- Based on the Kafka ecosystem documentation
- Built with LangChain, Claude, and Streamlit
- Inspired by modern RAG architectures

---

## Next Steps

1. **Start Here**: Read [GETTING_STARTED.md](GETTING_STARTED.md)
2. **Quick Reference**: Bookmark [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. **Understand Flow**: Review [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Start Building**: Follow Week 1 Day 2 tasks

**Questions?** Check the docs or review the inline code comments!

🚀 **Happy building!**
