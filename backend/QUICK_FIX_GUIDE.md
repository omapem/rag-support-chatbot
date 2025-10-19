# Quick Fix Guide - RAG Retrieval Improvements

## 🚀 Quick Start (Apply All Fixes)

### 1. Re-ingest Documents (REQUIRED)
```bash
cd backend
python reingest_documents.py
```

**Why?** The old vector database has 800-character chunks. The new system uses 1000-character chunks with better overlap to keep command examples together.

### 2. Restart Application
```bash
streamlit run streamlit_app.py
```

### 3. Test Improved Retrieval
Try the query that previously failed:
```
"How do I create a topic?"
```

**Expected**: You should now see `kafka-topics.sh` commands from page 33! ✅

---

## 🎯 What Was Fixed

| Fix | Impact | Files Changed |
|-----|--------|---------------|
| **Increased top_k** (4→6) | +50% retrieval coverage | `config.py` |
| **Hybrid Search** (BM25 + semantic) | Catches keyword matches | `retrieval.py` |
| **Better Chunking** (1000 chars, 200 overlap) | Preserves context | `config.py`, `ingestion.py` |
| **Query Expansion** | Bridges semantic gap | `retrieval.py` |
| **Debug Mode** | Visibility into rankings | `retrieval.py` |

---

## 🔧 Configuration Reference

### Current Settings (config.py)
```python
chunk_size: int = 1000        # Was: 800
chunk_overlap: int = 200      # Was: 100
top_k: int = 6                # Was: 4
```

### Hybrid Search Settings (retrieval.py)
```python
# Ensemble weights
semantic_weight = 0.7   # 70% semantic similarity
keyword_weight = 0.3    # 30% BM25 keyword matching
```

---

## 🐛 Debug Mode Usage

Enable detailed retrieval diagnostics:

```python
from src.retrieval import Retriever

retriever = Retriever(enable_hybrid=True)
results = retriever.retrieve_and_format(
    query="How do I create a topic?",
    debug=True  # Shows scores and rankings
)
```

**Output**:
```
=== RETRIEVAL DEBUG ===
Query: How do I create a topic?
Top-k: 6

Retrieved 6 documents:
1. Score: 0.7234 | Source: Kafka The Definitive Guide.pdf (page 33)
   Preview: Once the Kafka broker is started, we can verify it...
2. Score: 0.6891 | Source: Kafka The Definitive Guide.pdf (page 45)
...
```

---

## 🧪 Test Queries

Verify improvements with these test cases:

| Query | Should Retrieve | Page |
|-------|----------------|------|
| "How do I create a topic?" | `kafka-topics.sh --create` | 33 |
| "Delete a topic" | `kafka-topics.sh --delete` | - |
| "List all topics" | `kafka-topics.sh --list` | - |
| "Consumer group commands" | `kafka-consumer-groups.sh` | - |
| "Configure retention" | `log.retention.hours` | - |

---

## ⚙️ Advanced Configuration

### Disable Hybrid Search (if needed)
```python
retriever = Retriever(enable_hybrid=False)  # Semantic only
```

### Disable Query Expansion
```python
results = retriever.retrieve(query, expand_query=False)
```

### Adjust Retrieval Count
```python
# Via code
results = retriever.retrieve(query, top_k=10)

# Via Streamlit UI
Use the slider: "Number of sources to retrieve"
```

---

## 🔄 Re-ingestion Details

### What `reingest_documents.py` Does:
1. ✅ Backs up old vector DB (optional confirmation)
2. ✅ Deletes old Chroma database
3. ✅ Re-chunks PDFs with new settings (1000 chars, 200 overlap)
4. ✅ Creates new embeddings
5. ✅ Initializes hybrid search (BM25 index)
6. ✅ Tests retrieval with sample query

### When to Re-ingest:
- ✅ After changing `chunk_size` or `chunk_overlap`
- ✅ After adding new PDF documents
- ✅ After updating chunking separators
- ❌ NOT needed for top_k changes (runtime config)
- ❌ NOT needed for query expansion changes (runtime)

---

## 📊 Expected Performance

### Retrieval Quality
- **Before**: "How do I create a topic?" → ❌ Missing page 33 content
- **After**: "How do I create a topic?" → ✅ Shows CLI commands from page 33

### Speed
- **Initialization**: +2-3 seconds (one-time BM25 indexing)
- **Query Latency**: +50-100ms (hybrid search overhead)
- **User Impact**: Barely noticeable, better results worth it

### Storage
- **Vector DB Size**: ~20-25% larger (bigger chunks, more overlap)
- **Example**: 100MB PDF → ~25MB vector DB (was ~20MB)

---

## ❓ Troubleshooting

### "Hybrid search initialization failed"
**Cause**: Vector store empty or not loaded
**Fix**:
```bash
python reingest_documents.py  # Rebuild vector DB
```

### "Still not finding page 33 content"
**Diagnosis**:
```python
# Check if you re-ingested
from pathlib import Path
db_path = Path("data/chroma_db")
print(db_path.stat().st_mtime)  # Should be recent

# Test retrieval manually
from src.retrieval import Retriever
retriever = Retriever(enable_hybrid=True)
results = retriever.retrieve_and_format("How do I create a topic?", debug=True)
# Check debug output for page 33 content
```

### "Query expansion not working"
**Check**:
```python
from src.retrieval import Retriever
retriever = Retriever()
expanded = retriever.expand_query("How do I create a topic?")
print(expanded)
# Should include "kafka-topics.sh topic creation new topic"
```

---

## 📝 Verification Checklist

After applying fixes:

- [ ] Re-ingested documents with new chunking settings
- [ ] Tested query "How do I create a topic?"
- [ ] Verified page 33 content appears in results
- [ ] Checked that CLI commands are included
- [ ] Confirmed hybrid search initialized (check startup logs)
- [ ] Tested 3-5 other queries for quality

**All checks passed?** ✅ System is ready!

---

## 🆘 Need Help?

1. **Check logs**: Look for "Hybrid search initialized successfully"
2. **Run debug mode**: Use `debug=True` in `retrieve_and_format()`
3. **Review summary**: See [RAG_FIXES_SUMMARY.md](./RAG_FIXES_SUMMARY.md) for details
4. **Verify chunking**: Ensure `chunk_size=1000` in config.py

---

**Last Updated**: 2025-10-18
**System Version**: v1.1 (Post-fixes)
