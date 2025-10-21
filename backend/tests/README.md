# Test Suite Documentation

## Overview

Comprehensive test suite for the RAG Support Chatbot backend, including unit tests, integration tests, and test data infrastructure.

## Quick Start

### 1. Set Up Test Environment

```bash
# From backend/ directory
source venv/bin/activate

# Install test dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Initialize Test Vector Database

```bash
# Create test vector database with sample Kafka documents
python tests/setup_test_vectordb.py
```

This creates a small vector database in `tests/test_vectordb/` with 3 sample Kafka documents for testing.

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run only unit tests (fast)
pytest tests/test_models.py tests/test_conversation_memory.py -v

# Run integration tests (requires test vectordb)
pytest tests/test_integration_api.py -m integration -v

# Run with coverage
pytest tests/ --cov=app --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── README.md                         # This file
├── conftest.py                       # Pytest fixtures and configuration
├── setup_test_vectordb.py           # Script to create test vector database
├── test_data/                        # Sample documents for testing
│   ├── kafka_basics.txt
│   ├── kafka_replication.txt
│   └── kafka_producers_consumers.txt
├── test_vectordb/                    # Test Chroma database (created by setup script)
│   └── [chroma files]
├── test_models.py                    # Unit tests for Pydantic models (23 tests)
├── test_conversation_memory.py       # Unit tests for conversation service (17 tests)
├── test_api_endpoints.py            # Unit tests for API with mocking
└── test_integration_api.py          # Integration tests with real RAG

Total: 40+ unit tests, 15+ integration tests
```

## Test Categories

### Unit Tests (Fast, No Dependencies)
- **test_models.py**: Pydantic model validation
- **test_conversation_memory.py**: Conversation memory service
- **test_api_endpoints.py**: API endpoints with mocked RAG

### Integration Tests (Require Setup)
- **test_integration_api.py**: Full API with real vector database

## Test Markers

Tests are organized with pytest markers:

```bash
# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration       # Integration tests
pytest -m api               # API tests
pytest -m service           # Service layer tests
pytest -m slow              # Slow tests (threading, etc.)

# Combine markers
pytest -m "integration and not slow"
```

## Test Fixtures

### Shared Fixtures (conftest.py)

- **`test_config`**: Test configuration settings
- **`sample_chat_messages`**: Sample conversation data
- **`sample_rag_response`**: Mock RAG response
- **`test_vectordb_path`**: Path to test vector database
- **`use_test_vectordb`**: Configures tests to use test DB
- **`test_vectordb`**: Provides EmbeddingManager with test DB

### Using Fixtures in Tests

```python
def test_with_vectordb(test_vectordb):
    """Test uses the test vector database automatically."""
    retriever = test_vectordb.get_retriever(top_k=3)
    results = retriever.invoke("How do I create a topic?")
    assert len(results) > 0
```

## Test Data

### Sample Documents

Located in `tests/test_data/`:

1. **kafka_basics.txt** - Core concepts, topic creation, listing
2. **kafka_replication.txt** - Replication factor, fault tolerance, ISR
3. **kafka_producers_consumers.txt** - Producers, consumers, consumer groups

These documents are intentionally small and focused for fast, predictable testing.

### Test Vector Database

Created by `setup_test_vectordb.py`:
- **Location**: `tests/test_vectordb/`
- **Documents**: 3
- **Chunks**: ~9 (500 tokens each with 50 overlap)
- **Embeddings**: all-MiniLM-L6-v2 (free, local)
- **Collection**: kafka_test_docs

## Running Different Test Scenarios

### Fast Development Loop
```bash
# Watch mode - tests run automatically on save
ptw tests/ -- -v --tb=short -m "unit and not slow"
```

### Full Test Suite
```bash
# All tests with coverage
pytest tests/ --cov=app --cov=src --cov-report=html --cov-report=term
```

### Integration Tests Only
```bash
# Requires test vectordb and .env with ANTHROPIC_API_KEY
pytest tests/test_integration_api.py -v
```

### Specific Test Function
```bash
pytest tests/test_models.py::TestChatRequest::test_query_validation -v
```

## Skip Conditions

Integration tests automatically skip if:
- Test vector database doesn't exist → Run `setup_test_vectordb.py`
- `ANTHROPIC_API_KEY` not in environment → Add to `.env` file
- Marked as `slow` and running with `-m "not slow"`

## Coverage Goals

- **Unit Tests**: >95% coverage of models and services
- **Integration Tests**: >70% coverage of API routes
- **Overall**: >80% coverage of production code

### View Coverage Report
```bash
pytest tests/ --cov=app --cov=src --cov-report=html
open htmlcov/index.html  # macOS
```

## Continuous Integration

Tests are CI/CD ready:

```yaml
# Example GitHub Actions
- name: Set up test environment
  run: |
    python tests/setup_test_vectordb.py

- name: Run tests
  run: |
    pytest tests/ --cov=app --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Test Vector Database Not Found
```bash
# Solution: Run the setup script
python tests/setup_test_vectordb.py
```

### Integration Tests Skipped
```bash
# Check skip reason
pytest tests/test_integration_api.py -v -rs

# Common reasons:
# 1. Missing test vectordb - run setup script
# 2. Missing ANTHROPIC_API_KEY - add to .env
```

### Slow Test Performance
```bash
# Skip slow tests during development
pytest tests/ -m "not slow" -v
```

### Module Import Errors
```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Ensure you're in backend/ directory
cd backend
pytest tests/ -v
```

## Best Practices

1. **Run tests before committing**: `pytest tests/ -v`
2. **Use watch mode for TDD**: `ptw tests/`
3. **Keep tests fast**: Mock external dependencies
4. **Test one thing**: Each test should have a clear focus
5. **Use descriptive names**: Test names should describe what they test
6. **Clean up after tests**: Use `autouse` fixtures
7. **Update test data**: Keep sample documents relevant and minimal

## Adding New Tests

### Unit Test Template
```python
def test_new_feature():
    """Test description explaining what is being tested."""
    # Arrange
    input_data = {...}

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_value
```

### Integration Test Template
```python
@pytest.mark.integration
def test_new_integration(test_vectordb, client):
    """Integration test using real components."""
    # Test with real vector database and API client
    response = client.post("/chat/", json={"query": "test"})
    assert response.status_code == 200
```

## Maintenance

### Updating Test Data
1. Edit files in `tests/test_data/`
2. Run `python tests/setup_test_vectordb.py` to rebuild
3. Verify with `pytest tests/test_integration_api.py -v`

### Adding New Test Documents
1. Add `.txt` file to `tests/test_data/`
2. Run setup script to update vector database
3. Update tests if needed

## Support

For issues or questions:
1. Check [RUN_TESTS.md](../RUN_TESTS.md) for detailed testing guide
2. Review [TESTING_SUMMARY.md](../TESTING_SUMMARY.md) for current status
3. See test output with `pytest -v` for detailed error messages
