# RAG System for Course Q&A

A comprehensive Retrieval-Augmented Generation (RAG) system built to answer questions about course materials from DataTalks.Club courses. The system supports both **Elasticsearch** for traditional search and **Qdrant** for vector-based semantic search.

## 🏗️ Architecture

The system is modularly designed with the following components:

```
rag/
├── __init__.py                 # Main package exports
├── config.py                   # Configuration constants (including embedding config)
├── models/                     # Data models
│   ├── __init__.py
│   └── course.py              # Course enum with helper methods
├── data/                       # Data loading and processing
│   ├── __init__.py
│   ├── loader.py              # DocumentLoader for fetching documents
│   └── vector_store.py        # VectorStoreLoader, QdrantVectorLoader, and VectorSearcher
├── search/                     # Search and retrieval
│   ├── __init__.py
│   ├── elasticsearch_client.py # Elasticsearch operations
│   ├── qdrant_client_custom.py # Qdrant vector database operations
│   └── query_builder.py       # Query construction
├── formatting/                 # Text formatting and templates
│   ├── __init__.py
│   └── context.py             # Context and prompt formatting
├── llm/                        # Large Language Model integration
│   ├── __init__.py
│   └── openai_client.py       # OpenAI API client
├── pipeline/                   # Main RAG pipeline
│   ├── __init__.py
│   └── rag.py                 # RAGPipeline orchestrator
└── tests/                      # Comprehensive test suite
    ├── __init__.py
    ├── test_course.py
    ├── test_document_loader.py
    ├── test_query_builder.py
    ├── test_context_formatter.py
    ├── test_openai_client.py
    ├── test_elasticsearch_client.py
    ├── test_qdrant_client.py    # NEW: Qdrant client tests
    ├── test_rag_pipeline.py
    └── test_runner.py          # Custom test runner
```

## 🔍 Vector Search Support

This system now supports **dual search backends**:

### **Elasticsearch** (Traditional Search)
- Full-text search with BM25 scoring
- Course filtering and boosting
- Proven reliability for keyword-based queries

### **Qdrant** (Vector Search) 🆕
- Semantic similarity search using embeddings
- Support for `jinaai/jina-embeddings-v2-small-en` model
- 512-dimensional vector space
- Automatic document-to-vector conversion with VectorStoreLoader
- High-level APIs for easy integration with QdrantVectorLoader
- **VectorSearcher** class for dedicated search operations
- Cosine similarity matching
- Course filtering support
- Both text query and pre-computed vector search
- **Payload indexing** for optimized metadata filtering

### **Configuration**
```python
# Embedding Configuration (in rag/config.py)
EMBEDDING_DIMENSIONALITY = 512
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
QDRANT_URL = "http://localhost:6333"
```

### **VectorSearcher Usage Examples** 🆕

The new `VectorSearcher` class provides dedicated search functionality with proper separation of concerns:

```python
from rag.data.vector_store import VectorSearcher

# Initialize the searcher
searcher = VectorSearcher()

# Search with text query (embeds query automatically)
results = searcher.search(
    query="What is Docker?",
    collection_name="course-questions",
    limit=5,
    course_filter="docker-course"  # Optional course filter
)

# Search with pre-computed vector
query_vector = [0.1, 0.2, 0.3, ...]  # Your 512-dimensional vector
results = searcher.search_with_vector(
    collection_name="course-questions",
    query_vector=query_vector,
    limit=3
)

# Just embed a query (useful for pre-computing vectors)
embedding = searcher.embed_query("What is machine learning?")
```

**Key Features:**
- **Automatic embedding**: Text queries are embedded using the configured model
- **Flexible search**: Support for both text queries and pre-computed vectors  
- **Course filtering**: Filter results by specific courses
- **Error handling**: Comprehensive error handling and logging
- **Integration**: Works seamlessly with existing VectorStoreLoader and QdrantVectorLoader

### **Payload Indexing for Optimized Filtering** 🆕

The system now automatically creates payload indexes for metadata fields during collection setup, providing significant performance improvements for filtering operations:

```python
from rag.data.vector_store import QdrantVectorLoader
from rag.search.qdrant_client_custom import QdrantClientCustom
from qdrant_client.models import PayloadSchemaType

# Payload index is automatically created during collection setup
vector_loader = QdrantVectorLoader()
result = vector_loader.setup_collection(
    collection_name="course-questions",
    delete_if_exists=True
)

# Payload index creation is confirmed in the result
print(f"Payload index created: {result['payload_index_created']}")  # True

# Or create payload index manually
qdrant_client = QdrantClientCustom()
qdrant_client.create_payload_index(
    collection_name="course-questions",
    field_name="course",
    field_schema=PayloadSchemaType.KEYWORD  # for exact matching
)
```

**Benefits:**
- **Faster Queries**: Optimized filtering by 'course' field
- **Exact Matching**: Keyword schema ensures precise string matching
- **Better Performance**: Significantly improves query speed when using course filters
- **Automatic Setup**: Index is created automatically during collection initialization

## 📦 **Installation & Dependencies**

This project uses modern Python packaging standards with `pyproject.toml` as the single source of truth for dependencies.

### **Quick Start**

```bash
# Clone the repository
git clone <repository-url>
cd llm-zoomcamp

# Install in development mode with all dependencies
pip install -e ".[dev,jupyter,tokens]"

# Or just core dependencies (includes both Elasticsearch and Qdrant)
pip install -e .
```

### **Dependency Management Explained**

#### **🎯 Primary Approach: pyproject.toml**
- **Core dependencies**: Defined in `[project.dependencies]`
- **Optional dependencies**: Organized in `[project.optional-dependencies]`
- **Configuration**: All tool settings centralized

#### **📋 Available Dependency Groups**

| Group | Install Command | Purpose |
|-------|----------------|---------|
| **Core** | `pip install -e .` | Essential runtime dependencies (Elasticsearch, Qdrant, OpenAI) |
| **Development** | `pip install -e ".[dev]"` | Testing, linting, formatting |
| **Jupyter** | `pip install -e ".[jupyter]"` | Notebook support |
| **Tokens** | `pip install -e ".[tokens]"` | Token counting utilities |
| **All** | `pip install -e ".[dev,jupyter,tokens]"` | Everything |

#### **🔒 Security Updates**
- **urllib3**: Updated to `2.5.0` to address security vulnerabilities
- **Dependency scanning**: Removed Safety CLI (simplified dependency management)

#### **🔧 requirements.txt Role**
The `requirements.txt` file now serves as:
- **Convenience wrapper** for `pyproject.toml`
- **CI/CD compatibility** for older systems
- **Exact version pinning** when needed (commented out by default)

### **Development Workflows**

```bash
# Development setup
pip install -e ".[dev,jupyter,tokens]"

# Run tests (now includes Qdrant tests)
pytest rag/tests/

# Code formatting
black rag/
isort rag/

# Linting
flake8 rag/
pylint rag/

# Type checking
mypy rag/
```

### **Docker Compose Setup**

Start both Elasticsearch and Qdrant services:

```bash
# Start all services
docker-compose up -d

# Verify services
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:6333/collections      # Qdrant
```

The `docker-compose.yml` includes:
- **Elasticsearch**: Port 9200 (traditional search)
- **Qdrant**: Port 6333 (vector search)

### **CI/CD Integration**

Our GitHub Actions workflows automatically:
- Install dependencies using `pip install -e ".[dev,jupyter,tokens]"`
- Run comprehensive test suites (including new Qdrant tests)
- Perform code quality checks
- Generate coverage reports
- **Note**: Removed Safety CLI for simplified security scanning

### **Troubleshooting Dependencies**

| Issue | Solution |
|-------|----------|
| **Import errors** | Run `pip install -e ".[dev,jupyter,tokens]"` |
| **Missing test tools** | Ensure `[dev]` extras are installed |
| **Notebook issues** | Install with `[jupyter]` extras |
| **Token counting fails** | Install with `[tokens]` extras |
| **Qdrant connection fails** | Check `docker-compose up qdrant` |
| **CI failures** | Check workflow uses `pip install -e ".[dev,jupyter,tokens]"` |

## 🧪 Testing

The project includes comprehensive unit tests for all components, **including new Qdrant client and VectorSearcher tests**:

### Running Tests

```bash
# Run all tests (now 180+ tests including Qdrant, Vector Store, and VectorSearcher)
python -m pytest rag/tests/ -v

# Run tests with coverage
python -m pytest rag/tests/ --cov=rag --cov-report=html

# Run specific test modules
python -m pytest rag/tests/test_qdrant_client.py -v
python -m pytest rag/tests/test_vector_store.py -v  # Includes VectorSearcher tests

# Run custom test runner
python -m rag.tests.test_runner

# Run with the simple test script
python run_tests.py
```

### **New Test Coverage** 🆕

- **VectorSearcher Tests**: Complete unit tests for the new VectorSearcher class
  - `TestVectorSearcher`: Core functionality tests  
  - `TestVectorSearcherIntegration`: Integration tests with real-like dependencies
  - Tests for embedding generation, search operations, error handling, and parameter validation
- **Qdrant Client Tests**: Vector database operations and search functionality
- **Vector Store Tests**: Document loading, embedding generation, and Qdrant integration

### Test Coverage

The test suite includes:

- **Course Model Tests**: Enum functionality, validation, helper methods
- **Document Loader Tests**: Data fetching, processing, error handling
- **Query Builder Tests**: Search query construction, course filtering
- **Context Formatter Tests**: Document formatting, prompt building
- **OpenAI Client Tests**: API interactions, cost calculation, error handling
- **Elasticsearch Client Tests**: Index operations, document management
- **Qdrant Client Tests**: 🆕 Vector database initialization, connection handling
- **Vector Store Tests**: 🆕 Embedding generation, point creation, document loading
- **RAG Pipeline Tests**: End-to-end functionality, integration testing

### Using Make for Testing

```bash
# Run unit tests
make test-unit

# Run integration tests  
make test-integration

# Run specific module tests
make test-module MODULE=course

# Run custom test runner
make test-runner

# Run with coverage
make test-runner-coverage
```

## 🔧 Configuration

The system can be configured through environment variables or direct parameters:

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
ELASTICSEARCH_HOST=http://localhost:9200
QDRANT_URL=http://localhost:6333
INDEX_NAME=zoomcamp-courses-questions
```

### Configuration Options

```python
from rag import RAGPipeline

# Default configuration (uses Elasticsearch by default)
pipeline = RAGPipeline()

# Custom configuration
pipeline = RAGPipeline(
    elasticsearch_host="http://localhost:9200",
    index_name="custom-index",
    openai_model="gpt-4o",
    documents_url="https://custom-docs.json"
)

# Using Qdrant for vector search
from rag.data.vector_store import QdrantVectorLoader
from rag.search.qdrant_client_custom import QdrantClientCustom

# High-level interface for loading documents to Qdrant
# DocumentLoader automatically creates 'full_text' field by combining 'question' + 'text'
qdrant_loader = QdrantVectorLoader()

# Setup collection with all documents
result = qdrant_loader.setup_collection(
    collection_name="course-docs",
    delete_if_exists=True
)
print(f"Loaded {result['documents_loaded']} documents")

# Or setup with course filter
ml_result = qdrant_loader.setup_collection(
    collection_name="ml-docs",
    course_filter="machine-learning-zoomcamp",
    delete_if_exists=True
)

# Low-level vector operations
from rag.data.vector_store import VectorStoreLoader

vector_store = VectorStoreLoader()
# Note: documents must have 'full_text' field (auto-created by DocumentLoader)
embeddings = vector_store.generate_embeddings(documents)
points = vector_store.create_points(documents, embeddings)
uploaded = vector_store.load_to_qdrant("my-collection", documents)
```

## 📊 Usage Examples

### Basic Usage

```python
from rag import RAGPipeline, Course

# Initialize the pipeline
pipeline = RAGPipeline()

# Setup the index (one-time operation)
indexed_count = pipeline.setup_index()
print(f"Indexed {indexed_count} documents")

# Ask a question
answer = pipeline.ask_question("How do I copy files to a Docker container?")
print(answer)

# Ask with course filtering
answer = pipeline.ask_question(
    "What is Docker?",
    course_filter=Course.DATA_ENGINEERING_ZOOMCAMP
)
print(answer)
```

### Advanced Usage

```python
# Search for documents only
documents = pipeline.search("Docker containers", num_results=3)

# Generate answer with usage tracking
result = pipeline.ask_question(
    "How to install Docker?",
    include_usage=True,
    debug=True
)

print(f"Answer: {result['answer']}")
print(f"Tokens used: {result['usage']['total_tokens']}")
print(f"Cost: ${result['cost']['total_cost']:.4f}")
```

### Component Usage

```python
from rag.models import Course
from rag.search import QueryBuilder
from rag.search.qdrant_client_custom import QdrantClientCustom
from rag.formatting import ContextFormatter

# Course filtering
courses = Course.list_courses()
course = Course.from_string("data-engineering-zoomcamp")

# Query building
qb = QueryBuilder()
query = qb.build_search_query(
    "Docker question",
    course_filter=Course.DATA_ENGINEERING_ZOOMCAMP
)

# Qdrant client (vector search)
qdrant_client = QdrantClientCustom()

# Context formatting
cf = ContextFormatter()
context = cf.format_context(documents)
prompt = cf.build_prompt("Question", context)
```

## 🔍 CI/CD Pipeline

The project includes GitHub Actions workflows:

### Test Workflow (`.github/workflows/tests.yml`)

- **Matrix Testing**: Tests across Python 3.9, 3.10, 3.11
- **Unit Tests**: Comprehensive test suite execution (including Qdrant tests)
- **Coverage Reporting**: Code coverage analysis and reporting
- **Integration Tests**: Tests with real Elasticsearch instance
- **Dependency Caching**: Faster CI runs with dependency caching

### Code Quality Workflow (`.github/workflows/code-quality.yml`)

- **Code Formatting**: Black and isort checks
- **Linting**: flake8, pylint analysis
- **Type Checking**: mypy static type analysis
- **Security Scanning**: bandit security checks
- **Documentation**: Docstring style validation
- **Simplified Dependencies**: Removed Safety CLI for streamlined workflow

### Workflow Features

- **Parallel Execution**: Multiple jobs run simultaneously
- **Artifact Upload**: Coverage reports and test results
- **Integration Testing**: Real Elasticsearch service testing
- **Multi-Python Support**: Testing across Python versions
- **Streamlined Security**: Focused on bandit for security scanning

## 🛠️ Development Workflow

### Code Quality Tools

The project uses several tools for code quality:

```bash
# Format code
black rag/
isort rag/

# Lint code
flake8 rag/
pylint rag/

# Type checking
mypy rag/

# Security scanning
bandit -r rag/
```

### Pre-commit Workflow

```bash
# Quick development check
make quick-check

# Full CI simulation
make ci-test

# Clean temporary files
make clean
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Elasticsearch Connection**: Verify Elasticsearch is running (`docker-compose up elasticsearch`)
3. **Qdrant Connection**: Verify Qdrant is running (`docker-compose up qdrant`)
4. **OpenAI API**: Check API key configuration
5. **Test Failures**: Run tests individually to isolate issues

### Running Individual Components

```bash
# Test just the Course enum (no external dependencies)
python3 -c "
import sys
sys.path.append('.')
exec(open('rag/models/course.py').read())
print('Course enum loaded successfully')
"

# Test Qdrant client
python -m pytest rag/tests/test_qdrant_client.py::TestQdrantClient::test_init_success -v

# Test query building
python -m pytest rag/tests/test_query_builder.py::TestQueryBuilder::test_build_search_query_with_course_filter -v
```

## 📚 API Documentation

### Main Classes

- **`RAGPipeline`**: Main orchestrator for the RAG system
- **`Course`**: Enum for available courses with helper methods
- **`DocumentLoader`**: Handles document fetching and processing
- **`ElasticsearchClient`**: Manages Elasticsearch operations
- **`QdrantClientCustom`**: 🆕 Manages Qdrant vector database operations
- **`VectorStoreLoader`**: 🆕 Handles embedding generation and vector database operations
- **`QdrantVectorLoader`**: 🆕 High-level interface for loading documents to Qdrant
- **`QueryBuilder`**: Constructs search queries
- **`ContextFormatter`**: Formats documents and builds prompts
- **`OpenAIClient`**: Handles OpenAI API interactions

### Key Methods

- **`pipeline.setup_index()`**: Initialize and populate the search index
- **`pipeline.ask_question(question, course_filter=None)`**: End-to-end Q&A
- **`pipeline.search(question, course_filter=None)`**: Search for relevant documents
- **`pipeline.health_check()`**: Verify system components
- **`QdrantClientCustom(url)`**: 🆕 Initialize vector database client

### Development Guidelines

- Add comprehensive tests for new features
- Follow the existing code style (Black, isort)
- Update documentation for API changes
- Ensure all CI checks pass
- Test both Elasticsearch and Qdrant integrations

---

## 🚦 Quick Start

```bash
# 1. Clone and install
git clone <repository>
cd llm-zoomcamp
make dev-setup

# 2. Run tests (includes new Qdrant tests)
make test

# 3. Start both search backends
docker-compose up -d

# 4. Verify services
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:6333/collections      # Qdrant

# 5. Set environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# 6. Test the system
python example_usage.py
```

## 🆕 Recent Updates

- **✅ Qdrant Integration**: Added vector database support for semantic search
- **✅ Vector Store Implementation**: Complete vector store functionality with VectorStoreLoader and QdrantVectorLoader
- **✅ Security Updates**: Updated urllib3 to v2.5.0 to fix vulnerabilities  
- **✅ Simplified Dependencies**: Removed Safety CLI for streamlined workflow
- **✅ Enhanced Testing**: Added comprehensive Qdrant client and vector store tests (153 total tests)
- **✅ Code Quality**: Fixed docstring formatting and linting issues
- **✅ Docker Support**: Added Qdrant service to docker-compose.yml
- **✅ CI/CD Integration**: Added vector store integration tests to GitHub Actions
