# RAG System for Course Q&A

A comprehensive Retrieval-Augmented Generation (RAG) system built to answer questions about course materials from DataTalks.Club courses. This system transforms the original Jupyter notebook into a well-structured, production-ready Python package with comprehensive testing and CI/CD.

## 🏗️ Architecture

The system is modularly designed with the following components:

```
rag/
├── __init__.py                 # Main package exports
├── config.py                   # Configuration constants
├── models/                     # Data models
│   ├── __init__.py
│   └── course.py              # Course enum with helper methods
├── data/                       # Data loading and processing
│   ├── __init__.py
│   └── loader.py              # DocumentLoader for fetching documents
├── search/                     # Search and retrieval
│   ├── __init__.py
│   ├── elasticsearch_client.py # Elasticsearch operations
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
    ├── test_rag_pipeline.py
    └── test_runner.py          # Custom test runner
```

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Type Safety**: Full type annotations throughout the codebase
- **Comprehensive Testing**: 100+ unit tests covering all components
- **CI/CD Ready**: GitHub Actions workflows for testing and code quality
- **Course Filtering**: Support for filtering by specific courses
- **Cost Tracking**: Token usage and cost calculation for OpenAI API
- **Error Handling**: Robust error handling and logging
- **Configurable**: Flexible configuration options
- **Documentation**: Extensive docstrings and examples

## 📦 **Installation & Dependencies**

This project uses modern Python packaging standards with `pyproject.toml` as the single source of truth for dependencies.

### **Quick Start**

```bash
# Clone the repository
git clone <repository-url>
cd llm-zoomcamp-1

# Install in development mode with all dependencies
pip install -e ".[dev,jupyter,tokens]"

# Or just core dependencies
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
| **Core** | `pip install -e .` | Essential runtime dependencies |
| **Development** | `pip install -e ".[dev]"` | Testing, linting, formatting |
| **Jupyter** | `pip install -e ".[jupyter]"` | Notebook support |
| **Tokens** | `pip install -e ".[tokens]"` | Token counting utilities |
| **All** | `pip install -e ".[dev,jupyter,tokens]"` | Everything |

#### **🔧 requirements.txt Role**
The `requirements.txt` file now serves as:
- **Convenience wrapper** for `pyproject.toml`
- **CI/CD compatibility** for older systems
- **Exact version pinning** when needed (commented out by default)

### **Development Workflows**

```bash
# Development setup
pip install -e ".[dev,jupyter,tokens]"

# Run tests
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

### **CI/CD Integration**

Our GitHub Actions workflows automatically:
- Install dependencies using `pip install -e ".[dev,jupyter,tokens]"`
- Run comprehensive test suites
- Perform code quality checks
- Generate coverage reports

### **Troubleshooting Dependencies**

| Issue | Solution |
|-------|----------|
| **Import errors** | Run `pip install -e ".[dev,jupyter,tokens]"` |
| **Missing test tools** | Ensure `[dev]` extras are installed |
| **Notebook issues** | Install with `[jupyter]` extras |
| **Token counting fails** | Install with `[tokens]` extras |
| **CI failures** | Check workflow uses `pip install -e ".[dev,jupyter,tokens]"` |

## 🧪 Testing

The project includes comprehensive unit tests for all components:

### Running Tests

```bash
# Run all tests
python -m pytest rag/tests/ -v

# Run tests with coverage
python -m pytest rag/tests/ --cov=rag --cov-report=html

# Run specific test module
python -m pytest rag/tests/test_course.py -v

# Run custom test runner
python -m rag.tests.test_runner

# Run with the simple test script
python run_tests.py
```

### Test Coverage

The test suite includes:

- **Course Model Tests**: Enum functionality, validation, helper methods
- **Document Loader Tests**: Data fetching, processing, error handling
- **Query Builder Tests**: Search query construction, course filtering
- **Context Formatter Tests**: Document formatting, prompt building
- **OpenAI Client Tests**: API interactions, cost calculation, error handling
- **Elasticsearch Client Tests**: Index operations, document management
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
INDEX_NAME=zoomcamp-courses-questions
```

### Configuration Options

```python
from rag import RAGPipeline

# Default configuration
pipeline = RAGPipeline()

# Custom configuration
pipeline = RAGPipeline(
    elasticsearch_host="http://localhost:9200",
    index_name="custom-index",
    openai_model="gpt-4o",
    documents_url="https://custom-docs.json"
)
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

# Context formatting
cf = ContextFormatter()
context = cf.format_context(documents)
prompt = cf.build_prompt("Question", context)
```

## 🔍 CI/CD Pipeline

The project includes GitHub Actions workflows:

### Test Workflow (`.github/workflows/tests.yml`)

- **Matrix Testing**: Tests across Python 3.9, 3.10, 3.11
- **Unit Tests**: Comprehensive test suite execution
- **Coverage Reporting**: Code coverage analysis and reporting
- **Integration Tests**: Tests with real Elasticsearch instance
- **Dependency Caching**: Faster CI runs with dependency caching

### Code Quality Workflow (`.github/workflows/code-quality.yml`)

- **Code Formatting**: Black and isort checks
- **Linting**: flake8, pylint analysis
- **Type Checking**: mypy static type analysis
- **Security Scanning**: bandit security checks
- **Documentation**: Docstring style validation

### Workflow Features

- **Parallel Execution**: Multiple jobs run simultaneously
- **Artifact Upload**: Coverage reports and test results
- **Integration Testing**: Real Elasticsearch service testing
- **Security Checks**: Dependency vulnerability scanning
- **Multi-Python Support**: Testing across Python versions

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
safety scan
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

## 📈 Testing Metrics

Current test coverage includes:

- **100+ Test Cases**: Comprehensive coverage of all components
- **Mock Testing**: Isolated unit tests with proper mocking
- **Error Path Testing**: Edge cases and error conditions
- **Integration Testing**: End-to-end functionality verification
- **Performance Testing**: Token usage and cost validation

### Test Statistics

- **Course Model**: 8 test cases
- **Document Loader**: 15 test cases  
- **Query Builder**: 12 test cases (from original notebook)
- **Context Formatter**: 18 test cases
- **OpenAI Client**: 20 test cases
- **Elasticsearch Client**: 22 test cases
- **RAG Pipeline**: 12 test cases

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Elasticsearch Connection**: Verify Elasticsearch is running
3. **OpenAI API**: Check API key configuration
4. **Test Failures**: Run tests individually to isolate issues

### Running Individual Components

```bash
# Test just the Course enum (no external dependencies)
python3 -c "
import sys
sys.path.append('.')
exec(open('rag/models/course.py').read())
print('Course enum loaded successfully')
"

# Test query building
python -m pytest rag/tests/test_query_builder.py::TestQueryBuilder::test_build_search_query_with_course_filter -v
```

## 📚 API Documentation

### Main Classes

- **`RAGPipeline`**: Main orchestrator for the RAG system
- **`Course`**: Enum for available courses with helper methods
- **`DocumentLoader`**: Handles document fetching and processing
- **`ElasticsearchClient`**: Manages Elasticsearch operations
- **`QueryBuilder`**: Constructs search queries
- **`ContextFormatter`**: Formats documents and builds prompts
- **`OpenAIClient`**: Handles OpenAI API interactions

### Key Methods

- **`pipeline.setup_index()`**: Initialize and populate the search index
- **`pipeline.ask_question(question, course_filter=None)`**: End-to-end Q&A
- **`pipeline.search(question, course_filter=None)`**: Search for relevant documents
- **`pipeline.health_check()`**: Verify system components

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes and add tests**
4. **Run quality checks**: `make ci-test`
5. **Submit a pull request**

### Development Guidelines

- Add comprehensive tests for new features
- Follow the existing code style (Black, isort)
- Update documentation for API changes
- Ensure all CI checks pass

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **DataTalks.Club** for the course materials and original notebook
- **OpenAI** for the GPT models
- **Elasticsearch** for the search engine
- **Contributors** to the testing framework and CI/CD setup

---

## 🚦 Quick Start

```bash
# 1. Clone and install
git clone <repository>
cd llm-zoomcamp-1
make dev-setup

# 2. Run tests
make test

# 3. Start Elasticsearch (if testing full functionality)
docker run -d -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.15.0

# 4. Set environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# 5. Test the system
python example_usage.py
```

This RAG system transforms the original notebook into a production-ready package with comprehensive testing, CI/CD, and modular architecture suitable for real-world deployment.
