"""Configuration settings for the RAG system."""

import os
from typing import Any, Dict

# Elasticsearch Configuration
ELASTICSEARCH_URL = "http://localhost:9200"
DEFAULT_INDEX_NAME = "zoomcamp-courses-questions"

# Qdrant Configuration
QDRANT_URL = "http://localhost:6333"

# Embedding Configuration
EMBEDDING_DIMENSIONALITY = 512
EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"  # Default embedding model

# Document Source Configuration
DOCUMENTS_URL = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1"

# Search Configuration
DEFAULT_NUM_RESULTS = 5
DEFAULT_BOOST_FACTOR = 4

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o"

# Elasticsearch Index Settings
INDEX_SETTINGS: Dict[str, Any] = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
        },
    },
}


# Environment Variables
def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variables."""
    return os.getenv("OPENAI_API_KEY", "")


def get_env_file_path() -> str:
    """Get the path to the .env file."""
    return os.getenv("ENV_FILE_PATH", ".env")
