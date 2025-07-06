"""Main RAG pipeline for the system.

Orchestrates document loading, indexing, searching, and response generation.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from ..config import DEFAULT_INDEX_NAME, ELASTICSEARCH_URL, DEFAULT_COLLECTION_NAME
from ..data.loader import DocumentLoader
from ..data.vector_store import VectorSearcher
from ..formatting.context import ContextFormatter
from ..llm.openai_client import OpenAIClient
from ..models.course import Course
from ..search.elasticsearch_client import ElasticsearchClient
from ..search.query_builder import QueryBuilder

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    def __init__(
        self,
        es_url: str = ELASTICSEARCH_URL,
        index_name: str = DEFAULT_INDEX_NAME,
        collection_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
    ):
        """Initialize the RAG pipeline.

        Args:
            es_url: Elasticsearch URL
            index_name: Elasticsearch index name
            collection_name: Qdrant collection name for vector search
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use
        """
        self.index_name = index_name
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        # Initialize components
        self.document_loader = DocumentLoader()
        self.es_client = ElasticsearchClient(es_url=es_url)
        self.query_builder = QueryBuilder()
        self.context_formatter = ContextFormatter()
        self.llm_client = OpenAIClient(api_key=openai_api_key, model=openai_model)
        self.vector_searcher = VectorSearcher()

        logger.info("RAG pipeline initialized")

    def setup_index(self, load_documents: bool = True, delete_existing: bool = True) -> Dict[str, Any]:
        """Set up the Elasticsearch index and optionally load documents.

        Args:
            load_documents: Whether to load and index documents
            delete_existing: Whether to delete existing index

        Returns:
            Dictionary with setup results
        """
        logger.info("Setting up Elasticsearch index...")

        # Create the index
        self.es_client.create_index(self.index_name, delete_if_exists=delete_existing)

        results = {"index_created": True, "documents_loaded": 0, "documents_indexed": 0}

        if load_documents:
            # Load documents
            documents = self.document_loader.load_documents()
            results["documents_loaded"] = len(documents)

            # Index documents
            indexed_count = self.es_client.index_documents(documents, self.index_name)
            results["documents_indexed"] = indexed_count

            logger.info(f"Setup complete: {indexed_count} documents indexed")

        return results

    def search(
        self,
        question: str,
        course_filter: Optional[Course] = None,
        num_results: int = 5,
        boost: int = 4,
        return_raw: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for relevant documents.

        Args:
            question: The question to search for
            course_filter: Optional course filter
            num_results: Number of results to return
            boost: Boost factor for question field
            return_raw: Whether to return raw Elasticsearch response

        Returns:
            List of relevant documents
        """
        # Build the search query
        query = self.query_builder.build_search_query(
            question=question, course_filter=course_filter, num_results=num_results, boost=boost
        )

        # Execute the search
        results = self.es_client.search_documents(query, self.index_name, return_raw=return_raw)

        if return_raw:
            logger.debug(f"Search returned {len(results['hits']['hits'])} results")  # type: ignore[call-overload]
            return results
        else:
            logger.debug(f"Search returned {len(results)} results")
            return results

    def search_vector(
        self,
        question: str,
        course_filter: Optional[Course] = None,
        num_results: int = 5,
        score_threshold: Optional[float] = None,
        collection_name: Optional[str] = None,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Search for relevant documents using vector search.

        Args:
            question: The question to search for
            course_filter: Optional course filter
            num_results: Number of results to return
            score_threshold: Minimum similarity score threshold
            collection_name: Qdrant collection name (uses default if not provided)
            return_raw: Whether to return raw Qdrant response format

        Returns:
            List of relevant documents (same format as Elasticsearch search)
        """
        # Use default collection if not provided
        collection = collection_name or self.collection_name
        
        # Convert course filter to string if provided
        course_filter_str = course_filter.value if course_filter else None

        # Execute the vector search
        try:
            qdrant_results = self.vector_searcher.search(
                query=question,
                collection_name=collection,
                limit=num_results,
                course_filter=course_filter_str,
                score_threshold=score_threshold,
                with_payload=True,
            )

            # Return document list format (same as Elasticsearch)
            documents = [result["payload"] for result in qdrant_results if "payload" in result]
            logger.debug(f"Vector search returned {len(documents)} results")
            return documents

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

    def generate_response(
        self, question: str, documents: List[Dict[str, Any]], model: Optional[str] = None, include_context: bool = False
    ) -> Dict[str, Any]:
        """Generate a response using the LLM.

        Args:
            question: The user's question
            documents: Relevant documents to use as context
            model: LLM model to use
            include_context: Whether to include formatted context in response

        Returns:
            Dictionary with response and metadata
        """
        # Format the context
        context = self.context_formatter.format_context(documents)

        # Build the prompt
        prompt = self.context_formatter.build_prompt(question, context)

        # Generate response
        response = self.llm_client.get_response(prompt, model=model)

        result = {"response": response, "num_documents": len(documents), "prompt_length": len(prompt)}

        if include_context:
            result["context"] = context
            result["prompt"] = prompt

        return result

    def ask(
        self,
        question: str,
        search_engine: str,
        course_filter: Optional[Course] = None,
        num_results: int = 5,
        llm: Optional[str] = None,
        qdrant_collection_name: Optional[str] = None,
        qdrant_score_threshold: Optional[float] = None,
        elasticsearch_boost: int = 4,
        debug: bool = False,
    ) -> str:
        """Ask a question and get a response (end-to-end RAG).

        Args:
            question: The question to ask
            course_filter: Optional course filter
            num_results: Number of search results to use
            boost: Boost factor for question field
            model: LLM model to use
            debug: Whether to print debug information

        Returns:
            The generated response
        """
        if search_engine == "elasticsearch":
            search_result = self.search(question=question, course_filter=course_filter, num_results=num_results, boost=elasticsearch_boost)
        elif search_engine == "qdrant":
            search_result = self.search_vector(question=question, course_filter=course_filter, num_results=num_results, score_threshold=qdrant_score_threshold, collection_name=qdrant_collection_name)
        else:
            raise ValueError(f"Invalid search engine: {search_engine}")

        try:
            documents = search_result

            # Generate response
            result = self.generate_response(question=question, documents=documents, model=llm, include_context=debug)

            if debug:
                self._print_debug_info(question, documents, result, course_filter)

            return result["response"]  # type: ignore[no-any-return]

        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise

    def _print_debug_info(
        self, question: str, documents: List[Dict[str, Any]], result: Dict[str, Any], course_filter: Optional[Course]
    ) -> None:
        """Print debug information."""
        print(f"Question: {question}")
        print(f"Course filter: {course_filter.value if course_filter else 'None'}")
        print(f"Number of documents: {len(documents)}")
        print(f"Prompt length: {result['prompt_length']}")
        print("\n" + "=" * 50)
        print("CONTEXT:")
        print(result.get("context", "N/A"))
        print("\n" + "=" * 50)
        print("PROMPT:")
        print(result.get("prompt", "N/A"))
        print("\n" + "=" * 50)
        print("RESPONSE:")
        print(result["response"])
        print("=" * 50)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system.

        Returns:
            Dictionary with system statistics
        """
        try:
            doc_count = self.es_client.count_documents(self.index_name)
            doc_stats = self.document_loader.get_document_stats()

            return {
                "elasticsearch": {
                    "index_name": self.index_name,
                    "document_count": doc_count,
                    "index_exists": self.es_client.index_exists(self.index_name),
                },
                "qdrant": {
                    "collection_name": self.collection_name,
                    "vector_searcher_available": self.vector_searcher is not None,
                },
                "documents": doc_stats,
                "llm": {"model": self.llm_client.model},
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def health_check(self) -> Dict[str, bool]:
        """Perform a health check on all components.

        Returns:
            Dictionary with health status of each component
        """
        health = {}

        try:
            # Check Elasticsearch
            health["elasticsearch"] = self.es_client.health_check()
        except Exception:
            health["elasticsearch"] = False

        try:
            # Check Qdrant/Vector Search
            health["qdrant"] = self.vector_searcher is not None
        except Exception:
            health["qdrant"] = False

        try:
            # Check if documents are loaded
            health["documents"] = len(self.document_loader.documents) > 0
        except Exception:
            health["documents"] = False

        try:
            # Check OpenAI (basic check)
            health["openai"] = self.llm_client.api_key is not None
        except Exception:
            health["openai"] = False

        health["overall"] = all(health.values())
        return health
