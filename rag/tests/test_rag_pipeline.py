"""
Tests for the RAGPipeline class.
"""

import unittest
from unittest.mock import Mock

from ..models.course import Course
from ..pipeline.rag import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAGPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_documents = [
            {
                "text": "Docker is a containerization platform",
                "section": "Introduction",
                "question": "What is Docker?",
                "course": "data-engineering-zoomcamp",
                "doc_id": "docker_001",
            },
            {
                "text": "Containers are lightweight",
                "section": "Containers",
                "question": "What are containers?",
                "course": "data-engineering-zoomcamp",
                "doc_id": "docker_002",
            },
        ]

    def create_mocked_pipeline(self, **component_overrides):
        """Create a RAGPipeline with mocked components.

        Args:
            **component_overrides: Optional overrides for specific components

        Returns:
            RAGPipeline instance with mocked components
        """
        # Create default mocked components
        mock_document_loader = Mock()
        mock_es_client = Mock()
        mock_query_builder = Mock()
        mock_context_formatter = Mock()
        mock_llm_client = Mock()
        mock_vector_searcher = Mock()

        # Apply any overrides
        components = {
            "document_loader": component_overrides.get("document_loader", mock_document_loader),
            "es_client": component_overrides.get("es_client", mock_es_client),
            "query_builder": component_overrides.get("query_builder", mock_query_builder),
            "context_formatter": component_overrides.get("context_formatter", mock_context_formatter),
            "llm_client": component_overrides.get("llm_client", mock_llm_client),
            "vector_searcher": component_overrides.get("vector_searcher", mock_vector_searcher),
        }

        return RAGPipeline(**components)

    def test_init_success(self):
        """Test successful initialization."""
        pipeline = self.create_mocked_pipeline()

        self.assertIsNotNone(pipeline.document_loader)
        self.assertIsNotNone(pipeline.es_client)
        self.assertIsNotNone(pipeline.query_builder)
        self.assertIsNotNone(pipeline.context_formatter)
        self.assertIsNotNone(pipeline.llm_client)
        self.assertIsNotNone(pipeline.vector_searcher)
        self.assertEqual(pipeline.index_name, "zoomcamp-courses-questions")

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        pipeline = self.create_mocked_pipeline()
        pipeline.index_name = "custom-index"
        pipeline.collection_name = "custom-collection"

        self.assertEqual(pipeline.index_name, "custom-index")
        self.assertEqual(pipeline.collection_name, "custom-collection")

    def test_setup_index_success(self):
        """Test successful index setup."""
        # Create mocked components
        mock_loader = Mock()
        mock_loader.load_documents.return_value = self.sample_documents

        mock_es = Mock()
        mock_es.create_index.return_value = True
        mock_es.index_documents.return_value = 2

        # Create pipeline with injected mocks
        pipeline = self.create_mocked_pipeline(document_loader=mock_loader, es_client=mock_es)

        result = pipeline.setup_index()

        # Verify setup process
        mock_loader.load_documents.assert_called_once()
        mock_es.create_index.assert_called_once()
        mock_es.index_documents.assert_called_once_with(self.sample_documents, "zoomcamp-courses-questions")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["documents_indexed"], 2)

    def test_setup_index_failure(self):
        """Test index setup failure."""
        mock_loader = Mock()
        mock_loader.load_documents.side_effect = Exception("Failed to load documents")

        pipeline = self.create_mocked_pipeline(document_loader=mock_loader)

        with self.assertRaises(Exception) as context:
            pipeline.setup_index()

        self.assertIn("Failed to load documents", str(context.exception))

    def test_search_success(self):
        """Test successful document search."""
        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents

        mock_qb = Mock()
        mock_query = {"query": {"match": {"text": "Docker"}}}
        mock_qb.build_search_query.return_value = mock_query

        pipeline = self.create_mocked_pipeline(es_client=mock_es, query_builder=mock_qb)

        result = pipeline.search("What is Docker?")

        # Verify search process
        mock_qb.build_search_query.assert_called_once_with(
            question="What is Docker?", course_filter=None, num_results=5, boost=4
        )
        mock_es.search_documents.assert_called_once_with(mock_query, "zoomcamp-courses-questions", return_raw=False)
        self.assertEqual(result, self.sample_documents)

    def test_search_with_course_filter(self):
        """Test document search with course filter."""
        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents

        mock_qb = Mock()
        mock_query = {"query": {"bool": {"must": {}, "filter": {}}}}
        mock_qb.build_search_query.return_value = mock_query

        pipeline = self.create_mocked_pipeline(es_client=mock_es, query_builder=mock_qb)

        result = pipeline.search("What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP)

        mock_qb.build_search_query.assert_called_once_with(
            question="What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=5, boost=4
        )
        self.assertEqual(result, self.sample_documents)

    def test_search_with_string_course_filter(self):
        """Test search with string course filter."""
        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents

        mock_qb = Mock()
        mock_query = {"test": "query"}
        mock_qb.build_search_query.return_value = mock_query

        pipeline = self.create_mocked_pipeline(es_client=mock_es, query_builder=mock_qb)

        result = pipeline.search("What is Docker?", course_filter="data-engineering-zoomcamp")

        # Verify that string course filter is passed correctly to query builder
        mock_qb.build_search_query.assert_called_once_with(
            question="What is Docker?", course_filter="data-engineering-zoomcamp", num_results=5, boost=4
        )
        self.assertEqual(result, self.sample_documents)

    def test_generate_answer_success(self):
        """Test successful answer generation."""
        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Test answer"

        pipeline = self.create_mocked_pipeline(context_formatter=mock_fmt, llm_client=mock_ai)

        result = pipeline.generate_response("What is Docker?", self.sample_documents)

        mock_fmt.format_context.assert_called_once_with(self.sample_documents)
        mock_fmt.build_prompt.assert_called_once_with("What is Docker?", "Test context")
        mock_ai.get_response.assert_called_once_with("Test prompt", model=None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["response"], "Test answer")

    def test_generate_answer_with_usage(self):
        """Test answer generation with usage information."""
        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Test answer"

        pipeline = self.create_mocked_pipeline(context_formatter=mock_fmt, llm_client=mock_ai)

        result = pipeline.generate_response("What is Docker?", self.sample_documents, include_context=True)

        mock_ai.get_response.assert_called_once_with("Test prompt", model=None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["response"], "Test answer")
        self.assertIn("context", result)
        self.assertIn("prompt", result)

    def test_ask_question_elasticsearch_success(self):
        """Test successful end-to-end question answering with Elasticsearch."""
        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents

        mock_qb = Mock()
        mock_qb.build_search_query.return_value = {"query": {"match": {"text": "Docker"}}}

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."

        pipeline = self.create_mocked_pipeline(
            es_client=mock_es, query_builder=mock_qb, context_formatter=mock_fmt, llm_client=mock_ai
        )

        result = pipeline.ask("What is Docker?", search_engine="elasticsearch")

        # Verify the full pipeline was executed
        mock_qb.build_search_query.assert_called_once()
        mock_es.search_documents.assert_called_once()
        mock_fmt.format_context.assert_called_once()
        mock_fmt.build_prompt.assert_called_once()
        mock_ai.get_response.assert_called_once()

        self.assertEqual(result, "Docker is a containerization platform.")

    def test_ask_question_qdrant_success(self):
        """Test successful end-to-end question answering with Qdrant."""
        # Mock vector search results (Qdrant format)
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = mock_qdrant_results

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."

        pipeline = self.create_mocked_pipeline(
            vector_searcher=mock_vector_searcher, context_formatter=mock_fmt, llm_client=mock_ai
        )

        result = pipeline.ask("What is Docker?", search_engine="qdrant")

        # Verify the full pipeline was executed
        mock_vector_searcher.search.assert_called_once()
        mock_fmt.format_context.assert_called_once()
        mock_fmt.build_prompt.assert_called_once()
        mock_ai.get_response.assert_called_once()

        self.assertEqual(result, "Docker is a containerization platform.")

    def test_ask_question_elasticsearch_with_debug(self):
        """Test debug mode with Elasticsearch."""
        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents

        mock_qb = Mock()
        mock_qb.build_search_query.return_value = {"query": {"match": {"text": "Docker"}}}

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."

        pipeline = self.create_mocked_pipeline(
            es_client=mock_es, query_builder=mock_qb, context_formatter=mock_fmt, llm_client=mock_ai
        )

        result = pipeline.ask("What is Docker?", search_engine="elasticsearch", debug=True)

        # Should return the response string
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Docker is a containerization platform.")

    def test_ask_question_qdrant_with_debug(self):
        """Test debug mode with Qdrant."""
        # Mock vector search results (Qdrant format)
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = mock_qdrant_results

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."

        pipeline = self.create_mocked_pipeline(
            vector_searcher=mock_vector_searcher, context_formatter=mock_fmt, llm_client=mock_ai
        )

        result = pipeline.ask("What is Docker?", search_engine="qdrant", debug=True)

        # Should return the response string
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Docker is a containerization platform.")

    def test_ask_question_invalid_search_engine(self):
        """Test error handling for invalid search engine."""
        pipeline = self.create_mocked_pipeline()

        with self.assertRaises(ValueError) as context:
            pipeline.ask("What is Docker?", search_engine="invalid")

        self.assertIn("Invalid search engine: invalid", str(context.exception))

    def test_health_check_success(self):
        """Test successful health check."""
        mock_es = Mock()
        mock_es.health_check.return_value = True

        mock_ai = Mock()
        mock_ai.api_key = "test-key"

        mock_loader = Mock()
        mock_loader.documents = ["doc1", "doc2"]

        mock_vector_searcher = Mock()

        pipeline = self.create_mocked_pipeline(
            es_client=mock_es, llm_client=mock_ai, document_loader=mock_loader, vector_searcher=mock_vector_searcher
        )

        result = pipeline.health_check()

        self.assertTrue(result["elasticsearch"])
        self.assertTrue(result["openai"])
        self.assertTrue(result["qdrant"])
        self.assertTrue(result["documents"])
        self.assertTrue(result["overall"])

    def test_health_check_elasticsearch_failure(self):
        """Test health check with Elasticsearch failure."""
        mock_es = Mock()
        mock_es.health_check.return_value = False

        mock_ai = Mock()
        mock_ai.api_key = "test-key"

        mock_loader = Mock()
        mock_loader.documents = ["doc1", "doc2"]

        mock_vector_searcher = Mock()

        pipeline = self.create_mocked_pipeline(
            es_client=mock_es, llm_client=mock_ai, document_loader=mock_loader, vector_searcher=mock_vector_searcher
        )

        result = pipeline.health_check()

        self.assertFalse(result["elasticsearch"])
        self.assertTrue(result["openai"])
        self.assertTrue(result["qdrant"])
        self.assertTrue(result["documents"])
        self.assertFalse(result["overall"])

    def test_get_stats(self):
        """Test getting system statistics."""
        mock_es = Mock()
        mock_es.count_documents.return_value = 42
        mock_es.index_exists.return_value = True

        mock_ai = Mock()
        mock_ai.model = "gpt-4o"

        mock_loader = Mock()
        mock_loader.get_document_stats.return_value = {"total_documents": 100, "unique_courses": 4}

        mock_vector_searcher = Mock()

        pipeline = self.create_mocked_pipeline(
            es_client=mock_es, llm_client=mock_ai, document_loader=mock_loader, vector_searcher=mock_vector_searcher
        )

        result = pipeline.get_stats()

        self.assertIsInstance(result, dict)
        self.assertIn("elasticsearch", result)
        self.assertIn("qdrant", result)
        self.assertIn("documents", result)
        self.assertIn("llm", result)
        self.assertEqual(result["elasticsearch"]["document_count"], 42)
        self.assertEqual(result["qdrant"]["collection_name"], "zoomcamp-courses-questions")
        self.assertTrue(result["qdrant"]["vector_searcher_available"])

    def test_search_vector_success(self):
        """Test vector search functionality."""
        # Mock VectorSearcher.search to return Qdrant-style results
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = mock_qdrant_results

        pipeline = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher)

        # Test normal response
        result = pipeline.search_vector("What is Docker?")

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Docker is a containerization platform")
        self.assertEqual(result[0]["course"], "docker")

    def test_search_vector_empty_results(self):
        """Test vector search with empty results."""
        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = []

        pipeline = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher)

        result = pipeline.search_vector("What is unknown?")

        self.assertEqual(len(result), 0)

    def test_search_vector_with_parameters(self):
        """Test vector search with parameters."""
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = mock_qdrant_results

        pipeline = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher)

        result = pipeline.search_vector(
            question="What is Docker?",
            course_filter=Course.DATA_ENGINEERING_ZOOMCAMP,
            num_results=3,
            score_threshold=0.8,
            collection_name="test-collection",
        )

        # Verify VectorSearcher.search was called with correct parameters
        mock_vector_searcher.search.assert_called_once_with(
            query="What is Docker?",
            collection_name="test-collection",
            limit=3,
            course_filter="data-engineering-zoomcamp",
            score_threshold=0.8,
            with_payload=True,
        )

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Docker is a containerization platform")

    def test_search_vector_with_string_course_filter(self):
        """Test vector search with string course filter."""
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher = Mock()
        mock_vector_searcher.search.return_value = mock_qdrant_results

        pipeline = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher)

        result = pipeline.search_vector(
            question="What is Docker?",
            course_filter="data-engineering-zoomcamp",
            num_results=3,
            score_threshold=0.8,
            collection_name="test-collection",
        )

        # Verify VectorSearcher.search was called with correct parameters
        mock_vector_searcher.search.assert_called_once_with(
            query="What is Docker?",
            collection_name="test-collection",
            limit=3,
            course_filter="data-engineering-zoomcamp",
            score_threshold=0.8,
            with_payload=True,
        )

        # Verify result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Docker is a containerization platform")

    def test_search_vector_course_filter_equivalence(self):
        """Test that Course enum and string produce equivalent vector search results."""
        mock_qdrant_results = [
            {"id": "doc1", "score": 0.95, "payload": {"text": "Docker is a containerization platform", "course": "docker"}}
        ]

        mock_vector_searcher_enum = Mock()
        mock_vector_searcher_enum.search.return_value = mock_qdrant_results

        mock_vector_searcher_string = Mock()
        mock_vector_searcher_string.search.return_value = mock_qdrant_results

        # Test with Course enum
        pipeline_enum = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher_enum)
        result_enum = pipeline_enum.search_vector(
            question="What is Docker?",
            course_filter=Course.MACHINE_LEARNING_ZOOMCAMP,
            num_results=3,
        )

        # Test with equivalent string
        pipeline_string = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher_string)
        result_string = pipeline_string.search_vector(
            question="What is Docker?",
            course_filter="machine-learning-zoomcamp",
            num_results=3,
        )

        # Both should call vector searcher with the same course_filter string
        mock_vector_searcher_enum.search.assert_called_once()
        mock_vector_searcher_string.search.assert_called_once()

        enum_call_args = mock_vector_searcher_enum.search.call_args
        string_call_args = mock_vector_searcher_string.search.call_args

        # Verify both calls used the same string course filter
        self.assertEqual(enum_call_args[1]["course_filter"], "machine-learning-zoomcamp")
        self.assertEqual(string_call_args[1]["course_filter"], "machine-learning-zoomcamp")

        # Results should be identical
        self.assertEqual(result_enum, result_string)

    def test_vector_search_error_handling(self):
        """Test error handling in vector search."""
        mock_vector_searcher = Mock()
        mock_vector_searcher.search.side_effect = Exception("Vector search failed")

        pipeline = self.create_mocked_pipeline(vector_searcher=mock_vector_searcher)

        with self.assertRaises(Exception) as context:
            pipeline.search_vector("What is Docker?")

        self.assertIn("Vector search failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
