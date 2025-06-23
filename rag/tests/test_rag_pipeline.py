"""
Tests for the RAGPipeline class.
"""

import unittest
from unittest.mock import Mock, patch

from ..models.course import Course
from ..pipeline.rag import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test suite for the RAGPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_documents = [
            {
                "doc_id": "doc1",
                "text": "Docker is a containerization platform.",
                "question": "What is Docker?",
                "section": "General",
                "course": "data-engineering-zoomcamp",
            },
            {
                "doc_id": "doc2",
                "text": "You can install Docker from the official website.",
                "question": "How to install Docker?",
                "section": "Setup",
                "course": "data-engineering-zoomcamp",
            },
        ]

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_init_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful RAGPipeline initialization."""
        # Setup mocks
        mock_doc_loader.return_value = Mock()
        mock_es_client.return_value = Mock()
        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()

        # Verify all components were initialized
        self.assertIsNotNone(pipeline.document_loader)
        self.assertIsNotNone(pipeline.es_client)
        self.assertIsNotNone(pipeline.query_builder)
        self.assertIsNotNone(pipeline.context_formatter)
        self.assertIsNotNone(pipeline.llm_client)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_init_with_custom_config(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test RAGPipeline initialization with custom configuration."""
        mock_doc_loader.return_value = Mock()
        mock_es_client.return_value = Mock()
        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        custom_config = {
            "es_url": "http://custom:9200",
            "index_name": "custom-index",
            "openai_model": "gpt-3.5-turbo",
        }

        RAGPipeline(**custom_config)

        # Verify custom config was used
        mock_es_client.assert_called_with(es_url="http://custom:9200")
        mock_openai.assert_called_with(api_key=None, model="gpt-3.5-turbo")

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_setup_index_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful index setup."""
        # Setup mocks
        mock_loader = Mock()
        mock_loader.load_documents.return_value = self.sample_documents
        mock_doc_loader.return_value = mock_loader

        mock_es = Mock()
        mock_es.create_index.return_value = True
        mock_es.index_documents.return_value = 2
        mock_es_client.return_value = mock_es

        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()
        result = pipeline.setup_index()

        # Verify setup process
        mock_loader.load_documents.assert_called_once()
        mock_es.create_index.assert_called_once()
        mock_es.index_documents.assert_called_once_with(self.sample_documents, "zoomcamp-courses-questions")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["documents_indexed"], 2)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_setup_index_failure(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test index setup failure."""
        mock_loader = Mock()
        mock_loader.load_documents.side_effect = Exception("Failed to load documents")
        mock_doc_loader.return_value = mock_loader

        mock_es_client.return_value = Mock()
        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()

        with self.assertRaises(Exception) as context:
            pipeline.setup_index()

        self.assertIn("Failed to load documents", str(context.exception))

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_search_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful document search."""
        # Setup mocks
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents
        mock_es_client.return_value = mock_es

        mock_qb = Mock()
        mock_query = {"query": {"match": {"text": "Docker"}}}
        mock_qb.build_search_query.return_value = mock_query
        mock_query_builder.return_value = mock_qb

        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()
        result = pipeline.search("What is Docker?")

        # Verify search process
        mock_qb.build_search_query.assert_called_once_with(question="What is Docker?", course_filter=None, num_results=5, boost=4)
        mock_es.search_documents.assert_called_once_with(mock_query, "zoomcamp-courses-questions", return_raw=False)
        self.assertEqual(result, self.sample_documents)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_search_with_course_filter(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test document search with course filter."""
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents
        mock_es_client.return_value = mock_es

        mock_qb = Mock()
        mock_query = {"query": {"bool": {"must": {}, "filter": {}}}}
        mock_qb.build_search_query.return_value = mock_query
        mock_query_builder.return_value = mock_qb

        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()
        result = pipeline.search("What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP)

        mock_qb.build_search_query.assert_called_once_with(question="What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=5, boost=4)
        self.assertEqual(result, self.sample_documents)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_generate_answer_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful answer generation."""
        mock_doc_loader.return_value = Mock()
        mock_es_client.return_value = Mock()
        mock_query_builder.return_value = Mock()

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Test answer"
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.generate_response("What is Docker?", self.sample_documents)

        mock_fmt.format_context.assert_called_once_with(self.sample_documents)
        mock_fmt.build_prompt.assert_called_once_with("What is Docker?", "Test context")
        mock_ai.get_response.assert_called_once_with("Test prompt", model=None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["response"], "Test answer")

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_generate_answer_with_usage(
        self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader
    ):
        """Test answer generation with usage information."""
        mock_doc_loader.return_value = Mock()
        mock_es_client.return_value = Mock()
        mock_query_builder.return_value = Mock()

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Test answer"
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.generate_response("What is Docker?", self.sample_documents, include_context=True)

        mock_ai.get_response.assert_called_once_with("Test prompt", model=None)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["response"], "Test answer")
        self.assertIn("context", result)
        self.assertIn("prompt", result)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_ask_question_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful end-to-end question answering."""
        # Setup all mocks
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents
        mock_es_client.return_value = mock_es

        mock_qb = Mock()
        mock_qb.build_search_query.return_value = {"query": {"match": {"text": "Docker"}}}
        mock_query_builder.return_value = mock_qb

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.ask("What is Docker?")

        # Verify the full pipeline was executed
        mock_qb.build_search_query.assert_called_once()
        mock_es.search_documents.assert_called_once()
        mock_fmt.format_context.assert_called_once()
        mock_fmt.build_prompt.assert_called_once()
        mock_ai.get_response.assert_called_once()

        self.assertEqual(result, "Docker is a containerization platform.")

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_ask_question_with_debug(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test question answering with debug information."""
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.search_documents.return_value = self.sample_documents
        mock_es_client.return_value = mock_es

        mock_qb = Mock()
        mock_query = {"query": {"match": {"text": "Docker"}}}
        mock_qb.build_search_query.return_value = mock_query
        mock_query_builder.return_value = mock_qb

        mock_fmt = Mock()
        mock_fmt.format_context.return_value = "Test context"
        mock_fmt.build_prompt.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.ask("What is Docker?", debug=True)

        # Should return the response string
        self.assertIsInstance(result, str)
        self.assertEqual(result, "Docker is a containerization platform.")

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_health_check_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful health check."""
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.index_exists.return_value = True
        mock_es_client.return_value = mock_es

        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()

        mock_ai = Mock()
        mock_ai.list_available_models.return_value = ["gpt-4o", "gpt-3.5-turbo"]
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.health_check()

        self.assertIsInstance(result, dict)
        self.assertTrue(result["elasticsearch"])
        self.assertTrue(result["openai"])
        # Health check doesn't include available_models in current implementation
        self.assertIn("overall", result)

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_health_check_elasticsearch_failure(
        self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader
    ):
        """Test health check with Elasticsearch failure."""
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.index_exists.return_value = False
        mock_es_client.return_value = mock_es

        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()

        mock_ai = Mock()
        mock_ai.list_available_models.return_value = ["gpt-4o"]
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.health_check()

        self.assertFalse(result["elasticsearch"])
        self.assertTrue(result["openai"])

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_get_stats(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test getting pipeline statistics."""
        mock_loader = Mock()
        mock_loader.get_document_stats.return_value = {
            "total_documents": 100,
            "unique_courses": 4,
            "documents_by_course": {"course1": 25, "course2": 75},
        }
        mock_doc_loader.return_value = mock_loader

        mock_es = Mock()
        mock_es.count_documents.return_value = 100
        mock_es_client.return_value = mock_es

        mock_query_builder.return_value = Mock()
        mock_formatter.return_value = Mock()
        mock_openai.return_value = Mock()

        pipeline = RAGPipeline()
        result = pipeline.get_stats()

        self.assertIsInstance(result, dict)
        self.assertIn("documents", result)
        self.assertIn("elasticsearch", result)
        self.assertEqual(result["elasticsearch"]["document_count"], 100)
        self.assertEqual(result["documents"]["total_documents"], 100)


if __name__ == "__main__":
    unittest.main()
