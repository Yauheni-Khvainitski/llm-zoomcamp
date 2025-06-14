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
        self.assertIsNotNone(pipeline.formatter)
        self.assertIsNotNone(pipeline.openai_client)

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
            "elasticsearch_host": "http://custom:9200",
            "index_name": "custom-index",
            "openai_model": "gpt-3.5-turbo",
        }

        RAGPipeline(**custom_config)

        # Verify custom config was used
        mock_es_client.assert_called_with("http://custom:9200", "custom-index")
        mock_openai.assert_called_with(model="gpt-3.5-turbo", load_env=True)

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
        mock_es.index_documents.assert_called_once_with(self.sample_documents)
        self.assertEqual(result, 2)

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
        mock_qb.build_search_query.assert_called_once_with("What is Docker?", None, 5, 4)
        mock_es.search_documents.assert_called_once_with(mock_query)
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

        mock_qb.build_search_query.assert_called_once_with("What is Docker?", Course.DATA_ENGINEERING_ZOOMCAMP, 5, 4)
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
        mock_fmt.build_prompt_from_documents.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Test answer"
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.generate_answer("What is Docker?", self.sample_documents)

        mock_fmt.build_prompt_from_documents.assert_called_once_with("What is Docker?", self.sample_documents)
        mock_ai.get_response.assert_called_once_with("Test prompt")
        self.assertEqual(result, "Test answer")

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
        mock_fmt.build_prompt_from_documents.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_usage_response = {
            "response": "Test answer",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "gpt-4o",
            "finish_reason": "stop",
        }
        mock_ai.get_response_with_usage.return_value = mock_usage_response
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.generate_answer("What is Docker?", self.sample_documents, include_usage=True)

        mock_ai.get_response_with_usage.assert_called_once_with("Test prompt")
        self.assertEqual(result, mock_usage_response)

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
        mock_fmt.build_prompt_from_documents.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.ask_question("What is Docker?")

        # Verify the full pipeline was executed
        mock_qb.build_search_query.assert_called_once()
        mock_es.search_documents.assert_called_once()
        mock_fmt.build_prompt_from_documents.assert_called_once()
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
        mock_fmt.build_prompt_from_documents.return_value = "Test prompt"
        mock_formatter.return_value = mock_fmt

        mock_ai = Mock()
        mock_ai.get_response.return_value = "Docker is a containerization platform."
        mock_openai.return_value = mock_ai

        pipeline = RAGPipeline()
        result = pipeline.ask_question("What is Docker?", debug=True)

        # Should return debug information
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("search_query", result)
        self.assertIn("search_results", result)
        self.assertIn("prompt", result)

        self.assertEqual(result["answer"], "Docker is a containerization platform.")
        self.assertEqual(result["search_query"], mock_query)
        self.assertEqual(result["search_results"], self.sample_documents)
        self.assertEqual(result["prompt"], "Test prompt")

    @patch("rag.pipeline.rag.DocumentLoader")
    @patch("rag.pipeline.rag.ElasticsearchClient")
    @patch("rag.pipeline.rag.QueryBuilder")
    @patch("rag.pipeline.rag.ContextFormatter")
    @patch("rag.pipeline.rag.OpenAIClient")
    def test_health_check_success(self, mock_openai, mock_formatter, mock_query_builder, mock_es_client, mock_doc_loader):
        """Test successful health check."""
        mock_doc_loader.return_value = Mock()

        mock_es = Mock()
        mock_es.health_check.return_value = True
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
        self.assertIn("available_models", result)
        self.assertEqual(result["available_models"], ["gpt-4o", "gpt-3.5-turbo"])

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
        mock_es.health_check.return_value = False
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
        self.assertIn("document_stats", result)
        self.assertIn("index_document_count", result)
        self.assertEqual(result["index_document_count"], 100)
        self.assertEqual(result["document_stats"]["total_documents"], 100)


if __name__ == "__main__":
    unittest.main()
