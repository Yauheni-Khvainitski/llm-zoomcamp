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
        mock_qb.build_search_query.assert_called_once_with(
            question="What is Docker?", course_filter=None, num_results=5, boost=4
        )
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

        mock_qb.build_search_query.assert_called_once_with(
            question="What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=5, boost=4
        )
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
        self.assertTrue(result["qdrant"])
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
        self.assertTrue(result["qdrant"])
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
        self.assertIn("qdrant", result)
        self.assertEqual(result["elasticsearch"]["document_count"], 100)
        self.assertEqual(result["documents"]["total_documents"], 100)
        self.assertEqual(result["qdrant"]["collection_name"], "zoomcamp-courses-questions")
        self.assertTrue(result["qdrant"]["vector_searcher_available"])

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_ask_with_details_success(self, mock_vector_searcher):
        """Test successful ask_with_details operation."""
        # Create a RAG pipeline instance with mocked components
        rag_pipeline = RAGPipeline()
        
        # Mock the search method to return raw response
        rag_pipeline.search = Mock(
            return_value={
                "hits": {
                    "total": {"value": 2},
                    "max_score": 0.95,
                    "hits": [
                        {"_source": {"text": "Docker is a containerization platform"}},
                        {"_source": {"text": "Containers are lightweight"}},
                    ],
                }
            }
        )

        # Mock the generate_response method
        rag_pipeline.generate_response = Mock(
            return_value={
                "response": "Docker is a containerization platform that uses containers.",
                "context": "Docker is a containerization platform\nContainers are lightweight",
                "prompt": "Based on the context...",
            }
        )

        result = rag_pipeline.ask_with_details("What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP)

        # Verify the result structure
        self.assertEqual(result["question"], "What is Docker?")
        self.assertEqual(result["response"], "Docker is a containerization platform that uses containers.")
        self.assertEqual(result["search_results"]["total_hits"], 2)
        self.assertEqual(result["search_results"]["max_score"], 0.95)
        self.assertEqual(len(result["search_results"]["documents"]), 2)
        self.assertEqual(result["metadata"]["course_filter"], "data-engineering-zoomcamp")

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_search_vector_success(self, mock_vector_searcher):
        """Test successful vector search operation."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        # Mock VectorSearcher.search to return Qdrant-style results
        mock_qdrant_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {"text": "Docker is a containerization platform", "course": "docker"}
            },
            {
                "id": "doc2", 
                "score": 0.85,
                "payload": {"text": "Containers are lightweight", "course": "docker"}
            }
        ]
        
        rag_pipeline.vector_searcher.search = Mock(return_value=mock_qdrant_results)

        # Test non-raw response
        result = rag_pipeline.search_vector("What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "Docker is a containerization platform")
        self.assertEqual(result[1]["text"], "Containers are lightweight")

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_search_vector_raw_response(self, mock_vector_searcher):
        """Test vector search with raw response format."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        # Mock VectorSearcher.search to return Qdrant-style results
        mock_qdrant_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {"text": "Docker is a containerization platform", "course": "docker"}
            }
        ]
        
        rag_pipeline.vector_searcher.search = Mock(return_value=mock_qdrant_results)

        # Test raw response
        result = rag_pipeline.search_vector("What is Docker?", return_raw=True)
        
        self.assertIn("hits", result)
        self.assertEqual(result["hits"]["total"]["value"], 1)
        self.assertEqual(result["hits"]["max_score"], 0.95)
        self.assertEqual(len(result["hits"]["hits"]), 1)
        self.assertEqual(result["hits"]["hits"][0]["_id"], "doc1")
        self.assertEqual(result["hits"]["hits"][0]["_score"], 0.95)

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_search_vector_empty_results(self, mock_vector_searcher):
        """Test vector search with empty results."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        rag_pipeline.vector_searcher.search = Mock(return_value=[])

        result = rag_pipeline.search_vector("Non-existent question")
        
        self.assertEqual(result, [])

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_search_vector_with_parameters(self, mock_vector_searcher):
        """Test vector search with all parameters."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        mock_qdrant_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {"text": "Docker info", "course": "docker"}
            }
        ]
        
        rag_pipeline.vector_searcher.search = Mock(return_value=mock_qdrant_results)

        result = rag_pipeline.search_vector(
            question="What is Docker?",
            course_filter=Course.DATA_ENGINEERING_ZOOMCAMP,
            num_results=3,
            score_threshold=0.8,
            collection_name="test-collection"
        )
        
        # Verify VectorSearcher.search was called with correct parameters
        rag_pipeline.vector_searcher.search.assert_called_once_with(
            query="What is Docker?",
            collection_name="test-collection",
            limit=3,
            course_filter="data-engineering-zoomcamp",
            score_threshold=0.8,
            with_payload=True
        )

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_ask_vector_success(self, mock_vector_searcher):
        """Test successful ask_vector operation."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        # Mock vector search
        mock_qdrant_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {"text": "Docker is a containerization platform", "course": "docker"}
            }
        ]
        
        rag_pipeline.vector_searcher.search = Mock(return_value=mock_qdrant_results)

        # Mock response generation
        rag_pipeline.generate_response = Mock(
            return_value={"response": "Docker is a containerization platform."}
        )

        result = rag_pipeline.ask_vector("What is Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP)

        self.assertEqual(result, "Docker is a containerization platform.")

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_ask_vector_with_details_success(self, mock_vector_searcher):
        """Test successful ask_vector_with_details operation."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        # Mock vector search with raw response
        mock_qdrant_results = [
            {
                "id": "doc1",
                "score": 0.95,
                "payload": {"text": "Docker is a containerization platform", "course": "docker"}
            }
        ]
        
        rag_pipeline.vector_searcher.search = Mock(return_value=mock_qdrant_results)

        # Mock the search_vector method to return raw response when called with return_raw=True
        raw_response = {
            "hits": {
                "total": {"value": 1},
                "max_score": 0.95,
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 0.95,
                        "_source": {"text": "Docker is a containerization platform", "course": "docker"}
                    }
                ]
            }
        }
        
        # Mock search_vector to return raw response
        rag_pipeline.search_vector = Mock(return_value=raw_response)

        # Mock response generation
        rag_pipeline.generate_response = Mock(
            return_value={
                "response": "Docker is a containerization platform.",
                "context": "Docker is a containerization platform",
                "prompt": "Based on the context..."
            }
        )

        result = rag_pipeline.ask_vector_with_details(
            "What is Docker?",
            course_filter=Course.DATA_ENGINEERING_ZOOMCAMP,
            score_threshold=0.8,
            collection_name="test-collection"
        )

        # Verify the result structure
        self.assertEqual(result["question"], "What is Docker?")
        self.assertEqual(result["response"], "Docker is a containerization platform.")
        self.assertEqual(result["search_results"]["total_hits"], 1)
        self.assertEqual(result["search_results"]["max_score"], 0.95)
        self.assertEqual(len(result["search_results"]["documents"]), 1)
        self.assertEqual(result["metadata"]["course_filter"], "data-engineering-zoomcamp")
        self.assertEqual(result["metadata"]["score_threshold"], 0.8)
        self.assertEqual(result["metadata"]["collection_name"], "test-collection")
        self.assertEqual(result["metadata"]["search_type"], "vector")

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_vector_search_error_handling(self, mock_vector_searcher):
        """Test error handling in vector search."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        rag_pipeline.vector_searcher.search = Mock(side_effect=Exception("Vector search failed"))

        with self.assertRaises(Exception) as context:
            rag_pipeline.search_vector("What is Docker?")

        self.assertIn("Vector search failed", str(context.exception))

    @patch("rag.pipeline.rag.VectorSearcher")
    def test_ask_vector_error_handling(self, mock_vector_searcher):
        """Test error handling in ask_vector."""
        # Create a RAG pipeline instance
        rag_pipeline = RAGPipeline()
        
        rag_pipeline.vector_searcher.search = Mock(side_effect=Exception("Vector search failed"))

        with self.assertRaises(Exception) as context:
            rag_pipeline.ask_vector("What is Docker?")

        self.assertIn("Vector search failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
