"""
Tests for the ElasticsearchClient class.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest

from ..search.elasticsearch_client import ElasticsearchClient


class TestElasticsearchClient(unittest.TestCase):
    """Test suite for the ElasticsearchClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.host = "http://localhost:9200"
        self.index_name = "test-index"
        self.sample_documents = [
            {
                "doc_id": "doc1",
                "text": "Sample text 1",
                "question": "What is Docker?",
                "section": "General",
                "course": "data-engineering-zoomcamp",
            },
            {
                "doc_id": "doc2",
                "text": "Sample text 2",
                "question": "How to install Docker?",
                "section": "Setup",
                "course": "data-engineering-zoomcamp",
            },
        ]

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_init_success(self, mock_elasticsearch):
        """Test successful initialization."""
        mock_es = Mock()
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)

        self.assertEqual(client.host, self.host)
        self.assertEqual(client.index_name, self.index_name)
        mock_elasticsearch.assert_called_once_with(hosts=self.host)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_init_connection_error(self, mock_elasticsearch):
        """Test initialization with connection error."""
        mock_elasticsearch.side_effect = Exception("Connection failed")

        with self.assertRaises(Exception) as context:
            ElasticsearchClient(self.host, self.index_name)

        self.assertIn("Connection failed", str(context.exception))

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_create_index_success(self, mock_elasticsearch):
        """Test successful index creation."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = False
        mock_es.indices.create.return_value = {"acknowledged": True}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.create_index()

        self.assertTrue(result)
        mock_es.indices.exists.assert_called_once_with(index=self.index_name)
        mock_es.indices.create.assert_called_once()

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_create_index_already_exists(self, mock_elasticsearch):
        """Test index creation when index already exists."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = True
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.create_index()

        self.assertTrue(result)
        mock_es.indices.exists.assert_called_once_with(index=self.index_name)
        mock_es.indices.create.assert_not_called()

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_create_index_error(self, mock_elasticsearch):
        """Test index creation with error."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = False
        mock_es.indices.create.side_effect = Exception("Index creation failed")
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)

        with self.assertRaises(Exception) as context:
            client.create_index()

        self.assertIn("Index creation failed", str(context.exception))

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_delete_index_success(self, mock_elasticsearch):
        """Test successful index deletion."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = True
        mock_es.indices.delete.return_value = {"acknowledged": True}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.delete_index()

        self.assertTrue(result)
        mock_es.indices.exists.assert_called_once_with(index=self.index_name)
        mock_es.indices.delete.assert_called_once_with(index=self.index_name)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_delete_index_not_exists(self, mock_elasticsearch):
        """Test index deletion when index doesn't exist."""
        mock_es = Mock()
        mock_es.indices.exists.return_value = False
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.delete_index()

        self.assertTrue(result)
        mock_es.indices.exists.assert_called_once_with(index=self.index_name)
        mock_es.indices.delete.assert_not_called()

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_index_document_success(self, mock_elasticsearch):
        """Test successful document indexing."""
        mock_es = Mock()
        mock_es.index.return_value = {"result": "created"}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        doc = self.sample_documents[0]
        result = client.index_document(doc)

        self.assertTrue(result)
        mock_es.index.assert_called_once_with(index=self.index_name, id=doc["doc_id"], document=doc)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_index_document_error(self, mock_elasticsearch):
        """Test document indexing with error."""
        mock_es = Mock()
        mock_es.index.side_effect = Exception("Indexing failed")
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        doc = self.sample_documents[0]

        with self.assertRaises(Exception) as context:
            client.index_document(doc)

        self.assertIn("Indexing failed", str(context.exception))

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_index_documents_success(self, mock_elasticsearch):
        """Test successful bulk document indexing."""
        mock_es = Mock()
        mock_es.index.return_value = {"result": "created"}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.index_documents(self.sample_documents)

        self.assertEqual(result, 2)  # Should return count of indexed documents
        self.assertEqual(mock_es.index.call_count, 2)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_index_documents_partial_failure(self, mock_elasticsearch):
        """Test bulk document indexing with partial failures."""
        mock_es = Mock()
        # First document succeeds, second fails
        mock_es.index.side_effect = [{"result": "created"}, Exception("Failed")]
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.index_documents(self.sample_documents)

        self.assertEqual(result, 1)  # Should return count of successfully indexed documents
        self.assertEqual(mock_es.index.call_count, 2)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_search_documents_success(self, mock_elasticsearch):
        """Test successful document search."""
        mock_es = Mock()
        mock_response = {"hits": {"hits": [{"_source": self.sample_documents[0]}, {"_source": self.sample_documents[1]}]}}
        mock_es.search.return_value = mock_response
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        query = {"query": {"match_all": {}}}
        result = client.search_documents(query)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], self.sample_documents[0])
        self.assertEqual(result[1], self.sample_documents[1])
        mock_es.search.assert_called_once_with(index=self.index_name, body=query)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_search_documents_no_results(self, mock_elasticsearch):
        """Test document search with no results."""
        mock_es = Mock()
        mock_response = {"hits": {"hits": []}}
        mock_es.search.return_value = mock_response
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        query = {"query": {"match": {"text": "nonexistent"}}}
        result = client.search_documents(query)

        self.assertEqual(len(result), 0)
        self.assertEqual(result, [])

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_search_documents_error(self, mock_elasticsearch):
        """Test document search with error."""
        mock_es = Mock()
        mock_es.search.side_effect = Exception("Search failed")
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        query = {"query": {"match_all": {}}}

        with self.assertRaises(Exception) as context:
            client.search_documents(query)

        self.assertIn("Search failed", str(context.exception))

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_get_document_success(self, mock_elasticsearch):
        """Test successful document retrieval by ID."""
        mock_es = Mock()
        mock_response = {"_source": self.sample_documents[0]}
        mock_es.get.return_value = mock_response
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.get_document("doc1")

        self.assertEqual(result, self.sample_documents[0])
        mock_es.get.assert_called_once_with(index=self.index_name, id="doc1")

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_get_document_not_found(self, mock_elasticsearch):
        """Test document retrieval when document not found."""
        from elasticsearch.exceptions import NotFoundError

        mock_es = Mock()
        mock_es.get.side_effect = NotFoundError("Document not found")
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.get_document("nonexistent")

        self.assertIsNone(result)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_delete_document_success(self, mock_elasticsearch):
        """Test successful document deletion."""
        mock_es = Mock()
        mock_es.delete.return_value = {"result": "deleted"}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.delete_document("doc1")

        self.assertTrue(result)
        mock_es.delete.assert_called_once_with(index=self.index_name, id="doc1")

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_delete_document_not_found(self, mock_elasticsearch):
        """Test document deletion when document not found."""
        from elasticsearch.exceptions import NotFoundError

        mock_es = Mock()
        mock_es.delete.side_effect = NotFoundError("Document not found")
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.delete_document("nonexistent")

        self.assertFalse(result)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_count_documents_success(self, mock_elasticsearch):
        """Test successful document counting."""
        mock_es = Mock()
        mock_es.count.return_value = {"count": 100}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.count_documents()

        self.assertEqual(result, 100)
        mock_es.count.assert_called_once_with(index=self.index_name)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_count_documents_with_query(self, mock_elasticsearch):
        """Test document counting with query."""
        mock_es = Mock()
        mock_es.count.return_value = {"count": 50}
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        query = {"query": {"match": {"course": "data-engineering-zoomcamp"}}}
        result = client.count_documents(query)

        self.assertEqual(result, 50)
        mock_es.count.assert_called_once_with(index=self.index_name, body=query)

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_health_check_success(self, mock_elasticsearch):
        """Test successful health check."""
        mock_es = Mock()
        mock_es.ping.return_value = True
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.health_check()

        self.assertTrue(result)
        mock_es.ping.assert_called_once()

    @patch("rag.search.elasticsearch_client.Elasticsearch")
    def test_health_check_failure(self, mock_elasticsearch):
        """Test health check failure."""
        mock_es = Mock()
        mock_es.ping.return_value = False
        mock_elasticsearch.return_value = mock_es

        client = ElasticsearchClient(self.host, self.index_name)
        result = client.health_check()

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
