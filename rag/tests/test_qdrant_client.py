"""
Tests for the QdrantClientCustom class.
"""

import unittest
from unittest.mock import Mock, patch

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse

from ..config import QDRANT_URL
from ..search.qdrant_client_custom import QdrantClientCustom


class TestQdrantClient(unittest.TestCase):
    """Test suite for the QdrantClientCustom class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use config values for standard tests
        self.qdrant_url = QDRANT_URL

        # For error testing
        self.invalid_url = "http://invalid-host:9999"

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_success(self, mock_qdrant_client):
        """Test successful initialization."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        client = QdrantClientCustom(self.qdrant_url)

        self.assertEqual(client.qdrant_url, self.qdrant_url)
        self.assertEqual(client.qdrant, mock_client)
        mock_qdrant_client.assert_called_once_with(url=self.qdrant_url)
        # Verify that get_collections() was called to test connection
        mock_client.get_collections.assert_called_once()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_with_default_url(self, mock_qdrant_client):
        """Test initialization with default URL from config."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        client = QdrantClientCustom()

        self.assertEqual(client.qdrant_url, QDRANT_URL)
        self.assertEqual(client.qdrant, mock_client)
        mock_qdrant_client.assert_called_once_with(url=QDRANT_URL)
        # Verify that get_collections() was called to test connection
        mock_client.get_collections.assert_called_once()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_connection_error_from_constructor(self, mock_qdrant_client):
        """Test initialization with connection error during client creation."""
        mock_qdrant_client.side_effect = Exception("Connection failed")

        with self.assertRaises(Exception) as context:
            QdrantClientCustom(self.qdrant_url)

        self.assertIn("Connection failed", str(context.exception))
        mock_qdrant_client.assert_called_once_with(url=self.qdrant_url)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_connection_error_from_get_collections(self, mock_qdrant_client):
        """Test initialization with connection error during connection test."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.side_effect = Exception("Connection failed")

        with self.assertRaises(Exception) as context:
            QdrantClientCustom(self.qdrant_url)

        self.assertIn("Connection failed", str(context.exception))
        mock_qdrant_client.assert_called_once_with(url=self.qdrant_url)
        mock_client.get_collections.assert_called_once()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_with_custom_url(self, mock_qdrant_client):
        """Test initialization with custom URL."""
        custom_url = "http://custom-qdrant:7333"
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        client = QdrantClientCustom(custom_url)

        self.assertEqual(client.qdrant_url, custom_url)
        self.assertEqual(client.qdrant, mock_client)
        mock_qdrant_client.assert_called_once_with(url=custom_url)
        # Verify that get_collections() was called to test connection
        mock_client.get_collections.assert_called_once()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_collection_exists_true(self, mock_qdrant_client):
        """Test collection_exists when collection exists."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # get_collection succeeds (no exception)
        mock_client.get_collection.return_value = {"name": "test_collection"}

        client = QdrantClientCustom(self.qdrant_url)
        result = client.collection_exists("test_collection")

        self.assertTrue(result)
        mock_client.get_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_collection_exists_false(self, mock_qdrant_client):
        """Test collection_exists when collection doesn't exist (404 error)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate 404 error for non-existent collection
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)
        result = client.collection_exists("nonexistent_collection")

        self.assertFalse(result)
        mock_client.get_collection.assert_called_with("nonexistent_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_collection_exists_connection_error(self, mock_qdrant_client):
        """Test collection_exists with connection error (should re-raise)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate connection error
        mock_client.get_collection.side_effect = Exception("Connection refused")

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(Exception) as context:
            client.collection_exists("test_collection")

        self.assertIn("Connection refused", str(context.exception))
        mock_client.get_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_collection_exists_http_error_not_404(self, mock_qdrant_client):
        """Test collection_exists with HTTP error other than 404 (should re-raise)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate 500 server error
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=500, reason_phrase="Internal Server Error", content=b"Server error", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(UnexpectedResponse) as context:
            client.collection_exists("test_collection")

        self.assertEqual(context.exception.status_code, 500)
        mock_client.get_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_delete_collection_success(self, mock_qdrant_client):
        """Test successful collection deletion."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # delete_collection succeeds (no exception)
        mock_client.delete_collection.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.delete_collection("test_collection")

        self.assertTrue(result)
        mock_client.delete_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_delete_collection_not_exists(self, mock_qdrant_client):
        """Test deleting collection that doesn't exist (404 error - should return True)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate 404 error for non-existent collection
        mock_client.delete_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Collection not found", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)
        result = client.delete_collection("nonexistent_collection")

        self.assertTrue(result)
        mock_client.delete_collection.assert_called_with("nonexistent_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_delete_collection_connection_error(self, mock_qdrant_client):
        """Test delete_collection with connection error (should re-raise)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate connection error
        mock_client.delete_collection.side_effect = Exception("Connection refused")

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(Exception) as context:
            client.delete_collection("test_collection")

        self.assertIn("Connection refused", str(context.exception))
        mock_client.delete_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_delete_collection_http_error_not_404(self, mock_qdrant_client):
        """Test delete_collection with HTTP error other than 404 (should re-raise)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Simulate 500 server error
        mock_client.delete_collection.side_effect = UnexpectedResponse(
            status_code=500, reason_phrase="Internal Server Error", content=b"Server error", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(UnexpectedResponse) as context:
            client.delete_collection("test_collection")

        self.assertEqual(context.exception.status_code, 500)
        mock_client.delete_collection.assert_called_with("test_collection")

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_success(self, mock_qdrant_client):
        """Test successful collection creation."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        # Create collection succeeds
        mock_client.create_collection.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection", vector_size=768)

        self.assertTrue(result)
        mock_client.create_collection.assert_called_once()
        # Verify the call was made with correct parameters
        call_args = mock_client.create_collection.call_args
        self.assertEqual(call_args[1]["collection_name"], "test_collection")
        self.assertEqual(call_args[1]["vectors_config"].size, 768)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_with_default_vector_size(self, mock_qdrant_client):
        """Test creating collection with default vector size."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        mock_client.create_collection.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection")

        self.assertTrue(result)
        call_args = mock_client.create_collection.call_args
        self.assertEqual(call_args[1]["vectors_config"].size, 512)  # Default size

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_already_exists_no_delete(self, mock_qdrant_client):
        """Test creating collection that already exists (delete_if_exists=False)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection exists
        mock_client.get_collection.return_value = {"name": "test_collection"}

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection", vector_size=768)

        self.assertTrue(result)
        # Should not call create_collection since it already exists
        mock_client.create_collection.assert_not_called()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_already_exists_with_delete(self, mock_qdrant_client):
        """Test creating collection that already exists (delete_if_exists=True)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection exists initially
        mock_client.get_collection.return_value = {"name": "test_collection"}
        mock_client.delete_collection.return_value = None
        mock_client.create_collection.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection", vector_size=768, delete_if_exists=True)

        self.assertTrue(result)
        # Should call delete_collection first, then create_collection
        mock_client.delete_collection.assert_called_with("test_collection")
        mock_client.create_collection.assert_called_once()

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_409_conflict(self, mock_qdrant_client):
        """Test creating collection that gets 409 conflict (already exists)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        # But create_collection returns 409 (race condition)
        mock_client.create_collection.side_effect = UnexpectedResponse(
            status_code=409, reason_phrase="Conflict", content=b"Already exists", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection", vector_size=768)

        self.assertTrue(result)  # Should return True for 409 conflicts

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_http_error(self, mock_qdrant_client):
        """Test creating collection with HTTP error other than 409."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        # Create collection fails with 500 error
        mock_client.create_collection.side_effect = UnexpectedResponse(
            status_code=500, reason_phrase="Internal Server Error", content=b"Server error", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(UnexpectedResponse) as context:
            client.create_collection("test_collection", vector_size=768)

        self.assertEqual(context.exception.status_code, 500)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_connection_error(self, mock_qdrant_client):
        """Test creating collection with connection error."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        # Create collection fails with connection error
        mock_client.create_collection.side_effect = Exception("Connection refused")

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(Exception) as context:
            client.create_collection("test_collection", vector_size=768)

        self.assertIn("Connection refused", str(context.exception))

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_collection_with_custom_distance(self, mock_qdrant_client):
        """Test creating collection with custom distance metric."""
        from qdrant_client.models import Distance

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        # Collection doesn't exist initially
        mock_client.get_collection.side_effect = UnexpectedResponse(
            status_code=404, reason_phrase="Not Found", content=b"Not found", headers=httpx.Headers({})
        )
        mock_client.create_collection.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_collection("test_collection", vector_size=1024, distance=Distance.MANHATTAN)

        self.assertTrue(result)
        call_args = mock_client.create_collection.call_args
        self.assertEqual(call_args[1]["vectors_config"].size, 1024)
        self.assertEqual(call_args[1]["vectors_config"].distance, Distance.MANHATTAN)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_payload_index_success(self, mock_qdrant_client):
        """Test successful payload index creation."""
        from qdrant_client.models import PayloadSchemaType

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.create_payload_index.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_payload_index("test_collection", "course", PayloadSchemaType.KEYWORD)

        self.assertTrue(result)
        mock_client.create_payload_index.assert_called_once_with(
            collection_name="test_collection", field_name="course", field_schema=PayloadSchemaType.KEYWORD
        )

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_payload_index_default_schema(self, mock_qdrant_client):
        """Test payload index creation with default schema type."""
        from qdrant_client.models import PayloadSchemaType

        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.create_payload_index.return_value = None

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_payload_index("test_collection", "course")

        self.assertTrue(result)
        call_args = mock_client.create_payload_index.call_args
        self.assertEqual(call_args[1]["field_schema"], PayloadSchemaType.KEYWORD)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_payload_index_already_exists(self, mock_qdrant_client):
        """Test payload index creation when index already exists (409 conflict)."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.create_payload_index.side_effect = UnexpectedResponse(
            status_code=409, reason_phrase="Conflict", content=b"Index already exists", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)
        result = client.create_payload_index("test_collection", "course")

        self.assertTrue(result)  # Should return True for 409 conflicts

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_payload_index_http_error(self, mock_qdrant_client):
        """Test payload index creation with HTTP error other than 409."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.create_payload_index.side_effect = UnexpectedResponse(
            status_code=500, reason_phrase="Internal Server Error", content=b"Server error", headers=httpx.Headers({})
        )

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(UnexpectedResponse) as context:
            client.create_payload_index("test_collection", "course")

        self.assertEqual(context.exception.status_code, 500)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_create_payload_index_connection_error(self, mock_qdrant_client):
        """Test payload index creation with connection error."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.create_payload_index.side_effect = Exception("Connection refused")

        client = QdrantClientCustom(self.qdrant_url)

        with self.assertRaises(Exception) as context:
            client.create_payload_index("test_collection", "course")

        self.assertIn("Connection refused", str(context.exception))


if __name__ == "__main__":
    unittest.main()
