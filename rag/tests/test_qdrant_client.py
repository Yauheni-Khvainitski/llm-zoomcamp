"""
Tests for the QdrantClientCustom class.
"""

import unittest
from unittest.mock import Mock, patch

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

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_with_default_url(self, mock_qdrant_client):
        """Test initialization with default URL from config."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client

        client = QdrantClientCustom()

        self.assertEqual(client.qdrant_url, QDRANT_URL)
        self.assertEqual(client.qdrant, mock_client)
        mock_qdrant_client.assert_called_once_with(url=QDRANT_URL)

    @patch("rag.search.qdrant_client_custom.QdrantClient")
    def test_init_connection_error(self, mock_qdrant_client):
        """Test initialization with connection error."""
        mock_qdrant_client.side_effect = Exception("Connection failed")

        with self.assertRaises(Exception) as context:
            QdrantClientCustom(self.qdrant_url)

        self.assertIn("Connection failed", str(context.exception))
        mock_qdrant_client.assert_called_once_with(url=self.qdrant_url)

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


if __name__ == "__main__":
    unittest.main()
