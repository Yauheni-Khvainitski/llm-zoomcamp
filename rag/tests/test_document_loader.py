"""
Tests for the DocumentLoader class.
"""

import json
import unittest
from unittest.mock import Mock, patch

import requests

from ..data.loader import DocumentLoader


class TestDocumentLoader(unittest.TestCase):
    """Test suite for the DocumentLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
        self.sample_raw_documents = [
            {
                "course": "data-engineering-zoomcamp",
                "documents": [
                    {"text": "Sample text 1", "section": "General", "question": "What is Docker?"},
                    {"text": "Sample text 2", "section": "Setup", "question": "How to install Docker?"},
                ],
            },
            {
                "course": "machine-learning-zoomcamp",
                "documents": [{"text": "ML sample text", "section": "Algorithms", "question": "What is linear regression?"}],
            },
        ]

    def test_init_default_url(self):
        """Test initialization with default URL."""
        loader = DocumentLoader()
        self.assertIn("documents.json", loader.documents_url)
        self.assertEqual(loader.documents, [])

    def test_init_custom_url(self):
        """Test initialization with custom URL."""
        custom_url = "https://example.com/docs.json"
        loader = DocumentLoader(custom_url)
        self.assertEqual(loader.documents_url, custom_url)

    def test_generate_id(self):
        """Test document ID generation."""
        doc1 = {"text": "test", "question": "q1"}
        doc2 = {"text": "test", "question": "q1"}
        doc3 = {"text": "different", "question": "q1"}

        id1 = self.loader.generate_id(doc1)
        id2 = self.loader.generate_id(doc2)
        id3 = self.loader.generate_id(doc3)

        # Same documents should have same ID
        self.assertEqual(id1, id2)
        # Different documents should have different IDs
        self.assertNotEqual(id1, id3)
        # IDs should be MD5 hashes (32 characters)
        self.assertEqual(len(id1), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in id1))

    def test_generate_id_deterministic(self):
        """Test that ID generation is deterministic."""
        doc = {"text": "test", "question": "q1", "section": "s1"}

        # Generate ID multiple times
        ids = [self.loader.generate_id(doc) for _ in range(5)]

        # All IDs should be the same
        self.assertTrue(all(id == ids[0] for id in ids))

    @patch("requests.get")
    def test_fetch_documents_success(self, mock_get):
        """Test successful document fetching."""
        mock_response = Mock()
        mock_response.json.return_value = self.sample_raw_documents
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.loader.fetch_documents()

        mock_get.assert_called_once_with(self.loader.documents_url, timeout=30)
        mock_response.raise_for_status.assert_called_once()
        self.assertEqual(result, self.sample_raw_documents)

    @patch("requests.get")
    def test_fetch_documents_http_error(self, mock_get):
        """Test document fetching with HTTP error."""
        mock_get.side_effect = requests.RequestException("HTTP Error")

        with self.assertRaises(requests.RequestException):
            self.loader.fetch_documents()

    @patch("requests.get")
    def test_fetch_documents_json_error(self, mock_get):
        """Test document fetching with JSON parsing error."""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        with self.assertRaises(json.JSONDecodeError):
            self.loader.fetch_documents()

    def test_process_documents(self):
        """Test document processing."""
        processed = self.loader.process_documents(self.sample_raw_documents)

        # Should have 3 documents total (2 from first course, 1 from second)
        self.assertEqual(len(processed), 3)

        # Check first document
        doc1 = processed[0]
        self.assertEqual(doc1["course"], "data-engineering-zoomcamp")
        self.assertEqual(doc1["text"], "Sample text 1")
        self.assertEqual(doc1["question"], "What is Docker?")
        self.assertIn("doc_id", doc1)
        self.assertEqual(len(doc1["doc_id"]), 32)  # MD5 hash length

        # Check that all documents have required fields
        for doc in processed:
            self.assertIn("course", doc)
            self.assertIn("doc_id", doc)
            self.assertIn("text", doc)
            self.assertIn("question", doc)
            self.assertIn("section", doc)

    def test_process_documents_empty(self):
        """Test processing empty document list."""
        processed = self.loader.process_documents([])
        self.assertEqual(processed, [])

    def test_process_documents_unique_ids(self):
        """Test that processed documents have unique IDs."""
        processed = self.loader.process_documents(self.sample_raw_documents)
        doc_ids = [doc["doc_id"] for doc in processed]

        # All IDs should be unique
        self.assertEqual(len(doc_ids), len(set(doc_ids)))

    @patch.object(DocumentLoader, "fetch_documents")
    @patch.object(DocumentLoader, "process_documents")
    def test_load_documents_first_time(self, mock_process, mock_fetch):
        """Test loading documents for the first time."""
        mock_fetch.return_value = self.sample_raw_documents
        mock_process.return_value = [{"processed": "doc"}]

        result = self.loader.load_documents()

        mock_fetch.assert_called_once()
        mock_process.assert_called_once_with(self.sample_raw_documents)
        self.assertEqual(result, [{"processed": "doc"}])
        self.assertEqual(self.loader.documents, [{"processed": "doc"}])

    def test_load_documents_cached(self):
        """Test loading documents when already cached."""
        # Pre-populate documents
        self.loader.documents = [{"cached": "doc"}]

        with patch.object(self.loader, "fetch_documents") as mock_fetch:
            result = self.loader.load_documents()

            # Should not fetch again
            mock_fetch.assert_not_called()
            self.assertEqual(result, [{"cached": "doc"}])

    def test_get_documents_by_course(self):
        """Test filtering documents by course."""
        # Setup test documents
        test_docs = [
            {"course": "data-engineering-zoomcamp", "text": "DE doc"},
            {"course": "machine-learning-zoomcamp", "text": "ML doc"},
            {"course": "data-engineering-zoomcamp", "text": "Another DE doc"},
        ]
        self.loader.documents = test_docs

        # Test filtering
        de_docs = self.loader.get_documents_by_course("data-engineering-zoomcamp")
        ml_docs = self.loader.get_documents_by_course("machine-learning-zoomcamp")

        self.assertEqual(len(de_docs), 2)
        self.assertEqual(len(ml_docs), 1)
        self.assertEqual(de_docs[0]["text"], "DE doc")
        self.assertEqual(ml_docs[0]["text"], "ML doc")

    def test_get_documents_by_course_empty(self):
        """Test filtering by non-existent course."""
        self.loader.documents = [{"course": "data-engineering-zoomcamp", "text": "DE doc"}]

        result = self.loader.get_documents_by_course("non-existent-course")
        self.assertEqual(result, [])

    @patch.object(DocumentLoader, "load_documents")
    def test_get_documents_by_course_loads_if_empty(self, mock_load):
        """Test that get_documents_by_course loads documents if not already loaded."""

        def mock_load_side_effect():
            self.loader.documents = [{"course": "test", "text": "doc"}]
            return self.loader.documents

        mock_load.side_effect = mock_load_side_effect

        result = self.loader.get_documents_by_course("test")

        mock_load.assert_called_once()
        self.assertEqual(result, [{"course": "test", "text": "doc"}])

    def test_get_document_stats(self):
        """Test getting document statistics."""
        test_docs = [
            {"course": "data-engineering-zoomcamp", "text": "DE doc 1"},
            {"course": "data-engineering-zoomcamp", "text": "DE doc 2"},
            {"course": "machine-learning-zoomcamp", "text": "ML doc"},
            {"course": "mlops-zoomcamp", "text": "MLOps doc"},
        ]
        self.loader.documents = test_docs

        stats = self.loader.get_document_stats()

        self.assertEqual(stats["total_documents"], 4)
        self.assertEqual(stats["unique_courses"], 3)
        self.assertEqual(stats["documents_by_course"]["data-engineering-zoomcamp"], 2)
        self.assertEqual(stats["documents_by_course"]["machine-learning-zoomcamp"], 1)
        self.assertEqual(stats["documents_by_course"]["mlops-zoomcamp"], 1)

    @patch.object(DocumentLoader, "load_documents")
    def test_get_document_stats_loads_if_empty(self, mock_load):
        """Test that get_document_stats loads documents if not already loaded."""

        def mock_load_side_effect():
            self.loader.documents = [{"course": "test", "text": "doc"}]
            return self.loader.documents

        mock_load.side_effect = mock_load_side_effect

        stats = self.loader.get_document_stats()

        mock_load.assert_called_once()
        self.assertEqual(stats["total_documents"], 1)


if __name__ == "__main__":
    unittest.main()
