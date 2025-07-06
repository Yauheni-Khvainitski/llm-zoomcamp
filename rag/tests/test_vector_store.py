"""
Unit tests for vector store functionality.
"""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.models import Distance, PointStruct

from rag.data.loader import DocumentLoader
from rag.data.vector_store import QdrantVectorLoader, VectorStoreLoader
from rag.search.qdrant_client_custom import QdrantClientCustom


class TestVectorStoreLoader:
    """Test cases for VectorStoreLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add qdrant attribute
        self.vector_store = VectorStoreLoader(qdrant_client=self.mock_qdrant_client)

        # Sample test documents with full_text field
        self.sample_docs = [
            {
                "doc_id": "test_doc_1",
                "text": "Docker is a containerization platform",
                "question": "What is Docker?",
                "full_text": "What is Docker? Docker is a containerization platform",
                "section": "Introduction",
                "course": "docker-course",
            },
            {
                "doc_id": "test_doc_2",
                "text": "Kubernetes orchestrates containers",
                "question": "What is Kubernetes?",
                "full_text": "What is Kubernetes? Kubernetes orchestrates containers",
                "section": "Orchestration",
                "course": "k8s-course",
            },
        ]

    def test_init_default_client(self):
        """Test initialization with default client."""
        with patch("rag.data.vector_store.QdrantClientCustom") as mock_client:
            vector_store = VectorStoreLoader()
            mock_client.assert_called_once()
            assert vector_store.embedding_model_name == "jinaai/jina-embeddings-v2-small-en"
            assert vector_store.embedding_model is None

    def test_init_custom_client(self):
        """Test initialization with custom client."""
        custom_client = Mock(spec=QdrantClientCustom)
        vector_store = VectorStoreLoader(qdrant_client=custom_client)
        assert vector_store.qdrant_client is custom_client

    def test_init_custom_embedding_model(self):
        """Test initialization with custom embedding model."""
        vector_store = VectorStoreLoader(embedding_model="custom-model")
        assert vector_store.embedding_model_name == "custom-model"

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_success(self, mock_text_embedding):
        """Test successful embedding model initialization."""
        mock_embedding = Mock()
        mock_text_embedding.return_value = mock_embedding

        result = self.vector_store._get_embedding_model()

        mock_text_embedding.assert_called_once_with(model_name="jinaai/jina-embeddings-v2-small-en")
        assert result is mock_embedding
        assert self.vector_store.embedding_model is mock_embedding

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_cached(self, mock_text_embedding):
        """Test that embedding model is cached after first initialization."""
        mock_embedding = Mock()
        self.vector_store.embedding_model = mock_embedding

        result = self.vector_store._get_embedding_model()

        mock_text_embedding.assert_not_called()
        assert result is mock_embedding

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_failure(self, mock_text_embedding):
        """Test embedding model initialization failure."""
        mock_text_embedding.side_effect = Exception("Model not found")

        with pytest.raises(Exception) as exc_info:
            self.vector_store._get_embedding_model()

        assert "Embedding model initialization failed" in str(exc_info.value)

    def test_generate_embeddings_empty_documents(self):
        """Test embedding generation with empty document list."""
        result = self.vector_store.generate_embeddings([])
        assert result == []

    @patch("rag.data.vector_store.TextEmbedding")
    def test_generate_embeddings_success(self, mock_text_embedding):
        """Test successful embedding generation."""
        # Mock embedding model
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding

        result = self.vector_store.generate_embeddings(self.sample_docs)

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.1, 0.2, 0.3]

    @patch("rag.data.vector_store.TextEmbedding")
    def test_generate_embeddings_failure(self, mock_text_embedding):
        """Test embedding generation failure."""
        mock_embedding = Mock()
        mock_embedding.embed.side_effect = Exception("Embedding failed")
        mock_text_embedding.return_value = mock_embedding

        with pytest.raises(Exception) as exc_info:
            self.vector_store.generate_embeddings(self.sample_docs)

        assert "Embedding generation failed" in str(exc_info.value)

    def test_create_points_success(self):
        """Test successful point creation."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        result = self.vector_store.create_points(self.sample_docs, embeddings)

        assert len(result) == 2
        assert all(isinstance(point, PointStruct) for point in result)

        # Check first point
        point1 = result[0]
        assert point1.id == abs(hash("test_doc_1"))
        assert point1.vector == [0.1, 0.2, 0.3]
        assert point1.payload["doc_id"] == "test_doc_1"
        assert point1.payload["text"] == "Docker is a containerization platform"
        assert point1.payload["question"] == "What is Docker?"

    def test_create_points_mismatched_lengths(self):
        """Test point creation with mismatched document and embedding lengths."""
        embeddings = [[0.1, 0.2, 0.3]]  # Only one embedding for two documents

        with pytest.raises(ValueError) as exc_info:
            self.vector_store.create_points(self.sample_docs, embeddings)

        assert "Number of documents (2) must match number of embeddings (1)" in str(exc_info.value)

    def test_create_points_missing_doc_id(self):
        """Test point creation with missing doc_id."""
        docs = [{"text": "Test document", "question": "Test?"}]
        embeddings = [[0.1, 0.2, 0.3]]

        result = self.vector_store.create_points(docs, embeddings)

        assert len(result) == 1
        assert result[0].payload["doc_id"] == "doc_0"

    def test_create_points_failure(self):
        """Test point creation failure."""
        docs = [{"doc_id": "test", "text": None}]  # Invalid data
        embeddings = [[0.1, 0.2, 0.3]]

        # Mock PointStruct to raise exception
        with patch("rag.data.vector_store.PointStruct", side_effect=Exception("Point creation failed")):
            with pytest.raises(Exception) as exc_info:
                self.vector_store.create_points(docs, embeddings)

            assert "Point creation failed" in str(exc_info.value)

    def test_load_to_qdrant_empty_documents(self):
        """Test loading empty document list."""
        result = self.vector_store.load_to_qdrant("test-collection", [])
        assert result == 0

    def test_load_to_qdrant_collection_not_exists(self):
        """Test loading to non-existent collection."""
        self.mock_qdrant_client.collection_exists.return_value = False

        with pytest.raises(ValueError) as exc_info:
            self.vector_store.load_to_qdrant("test-collection", self.sample_docs)

        assert "Collection 'test-collection' does not exist" in str(exc_info.value)

    @patch("rag.data.vector_store.TextEmbedding")
    def test_load_to_qdrant_success(self, mock_text_embedding):
        """Test successful loading to Qdrant."""
        # Setup mocks
        self.mock_qdrant_client.collection_exists.return_value = True
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding

        result = self.vector_store.load_to_qdrant("test-collection", self.sample_docs)

        assert result == 2
        self.mock_qdrant_client.qdrant.upsert.assert_called_once()

    @patch("rag.data.vector_store.TextEmbedding")
    def test_load_to_qdrant_batch_upload_failure(self, mock_text_embedding):
        """Test batch upload failure during loading."""
        # Setup mocks
        self.mock_qdrant_client.collection_exists.return_value = True
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding
        self.mock_qdrant_client.qdrant.upsert.side_effect = Exception("Upload failed")

        with pytest.raises(Exception) as exc_info:
            self.vector_store.load_to_qdrant("test-collection", self.sample_docs)

        assert "Batch upload failed" in str(exc_info.value)

    @patch("rag.data.vector_store.TextEmbedding")
    def test_load_to_qdrant_embedding_failure(self, mock_text_embedding):
        """Test embedding generation failure during loading."""
        self.mock_qdrant_client.collection_exists.return_value = True
        mock_text_embedding.side_effect = Exception("Embedding failed")

        with pytest.raises(Exception) as exc_info:
            self.vector_store.load_to_qdrant("test-collection", self.sample_docs)

        assert "Qdrant loading failed" in str(exc_info.value)


class TestQdrantVectorLoader:
    """Test cases for QdrantVectorLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_document_loader = Mock(spec=DocumentLoader)
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add qdrant attribute
        self.qdrant_loader = QdrantVectorLoader(
            document_loader=self.mock_document_loader, qdrant_client=self.mock_qdrant_client
        )

        # Sample documents with full_text field
        self.sample_docs = [
            {
                "doc_id": "test_1",
                "text": "Test document 1",
                "question": "Test question 1?",
                "full_text": "Test question 1? Test document 1",
                "course": "test-course",
            },
            {
                "doc_id": "test_2",
                "text": "Test document 2",
                "question": "Test question 2?",
                "full_text": "Test question 2? Test document 2",
                "course": "other-course",
            },
        ]

    def test_init_default_dependencies(self):
        """Test initialization with default dependencies."""
        with patch("rag.data.vector_store.DocumentLoader") as mock_doc_loader:
            with patch("rag.data.vector_store.VectorStoreLoader") as mock_vector_store:
                QdrantVectorLoader()

                mock_doc_loader.assert_called_once()
                mock_vector_store.assert_called_once()

    def test_init_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        custom_doc_loader = Mock(spec=DocumentLoader)
        custom_qdrant_client = Mock(spec=QdrantClientCustom)

        qdrant_loader = QdrantVectorLoader(document_loader=custom_doc_loader, qdrant_client=custom_qdrant_client)

        assert qdrant_loader.document_loader is custom_doc_loader
        assert qdrant_loader.vector_store.qdrant_client is custom_qdrant_client

    @patch("rag.data.vector_store.TextEmbedding")
    def test_setup_collection_success(self, mock_text_embedding):
        """Test successful collection setup."""
        # Setup mocks
        self.mock_document_loader.load_documents.return_value = self.sample_docs
        self.mock_qdrant_client.collection_exists.return_value = True
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding

        result = self.qdrant_loader.setup_collection("test-collection")

        assert result["collection_name"] == "test-collection"
        assert result["documents_loaded"] == 2
        assert result["points_uploaded"] == 2
        assert result["course_filter"] is None

        # Verify collection creation was called
        self.qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once()

    @patch("rag.data.vector_store.TextEmbedding")
    def test_setup_collection_with_course_filter(self, mock_text_embedding):
        """Test collection setup with course filter."""
        # Setup mocks
        self.mock_document_loader.load_documents.return_value = self.sample_docs
        self.mock_qdrant_client.collection_exists.return_value = True
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding

        result = self.qdrant_loader.setup_collection("test-collection", course_filter="test-course")

        assert result["collection_name"] == "test-collection"
        assert result["documents_loaded"] == 1  # Only one document matches filter
        assert result["points_uploaded"] == 1
        assert result["course_filter"] == "test-course"

    def test_setup_collection_document_loading_failure(self):
        """Test collection setup with document loading failure."""
        self.mock_document_loader.load_documents.side_effect = Exception("Document loading failed")

        with pytest.raises(Exception) as exc_info:
            self.qdrant_loader.setup_collection("test-collection")

        assert "Collection setup failed" in str(exc_info.value)

    @patch("rag.data.vector_store.TextEmbedding")
    def test_setup_collection_vector_loading_failure(self, mock_text_embedding):
        """Test collection setup with vector loading failure."""
        # Setup mocks
        self.mock_document_loader.load_documents.return_value = self.sample_docs
        self.mock_qdrant_client.collection_exists.return_value = False  # Collection doesn't exist

        with pytest.raises(Exception) as exc_info:
            self.qdrant_loader.setup_collection("test-collection")

        assert "Collection setup failed" in str(exc_info.value)

    def test_setup_collection_delete_if_exists(self):
        """Test collection setup with delete_if_exists flag."""
        self.mock_document_loader.load_documents.return_value = []

        self.qdrant_loader.setup_collection("test-collection", delete_if_exists=True)

        # Verify create_collection was called with delete_if_exists=True
        self.qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once_with(
            collection_name="test-collection", vector_size=512, distance=Distance.COSINE, delete_if_exists=True
        )


class TestIntegration:
    """Integration tests for vector store components."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add qdrant attribute
        self.mock_qdrant_client.collection_exists.return_value = True

        # Create sample documents with full_text field
        self.sample_docs = [
            {
                "doc_id": "integration_test_1",
                "text": "Integration test document 1",
                "question": "What is integration testing?",
                "full_text": "What is integration testing? Integration test document 1",
                "section": "Testing",
                "course": "software-engineering",
            }
        ]

    @patch("rag.data.vector_store.TextEmbedding")
    def test_end_to_end_vector_loading(self, mock_text_embedding):
        """Test end-to-end vector loading process."""
        # Setup embedding mock
        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_text_embedding.return_value = mock_embedding

        # Test VectorStoreLoader
        vector_store = VectorStoreLoader(qdrant_client=self.mock_qdrant_client)

        # Generate embeddings
        embeddings = vector_store.generate_embeddings(self.sample_docs)
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Create points
        points = vector_store.create_points(self.sample_docs, embeddings)
        assert len(points) == 1
        assert points[0].id == abs(hash("integration_test_1"))
        assert points[0].vector == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Load to Qdrant
        result = vector_store.load_to_qdrant("test-collection", self.sample_docs)
        assert result == 1

        # Verify upsert was called
        self.mock_qdrant_client.qdrant.upsert.assert_called_once()

    @patch("rag.data.vector_store.TextEmbedding")
    def test_high_level_interface_integration(self, mock_text_embedding):
        """Test high-level interface integration."""
        # Setup mocks
        mock_document_loader = Mock(spec=DocumentLoader)
        mock_document_loader.load_documents.return_value = self.sample_docs

        mock_embedding = Mock()
        mock_embedding.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_text_embedding.return_value = mock_embedding

        # Test QdrantVectorLoader
        qdrant_loader = QdrantVectorLoader(document_loader=mock_document_loader, qdrant_client=self.mock_qdrant_client)

        result = qdrant_loader.setup_collection("integration-test")

        assert result["collection_name"] == "integration-test"
        assert result["documents_loaded"] == 1
        assert result["points_uploaded"] == 1

        # Verify all components were called
        mock_document_loader.load_documents.assert_called_once()
        qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once()
        self.mock_qdrant_client.qdrant.upsert.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
