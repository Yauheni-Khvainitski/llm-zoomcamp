"""
Unit tests for vector store functionality.
"""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.models import Distance, PointStruct

from rag.data.loader import DocumentLoader
from rag.data.vector_store import QdrantVectorLoader, VectorSearcher, VectorStoreLoader
from rag.search.qdrant_client_custom import QdrantClientCustom


class TestVectorStoreLoader:  # pylint: disable=attribute-defined-outside-init
    """Test cases for VectorStoreLoader class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add the qdrant attribute
        self.vector_store = VectorStoreLoader(qdrant_client=self.mock_qdrant_client)

        # Sample documents for testing
        self.sample_docs = [
            {
                "doc_id": "test_doc_1",
                "text": "Docker is a containerization platform",
                "question": "What is Docker?",
                "course": "docker",
                "section": "Introduction",
            },
            {
                "doc_id": "test_doc_2",
                "text": "Kubernetes is a container orchestration system",
                "question": "What is Kubernetes?",
                "course": "kubernetes",
                "section": "Orchestration",
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

    @patch("rag.data.vector_store.QdrantClientCustom")
    def test_init_custom_embedding_model(self, mock_qdrant_client_class):
        """Test initialization with custom embedding model."""
        mock_qdrant_client_class.return_value = Mock()
        vector_store = VectorStoreLoader(embedding_model="custom-model")
        assert vector_store.embedding_model_name == "custom-model"

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_success(self, mock_text_embedding):
        """Test successful embedding model initialization."""
        mock_embedding = Mock()
        mock_text_embedding.return_value = mock_embedding

        result = self.vector_store.get_embedding_model()

        mock_text_embedding.assert_called_once_with(model_name="jinaai/jina-embeddings-v2-small-en")
        assert result is mock_embedding
        assert self.vector_store.embedding_model is mock_embedding

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_cached(self, mock_text_embedding):
        """Test that embedding model is cached after first initialization."""
        mock_embedding = Mock()
        self.vector_store.embedding_model = mock_embedding

        result = self.vector_store.get_embedding_model()

        mock_text_embedding.assert_not_called()
        assert result is mock_embedding

    @patch("rag.data.vector_store.TextEmbedding")
    def test_get_embedding_model_failure(self, mock_text_embedding):
        """Test embedding model initialization failure."""
        mock_text_embedding.side_effect = Exception("Model not found")

        with pytest.raises(Exception) as exc_info:
            self.vector_store.get_embedding_model()

        assert "Embedding model initialization failed" in str(exc_info.value)

    def test_generate_embeddings_empty_documents(self):
        """Test embedding generation with empty document list."""
        result = self.vector_store.generate_embeddings([])
        assert not result

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


class TestQdrantVectorLoader:  # pylint: disable=attribute-defined-outside-init
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
                with patch("rag.data.vector_store.QdrantClientCustom"):
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
        assert result["payload_index_created"] is True

        # Verify collection creation and payload index creation were called
        self.qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once()
        self.qdrant_loader.vector_store.qdrant_client.create_payload_index.assert_called_once()

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
        assert result["payload_index_created"] is True

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

        result = self.qdrant_loader.setup_collection("test-collection", delete_if_exists=True)

        # Verify payload index was created
        assert result["payload_index_created"] is True

        # Verify create_collection was called with delete_if_exists=True
        self.qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once_with(
            collection_name="test-collection", vector_size=512, distance=Distance.COSINE, delete_if_exists=True
        )

        # Verify payload index creation was called
        self.qdrant_loader.vector_store.qdrant_client.create_payload_index.assert_called_once()


class TestIntegration:  # pylint: disable=attribute-defined-outside-init
    """Test cases for integration testing."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add the qdrant attribute

        # Create a vector store loader
        self.sample_docs = [
            {
                "doc_id": "integration_test_1",
                "text": "Integration test document 1",
                "question": "What is integration testing?",
                "full_text": "What is integration testing? Integration test document 1",
                "course": "test-course",
                "section": "Testing",
            },
            {
                "doc_id": "integration_test_2",
                "text": "Integration test document 2",
                "question": "How does integration testing work?",
                "full_text": "How does integration testing work? Integration test document 2",
                "course": "test-course",
                "section": "Testing",
            },
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
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert embeddings[1] == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Create points
        points = vector_store.create_points(self.sample_docs, embeddings)
        assert len(points) == 2
        assert points[0].id == abs(hash("integration_test_1"))
        assert points[0].vector == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert points[1].id == abs(hash("integration_test_2"))
        assert points[1].vector == [0.1, 0.2, 0.3, 0.4, 0.5]

        # Load to Qdrant
        result = vector_store.load_to_qdrant("test-collection", self.sample_docs)
        assert result == 2

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
        assert result["documents_loaded"] == 2
        assert result["points_uploaded"] == 2
        assert result["payload_index_created"] is True

        # Verify all components were called
        mock_document_loader.load_documents.assert_called_once()
        qdrant_loader.vector_store.qdrant_client.create_collection.assert_called_once()
        qdrant_loader.vector_store.qdrant_client.create_payload_index.assert_called_once()
        self.mock_qdrant_client.qdrant.upsert.assert_called_once()


class TestVectorSearcher:  # pylint: disable=attribute-defined-outside-init
    """Test cases for VectorSearcher class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add the qdrant attribute
        self.vector_searcher = VectorSearcher(qdrant_client=self.mock_qdrant_client)

    def test_init_default_dependencies(self):
        """Test initialization with default dependencies."""
        with patch("rag.data.vector_store.VectorStoreLoader") as mock_vector_store:
            with patch("rag.data.vector_store.QdrantClientCustom"):
                VectorSearcher()

                mock_vector_store.assert_called_once()

    def test_init_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        custom_qdrant_client = Mock(spec=QdrantClientCustom)

        searcher = VectorSearcher(embedding_model="custom-model", qdrant_client=custom_qdrant_client)

        assert searcher.qdrant_client is custom_qdrant_client
        assert searcher.vector_store is not None

    def test_embed_query_success(self):
        """Test successful query embedding."""
        # Mock the embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        query = "What is Docker?"
        result = self.vector_searcher.embed_query(query)

        # Verify embedding was called correctly
        mock_embedding_model.embed.assert_called_once_with([query])
        self.vector_searcher.vector_store.get_embedding_model.assert_called_once()
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_embed_query_empty_query(self):
        """Test embedding with empty query."""
        with pytest.raises(ValueError) as exc_info:
            self.vector_searcher.embed_query("")

        assert "Query cannot be empty" in str(exc_info.value)

    def test_embed_query_whitespace_only(self):
        """Test embedding with whitespace-only query."""
        with pytest.raises(ValueError) as exc_info:
            self.vector_searcher.embed_query("   ")

        assert "Query cannot be empty" in str(exc_info.value)

    def test_embed_query_numpy_array_handling(self):
        """Test embedding with numpy array return value."""
        # Mock numpy array-like object
        mock_numpy_array = Mock()
        mock_numpy_array.tolist.return_value = [0.1, 0.2, 0.3]

        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [mock_numpy_array]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        result = self.vector_searcher.embed_query("test query")

        # Should handle numpy array conversion
        mock_numpy_array.tolist.assert_called_once()
        assert result == [0.1, 0.2, 0.3]

    def test_embed_query_embedding_failure(self):
        """Test embedding failure handling."""
        mock_embedding_model = Mock()
        mock_embedding_model.embed.side_effect = Exception("Embedding failed")
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        with pytest.raises(RuntimeError) as exc_info:
            self.vector_searcher.embed_query("test query")

        assert "Query embedding generation failed" in str(exc_info.value)

    def test_search_success(self):
        """Test successful vector search with text query."""
        # Mock embedding generation
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        # Mock search results
        mock_search_results = [
            {"id": "doc_1", "payload": {"text": "Docker is a containerization platform", "score": 0.95}, "score": 0.95}
        ]
        self.mock_qdrant_client.search_with_vector.return_value = mock_search_results

        query = "What is Docker?"
        result = self.vector_searcher.search(query, collection_name="test-collection", limit=5)

        # Verify embedding was generated
        mock_embedding_model.embed.assert_called_once_with([query])

        # Verify search was called with correct parameters
        self.mock_qdrant_client.search_with_vector.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3],
            collection_name="test-collection",
            limit=5,
            course_filter=None,
            score_threshold=None,
            with_payload=True,
        )

        assert result == mock_search_results

    def test_search_with_course_filter(self):
        """Test vector search with course filter."""
        # Mock embedding generation
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        # Mock search results
        mock_search_results = [{"id": "doc_1", "payload": {"course": "docker-course"}}]
        self.mock_qdrant_client.search_with_vector.return_value = mock_search_results

        result = self.vector_searcher.search(
            "What is Docker?", collection_name="test-collection", limit=3, course_filter="docker-course"
        )

        # Verify search was called with course filter
        call_args = self.mock_qdrant_client.search_with_vector.call_args
        assert call_args[1]["course_filter"] == "docker-course"
        assert result == mock_search_results

    def test_search_default_parameters(self):
        """Test search with default parameters."""
        # Mock embedding generation
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        self.mock_qdrant_client.search_with_vector.return_value = []

        self.vector_searcher.search("test query", collection_name="test-collection")

        # Verify default parameters were used
        call_args = self.mock_qdrant_client.search_with_vector.call_args
        assert call_args[1]["limit"] == 5  # default limit
        assert call_args[1]["with_payload"] is True  # default with_payload

    def test_search_embedding_failure(self):
        """Test search with embedding failure."""
        mock_embedding_model = Mock()
        mock_embedding_model.embed.side_effect = Exception("Embedding failed")
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        with pytest.raises(RuntimeError) as exc_info:
            self.vector_searcher.search("test query", collection_name="test-collection")

        assert "Search failed" in str(exc_info.value)

    def test_search_qdrant_search_failure(self):
        """Test search with Qdrant search failure."""
        # Mock successful embedding
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3]]
        self.vector_searcher.vector_store.get_embedding_model = Mock(return_value=mock_embedding_model)

        # Mock Qdrant search failure
        self.mock_qdrant_client.search_with_vector.side_effect = Exception("Qdrant search failed")

        with pytest.raises(RuntimeError) as exc_info:
            self.vector_searcher.search("test query", collection_name="test-collection")

        assert "Search failed" in str(exc_info.value)

    def test_search_with_vector_success(self):
        """Test successful vector search with pre-computed vector."""
        mock_search_results = [{"id": "doc_1", "payload": {"text": "Docker containers", "score": 0.92}, "score": 0.92}]
        self.mock_qdrant_client.search_with_vector.return_value = mock_search_results

        query_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = self.vector_searcher.search_with_vector(
            collection_name="test-collection", query_vector=query_vector, limit=10
        )

        # Verify search was called with correct parameters
        self.mock_qdrant_client.search_with_vector.assert_called_once_with(
            query_vector=query_vector,
            collection_name="test-collection",
            limit=10,
            course_filter=None,
            score_threshold=None,
            with_payload=True,
        )

        assert result == mock_search_results

    def test_search_with_vector_with_course_filter(self):
        """Test vector search with pre-computed vector and course filter."""
        mock_search_results = []
        self.mock_qdrant_client.search_with_vector.return_value = mock_search_results

        query_vector = [0.1, 0.2, 0.3]
        result = self.vector_searcher.search_with_vector(
            collection_name="test-collection", query_vector=query_vector, limit=3, course_filter="ml-course"
        )

        # Verify search was called with course filter
        call_args = self.mock_qdrant_client.search_with_vector.call_args
        assert call_args[1]["course_filter"] == "ml-course"
        assert result == mock_search_results

    def test_search_with_vector_empty_vector(self):
        """Test search with empty vector."""
        with pytest.raises(ValueError) as exc_info:
            self.vector_searcher.search_with_vector(collection_name="test-collection", query_vector=[])

        assert "Query vector cannot be empty" in str(exc_info.value)

    def test_search_with_vector_none_vector(self):
        """Test search with None vector."""
        with pytest.raises(ValueError) as exc_info:
            self.vector_searcher.search_with_vector(collection_name="test-collection", query_vector=None)

        assert "Query vector cannot be empty" in str(exc_info.value)

    def test_search_with_vector_qdrant_failure(self):
        """Test search with vector when Qdrant search fails."""
        self.mock_qdrant_client.search_with_vector.side_effect = Exception("Qdrant error")

        with pytest.raises(RuntimeError) as exc_info:
            self.vector_searcher.search_with_vector(collection_name="test-collection", query_vector=[0.1, 0.2, 0.3])

        assert "Search failed" in str(exc_info.value)

    def test_search_with_vector_default_parameters(self):
        """Test search with vector using default parameters."""
        self.mock_qdrant_client.search_with_vector.return_value = []

        self.vector_searcher.search_with_vector(collection_name="test-collection", query_vector=[0.1, 0.2, 0.3])

        # Verify default parameters were used
        call_args = self.mock_qdrant_client.search_with_vector.call_args
        assert call_args[1]["limit"] == 5  # default limit
        assert call_args[1]["with_payload"] is True  # default with_payload
        assert call_args[1]["course_filter"] is None  # default course_filter


class TestVectorSearcherIntegration:  # pylint: disable=attribute-defined-outside-init
    """Integration tests for VectorSearcher."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_client = Mock(spec=QdrantClientCustom)
        self.mock_qdrant_client.qdrant = Mock()  # Add the qdrant attribute

        # Create the vector searcher
        self.vector_searcher = VectorSearcher(qdrant_client=self.mock_qdrant_client)

    @patch("rag.data.vector_store.TextEmbedding")
    def test_search_integration_success(self, mock_text_embedding):
        """Test successful integration between VectorSearcher and VectorStoreLoader."""
        # Mock embedding model
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_text_embedding.return_value = mock_embedding_model

        # Mock search results
        mock_search_results = [
            {
                "id": "integration_doc_1",
                "payload": {
                    "text": "Docker integration test",
                    "question": "How does Docker work?",
                    "course": "docker-course",
                    "score": 0.89,
                },
                "score": 0.89,
            }
        ]
        self.mock_qdrant_client.search_with_vector.return_value = mock_search_results

        # Test search
        result = self.vector_searcher.search("Docker integration test", collection_name="integration-collection", limit=3)

        # Verify the full pipeline worked
        mock_text_embedding.assert_called_once_with(model_name="jinaai/jina-embeddings-v2-small-en")
        mock_embedding_model.embed.assert_called_once_with(["Docker integration test"])
        self.mock_qdrant_client.search_with_vector.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            collection_name="integration-collection",
            limit=3,
            course_filter=None,
            score_threshold=None,
            with_payload=True,
        )

        assert result == mock_search_results

    @patch("rag.data.vector_store.TextEmbedding")
    def test_embed_query_integration_success(self, mock_text_embedding):
        """Test successful embedding integration."""
        # Mock embedding model with numpy-like return
        mock_numpy_array = Mock()
        mock_numpy_array.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]

        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [mock_numpy_array]
        mock_text_embedding.return_value = mock_embedding_model

        # Test embedding
        result = self.vector_searcher.embed_query("Integration test query")

        # Verify embedding model was initialized properly
        mock_text_embedding.assert_called_once_with(model_name="jinaai/jina-embeddings-v2-small-en")
        mock_embedding_model.embed.assert_called_once_with(["Integration test query"])
        mock_numpy_array.tolist.assert_called_once()

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]


if __name__ == "__main__":
    pytest.main([__file__])
