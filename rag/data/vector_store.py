"""Vector store operations for RAG system.

Handles embedding generation and vector database operations.
"""

import logging
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client.models import Distance, PointStruct

from ..config import EMBEDDING_DIMENSIONALITY, EMBEDDING_MODEL
from ..search.qdrant_client_custom import QdrantClientCustom
from .loader import DocumentLoader

logger = logging.getLogger(__name__)


class VectorStoreLoader:
    """Handles embedding generation and vector database operations."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL, qdrant_client: Optional[QdrantClientCustom] = None):
        """Initialize the vector store loader.

        Args:
            embedding_model: Name of the embedding model to use
            qdrant_client: Optional QdrantClientCustom instance. If None, creates a new one.
        """
        self.embedding_model_name = embedding_model
        self.embedding_model: Optional[TextEmbedding] = None
        self.qdrant_client = qdrant_client or QdrantClientCustom()

    def _get_embedding_model(self) -> TextEmbedding:
        """Get or initialize the embedding model.

        Returns:
            TextEmbedding instance

        Raises:
            Exception: If embedding model initialization fails
        """
        if self.embedding_model is None:
            try:
                logger.info(f"Initializing embedding model: {self.embedding_model_name}")
                self.embedding_model = TextEmbedding(model_name=self.embedding_model_name)
            except Exception as e:
                logger.error(f"Failed to initialize embedding model '{self.embedding_model_name}': {e}")
                raise RuntimeError(f"Embedding model initialization failed: {e}") from e
        return self.embedding_model

    def generate_embeddings(self, documents: List[Dict[str, Any]], text_field: str = "full_text") -> List[List[float]]:
        """Generate embeddings for documents.

        Args:
            documents: List of documents to embed
            text_field: Field name containing text to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return []

        try:
            logger.info(f"Generating embeddings for {len(documents)} documents")
            embedding_model = self._get_embedding_model()

            # Extract text for embedding
            texts = []
            for doc in documents:
                if "full_text" in doc:
                    text = doc["full_text"]
                else:
                    raise ValueError(f"Document {doc} does not have a full_text field")
                texts.append(text)

            # Generate embeddings
            embeddings = []
            for text in texts:
                try:
                    embedding_vector = list(embedding_model.embed([text]))[0]
                    # Handle both numpy arrays and lists
                    if hasattr(embedding_vector, "tolist"):
                        embeddings.append(embedding_vector.tolist())
                    else:
                        embeddings.append(embedding_vector)
                except Exception as e:
                    logger.error(f"Failed to generate embedding for text: {text[:50]}...")
                    raise RuntimeError(f"Embedding generation failed for document: {e}") from e

            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def create_points(self, documents: List[Dict[str, Any]], embeddings: List[List[float]]) -> List[PointStruct]:
        """Create Qdrant points from documents and embeddings.

        Args:
            documents: List of documents
            embeddings: List of embedding vectors

        Returns:
            List of PointStruct objects

        Raises:
            RuntimeError: If point creation fails
        """
        if len(documents) != len(embeddings):
            raise ValueError(f"Number of documents ({len(documents)}) must match number of embeddings ({len(embeddings)})")

        try:
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                try:
                    # Use doc_id as point ID to avoid overwrites when loading same data multiple times
                    doc_id = doc.get("doc_id", f"doc_{idx}")
                    point_id = hash(doc_id)
                    # Use absolute value to ensure positive ID (Qdrant requirement)
                    point_id = abs(point_id)

                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": doc.get("text", ""),
                            "question": doc.get("question", ""),
                            "section": doc.get("section", ""),
                            "course": doc.get("course", ""),
                            "doc_id": doc_id,
                        },
                    )
                    points.append(point)
                except Exception as e:
                    logger.error(f"Failed to create point for document {idx}: {e}")
                    raise RuntimeError(f"Point creation failed for document {idx}: {e}") from e

            logger.info(f"Created {len(points)} points using doc_id as point ID")
            return points

        except Exception as e:
            logger.error(f"Failed to create points: {e}")
            raise RuntimeError(f"Point creation failed: {e}") from e

    def load_to_qdrant(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """Load documents to an existing Qdrant collection.

        Args:
            collection_name: Name of the Qdrant collection (must exist)
            documents: List of documents to load
            batch_size: Batch size for uploading points

        Returns:
            Number of points uploaded

        Raises:
            ValueError: If collection doesn't exist
            RuntimeError: If loading fails
        """
        if not documents:
            logger.warning("No documents to load to Qdrant")
            return 0

        try:
            # Check if collection exists - fail if it doesn't
            if not self.qdrant_client.collection_exists(collection_name):
                raise ValueError(
                    f"Collection '{collection_name}' does not exist. "
                    f"Please create the collection first using QdrantClientCustom.create_collection()"
                )

            # Warn about loading to existing collection
            logger.warning(
                f"Loading {len(documents)} documents to collection '{collection_name}'. "
                f"Documents with same doc_id will overwrite existing points."
            )

            # Generate embeddings and create points
            embeddings = self.generate_embeddings(documents)
            points = self.create_points(documents, embeddings)

            # Upload points in batches
            logger.info(f"Uploading {len(points)} points to collection '{collection_name}' in batches of {batch_size}")

            total_uploaded = 0
            for i in range(0, len(points), batch_size):
                try:
                    batch = points[i : i + batch_size]
                    self.qdrant_client.qdrant.upsert(collection_name=collection_name, points=batch)
                    total_uploaded += len(batch)
                    logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} points")
                except Exception as e:
                    logger.error(f"Failed to upload batch {i//batch_size + 1}: {e}")
                    raise RuntimeError(f"Batch upload failed: {e}") from e

            logger.info(f"Successfully uploaded {total_uploaded} points to collection '{collection_name}'")
            return total_uploaded

        except ValueError:
            # Re-raise ValueError as-is (collection doesn't exist)
            raise
        except Exception as e:
            logger.error(f"Failed to load documents to Qdrant: {e}")
            raise RuntimeError(f"Qdrant loading failed: {e}") from e


class QdrantVectorLoader:
    """High-level interface for loading documents to Qdrant."""

    def __init__(self, document_loader: Optional[DocumentLoader] = None, qdrant_client: Optional[QdrantClientCustom] = None):
        """Initialize the Qdrant vector loader.

        Args:
            document_loader: DocumentLoader instance for loading documents
            qdrant_client: Optional QdrantClientCustom instance for shared client usage
        """
        self.document_loader = document_loader or DocumentLoader()
        self.vector_store = VectorStoreLoader(qdrant_client=qdrant_client)

    def setup_collection(
        self,
        collection_name: str,
        course_filter: Optional[str] = None,
        delete_if_exists: bool = False,
    ) -> Dict[str, Any]:
        """Set up a Qdrant collection with documents.

        Args:
            collection_name: Name of the collection to create
            course_filter: Optional course filter to limit documents
            delete_if_exists: Whether to delete existing collection

        Returns:
            Dictionary with setup results

        Raises:
            RuntimeError: If setup fails
        """
        try:
            logger.info(f"Setting up Qdrant collection '{collection_name}'")

            # Create collection first (separate from data loading)
            self.vector_store.qdrant_client.create_collection(
                collection_name=collection_name,
                vector_size=EMBEDDING_DIMENSIONALITY,
                distance=Distance.COSINE,
                delete_if_exists=delete_if_exists,
            )

            # Load documents using DocumentLoader
            documents = self.document_loader.load_documents()

            # Filter by course if requested
            if course_filter:
                documents = [doc for doc in documents if doc["course"] == course_filter]
                logger.info(f"Filtered documents by course '{course_filter}': {len(documents)} documents")

            # Load to Qdrant using VectorStoreLoader (now assumes collection exists)
            points_uploaded = self.vector_store.load_to_qdrant(
                collection_name=collection_name,
                documents=documents,
            )

            return {
                "collection_name": collection_name,
                "documents_loaded": len(documents),
                "points_uploaded": points_uploaded,
                "course_filter": course_filter,
            }

        except Exception as e:
            logger.error(f"Failed to setup collection '{collection_name}': {e}")
            raise RuntimeError(f"Collection setup failed: {e}") from e
