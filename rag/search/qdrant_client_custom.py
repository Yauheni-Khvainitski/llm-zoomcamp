"""Qdrant client wrapper for RAG system."""

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, VectorParams

from ..config import QDRANT_URL

logger = logging.getLogger(__name__)


class QdrantClientCustom:
    """Custom Qdrant client wrapper for RAG operations."""

    def __init__(self, qdrant_url: Optional[str] = None):
        """Initialize the Qdrant client.

        Args:
            qdrant_url: The URL of the Qdrant server.
                       If not provided, uses the default from config.

        Raises:
            Exception: If connection to Qdrant fails.
        """
        self.qdrant_url = qdrant_url or QDRANT_URL

        try:
            self.qdrant = QdrantClient(url=self.qdrant_url)
            # Test the connection by trying to get collections
            self.qdrant.get_collections()
            logger.info(f"Connected to Qdrant: {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {e}")
            raise

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 512,
        distance: Distance = Distance.COSINE,
        delete_if_exists: bool = False,
    ) -> bool:
        """Create a Qdrant collection.

        Args:
            collection_name: Name of the collection to create (REQUIRED)
            vector_size: Size of the vectors to be stored in the collection
            distance: Distance metric to use for similarity search
            delete_if_exists: Whether to delete existing collection (DANGEROUS - will delete all data!)
        """
        # Safety check: warn about destructive operation
        if delete_if_exists and self.collection_exists(collection_name):
            logger.warning(
                f"⚠️  DESTRUCTIVE OPERATION: About to delete existing collection '{collection_name}' and all its data!"
            )

        # Check if collection already exists
        if self.collection_exists(collection_name) and not delete_if_exists:
            logger.info(f"Collection '{collection_name}' already exists. Use delete_if_exists=True to overwrite.")
            return True

        # Delete existing collection if requested
        if delete_if_exists:
            self.delete_collection(collection_name)

        # Create new collection
        try:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            logger.info(f"Created collection '{collection_name}'")
            return True
        except UnexpectedResponse as e:
            if e.status_code == 409:
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            else:
                logger.error(f"Error creating collection '{collection_name}': {e}")
                raise
        except Exception as e:
            logger.error(f"Error creating collection '{collection_name}': {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise
        """
        logger.info(f"Checking if collection '{collection_name}' exists")
        try:
            self.qdrant.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' exists")
            return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.info(f"Collection '{collection_name}' does not exist")
                return False
            else:
                # Re-raise other HTTP errors (like connection issues)
                logger.error(f"Error checking collection '{collection_name}': {e}")
                raise
        except Exception as e:
            # Re-raise connection errors and other exceptions
            logger.error(f"Error checking collection '{collection_name}': {e}")
            raise

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a Qdrant collection.

        Args:
            collection_name: Name of the collection to delete (REQUIRED)

        Returns:
            True if collection was deleted or didn't exist
        """
        try:
            if self.collection_exists(collection_name):
                self.qdrant.delete_collection(collection_name)
                logger.info(f"Deleted collection '{collection_name}'")
                return True
            else:
                logger.info(f"Collection '{collection_name}' does not exist, no need to delete")
                return True
        except UnexpectedResponse as e:
            if e.status_code == 404:
                logger.info(f"Collection '{collection_name}' does not exist, no need to delete")
                return True
            else:
                # Re-raise other HTTP errors
                logger.error(f"Error deleting collection '{collection_name}': {e}")
                raise
        except Exception as e:
            logger.error(f"Error deleting collection '{collection_name}': {e}")
            raise

    def search_with_vector(
        self,
        query_vector: List[float],
        collection_name: str,
        limit: int = 5,
        course_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
        with_payload: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search documents in Qdrant using pre-computed vector.

        Args:
            query_vector: Pre-computed query vector for similarity search
            collection_name: Collection name (REQUIRED)
            limit: Number of results to return
            course_filter: Optional course to filter by
            score_threshold: Minimum similarity score threshold
            with_payload: Whether to include document payload in results

        Returns:
            List of search results with documents and scores
        """
        try:
            # Build filter if course is specified
            query_filter = None
            if course_filter:
                query_filter = Filter(must=[FieldCondition(key="course", match=MatchValue(value=course_filter))])

            # Perform vector search
            logger.info(f"Searching in collection '{collection_name}' with limit {limit}")
            response = self.qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=with_payload,
            )

            # Format results
            results = []
            for point in response:
                result: Dict[str, Any] = {
                    "id": point.id,
                    "score": point.score,
                }
                if with_payload and point.payload:
                    result["payload"] = dict(point.payload)
                results.append(result)

            logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def create_payload_index(
        self, collection_name: str, field_name: str, field_schema: PayloadSchemaType = PayloadSchemaType.KEYWORD
    ) -> bool:
        """Create a payload index for a specific field in a Qdrant collection.

        Args:
            collection_name: Name of the collection to create payload index for
            field_name: Name of the payload field to index
            field_schema: Schema type for the field (default: KEYWORD for exact matching)

        Returns:
            True if payload index was created successfully

        Raises:
            Exception: If index creation fails
        """
        try:
            logger.info(f"Creating payload index for field '{field_name}' in collection '{collection_name}'")

            self.qdrant.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

            logger.info(f"Successfully created payload index for field '{field_name}' in collection '{collection_name}'")
            return True

        except UnexpectedResponse as e:
            if e.status_code == 409:
                logger.info(f"Payload index for field '{field_name}' already exists in collection '{collection_name}'")
                return True
            else:
                logger.error(f"Error creating payload index for field '{field_name}': {e}")
                raise
        except Exception as e:
            logger.error(f"Error creating payload index for field '{field_name}': {e}")
            raise
