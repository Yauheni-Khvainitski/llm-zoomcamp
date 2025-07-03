"""
Elasticsearch client for RAG system.

Handles connection to Elasticsearch, index management, and document operations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from tqdm.auto import tqdm

from ..config import ELASTICSEARCH_URL, INDEX_SETTINGS

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for document indexing and searching."""

    def __init__(self, es_url: str = ELASTICSEARCH_URL):
        """
        Initialize the Elasticsearch client.

        Args:
            es_url: Elasticsearch URL
        """
        self.es_url = es_url

        self.es = Elasticsearch(hosts=es_url)

        # Test connection
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch: {info['cluster_name']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def get_client(self) -> Elasticsearch:
        """
        Get the Elasticsearch client instance.

        Returns:
            Elasticsearch client
        """
        return self.es

    def create_index(self, index_name: str, settings: Optional[Dict[str, Any]] = None, delete_if_exists: bool = False) -> bool:
        """
        Create an Elasticsearch index.

        Args:
            index_name: Name of the index to create (REQUIRED)
            settings: Index settings and mappings
            delete_if_exists: Whether to delete existing index (DANGEROUS - will delete all data!)

        Returns:
            True if index was created successfully

        Raises:
            Exception: If index already exists and delete_if_exists is False
        """
        if settings is None:
            settings = INDEX_SETTINGS

        # Safety check: warn about destructive operation
        if delete_if_exists and self.index_exists(index_name):
            logger.warning(f"⚠️  DESTRUCTIVE OPERATION: About to delete existing index '{index_name}' and all its data!")

        # Check if index already exists
        if self.index_exists(index_name) and not delete_if_exists:
            logger.info(f"Index '{index_name}' already exists. Use delete_if_exists=True to overwrite.")
            return True

        # Delete existing index if requested
        if delete_if_exists:
            self.delete_index(index_name)

        try:
            response = self.es.indices.create(index=index_name, settings=settings["settings"], mappings=settings["mappings"])
            logger.info(f"Created index '{index_name}': {response}")
            return True
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            raise

    def delete_index(self, index_name: str) -> bool:
        """
        Delete an Elasticsearch index.

        Args:
            index_name: Name of the index to delete (REQUIRED)

        Returns:
            True if index was deleted or didn't exist
        """
        try:
            self.es.indices.delete(index=index_name)
            logger.info(f"Deleted index '{index_name}'")
            return True
        except NotFoundError:
            logger.info(f"Index '{index_name}' does not exist, no need to delete")
            return True
        except Exception as e:
            logger.error(f"Error deleting index '{index_name}': {e}")
            raise

    def index_document(self, document: Dict[str, Any], index_name: str, doc_id: Optional[str] = None) -> bool:
        """
        Index a single document.

        Args:
            document: Document to index
            index_name: Index name (REQUIRED)
            doc_id: Document ID (uses document's doc_id if not provided)

        Returns:
            True if document was indexed successfully
        """
        if doc_id is None:
            doc_id = document.get("doc_id")

        try:
            self.es.index(index=index_name, id=doc_id, document=document)
            return True
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            raise

    def index_documents(self, documents: List[Dict[str, Any]], index_name: str, show_progress: bool = True) -> int:
        """
        Index multiple documents.

        Args:
            documents: List of documents to index
            index_name: Index name (REQUIRED)
            show_progress: Whether to show progress bar

        Returns:
            Number of successfully indexed documents
        """
        indexed_count = 0

        iterator = tqdm(documents, desc="Indexing documents") if show_progress else documents

        for doc in iterator:
            try:
                doc_id = doc.get("doc_id")
                self.es.index(index=index_name, id=doc_id, document=doc)
                indexed_count += 1
            except Exception as e:
                logger.error(f"Error indexing document {doc.get('doc_id')}: {e}")

        logger.info(f"Indexed {indexed_count}/{len(documents)} documents")
        return indexed_count

    def search_documents(
        self, query: Dict[str, Any], index_name: str, return_raw: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search documents in Elasticsearch.

        Args:
            query: Elasticsearch query
            index_name: Index name (REQUIRED)
            return_raw: Whether to return raw Elasticsearch response

        Returns:
            List of documents or raw response
        """
        try:
            response = self.es.search(index=index_name, body=query)

            if return_raw:
                return response  # type: ignore[return-value]
            else:
                return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def count_documents(self, index_name: str) -> int:
        """
        Count documents in an index.

        Args:
            index_name: Index name (REQUIRED)

        Returns:
            Number of documents in the index
        """
        try:
            response = self.es.count(index=index_name)
            return response["count"]  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            raise

    def count_documents_with_query(self, index_name: str, query: Dict[str, Any]) -> int:
        """
        Count documents in an index with a query.

        Args:
            index_name: Index name (REQUIRED)
            query: Query to filter documents

        Returns:
            Number of documents matching the query
        """
        try:
            response = self.es.count(index=index_name, body=query)
            return response["count"]  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"Error counting documents with query: {e}")
            raise

    def get_document(self, doc_id: str, index_name: str) -> Union[Dict[str, Any], None]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID to retrieve
            index_name: Index name (REQUIRED)

        Returns:
            Document if found, None if not found
        """
        try:
            response = self.es.get(index=index_name, id=doc_id)
            return response["_source"]  # type: ignore[no-any-return]
        except NotFoundError:
            logger.info(f"Document {doc_id} not found in index {index_name}")
            return None
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            raise

    def delete_document(self, doc_id: str, index_name: str) -> bool:
        """
        Delete a document by ID.

        Args:
            doc_id: Document ID to delete
            index_name: Index name (REQUIRED)

        Returns:
            True if document was deleted, False if not found
        """
        try:
            self.es.delete(index=index_name, id=doc_id)
            logger.info(f"Deleted document {doc_id} from index {index_name}")
            return True
        except NotFoundError:
            logger.info(f"Document {doc_id} not found in index {index_name}")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            raise

    def health_check(self) -> bool:
        """
        Check if Elasticsearch is available.

        Returns:
            True if Elasticsearch is available
        """
        try:
            return self.es.ping()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Index name (REQUIRED)

        Returns:
            True if index exists
        """
        logger.info(f"Checking if index '{index_name}' exists")
        exists = bool(self.es.indices.exists(index=index_name))
        logger.info(f"Index {index_name} exists: {exists}")
        return exists
