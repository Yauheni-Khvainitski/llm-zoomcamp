"""
Elasticsearch client for RAG system.

Handles connection to Elasticsearch, index management, and document operations.
"""

import logging
from typing import Any, Dict, List, Union

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import BadRequestError, NotFoundError
from tqdm.auto import tqdm

from ..config import DEFAULT_INDEX_NAME, ELASTICSEARCH_URL, INDEX_SETTINGS

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client for document indexing and searching."""

    def __init__(self, es_url: str = ELASTICSEARCH_URL, index_name: str = DEFAULT_INDEX_NAME):
        """
        Initialize the Elasticsearch client.

        Args:
            es_url: Elasticsearch URL
            index_name: Default index name
        """
        self.es_url = es_url
        self.index_name = index_name
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

    def create_index(self, index_name: str = None, settings: Dict[str, Any] = None, delete_if_exists: bool = True) -> bool:
        """
        Create an Elasticsearch index.

        Args:
            index_name: Name of the index to create
            settings: Index settings and mappings
            delete_if_exists: Whether to delete existing index

        Returns:
            True if index was created successfully
        """
        if index_name is None:
            index_name = self.index_name
        if settings is None:
            settings = INDEX_SETTINGS

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

    def delete_index(self, index_name: str = None) -> bool:
        """
        Delete an Elasticsearch index.

        Args:
            index_name: Name of the index to delete

        Returns:
            True if index was deleted or didn't exist
        """
        if index_name is None:
            index_name = self.index_name

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

    def index_document(self, document: Dict[str, Any], doc_id: str = None, index_name: str = None) -> bool:
        """
        Index a single document.

        Args:
            document: Document to index
            doc_id: Document ID (uses document's doc_id if not provided)
            index_name: Index name

        Returns:
            True if document was indexed successfully
        """
        if index_name is None:
            index_name = self.index_name
        if doc_id is None:
            doc_id = document.get("doc_id")

        try:
            self.es.index(index=index_name, id=doc_id, document=document)
            return True
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}: {e}")
            raise

    def index_documents(self, documents: List[Dict[str, Any]], index_name: str = None, show_progress: bool = True) -> int:
        """
        Index multiple documents.

        Args:
            documents: List of documents to index
            index_name: Index name
            show_progress: Whether to show progress bar

        Returns:
            Number of successfully indexed documents
        """
        if index_name is None:
            index_name = self.index_name

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
        self, query: Dict[str, Any], index_name: str = None, return_raw: bool = False
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search documents in Elasticsearch.

        Args:
            query: Elasticsearch query
            index_name: Index name
            return_raw: Whether to return raw Elasticsearch response

        Returns:
            List of documents or raw response
        """
        if index_name is None:
            index_name = self.index_name

        try:
            response = self.es.search(index=index_name, body=query)

            if return_raw:
                return response
            else:
                return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    def count_documents(self, index_name: str = None) -> int:
        """
        Count documents in an index.

        Args:
            index_name: Index name

        Returns:
            Number of documents in the index
        """
        if index_name is None:
            index_name = self.index_name

        try:
            response = self.es.count(index=index_name)
            return response["count"]
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            raise

    def index_exists(self, index_name: str = None) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Index name

        Returns:
            True if index exists
        """
        if index_name is None:
            index_name = self.index_name

        return self.es.indices.exists(index=index_name)
