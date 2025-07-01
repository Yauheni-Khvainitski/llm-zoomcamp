"""Qdrant client wrapper for RAG system."""

import logging
from typing import Optional

from qdrant_client import QdrantClient

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
            logger.info(f"Connected to Qdrant: {self.qdrant_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.qdrant_url}: {e}")
            raise
