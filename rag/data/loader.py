"""
Document loader for RAG system.

Handles loading documents from external sources and preprocessing them.
"""

import hashlib
import json
import logging
from typing import Any, Dict, List

import requests

from ..config import DOCUMENTS_URL

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Handles loading and preprocessing of documents for the RAG system."""

    def __init__(self, documents_url: str = DOCUMENTS_URL):
        """
        Initialize the document loader.

        Args:
            documents_url: URL to fetch documents from
        """
        self.documents_url = documents_url
        self.documents: List[Dict[str, Any]] = []

    def generate_id(self, doc: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a document using MD5 hash.

        Args:
            doc: Document dictionary

        Returns:
            Unique document ID
        """
        doc_str = json.dumps(doc, sort_keys=True)
        doc_hash = hashlib.md5(doc_str.encode(), usedforsecurity=False).hexdigest()
        return doc_hash

    def fetch_documents(self) -> List[Dict[str, Any]]:
        """
        Fetch documents from the configured URL.

        Returns:
            Raw documents data

        Raises:
            requests.RequestException: If fetching documents fails
        """
        try:
            logger.info(f"Fetching documents from {self.documents_url}")
            response = requests.get(self.documents_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data  # type: ignore[return-value]
        except requests.RequestException as e:
            logger.error(f"Failed to fetch documents: {e}")
            raise

    def process_documents(self, documents_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw documents by adding course information and unique IDs.

        Args:
            documents_raw: Raw documents from the source

        Returns:
            Processed documents with course and doc_id fields
        """
        processed_documents = []

        for course in documents_raw:
            course_name = course["course"]

            for doc in course["documents"]:
                # Add course information
                doc["course"] = course_name

                # Generate and add unique ID
                doc_id = self.generate_id(doc)
                doc["doc_id"] = doc_id

                processed_documents.append(doc)

        logger.info(f"Processed {len(processed_documents)} documents")
        return processed_documents

    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load and process documents from the configured source.

        Returns:
            List of processed documents
        """
        if not self.documents:
            documents_raw = self.fetch_documents()
            self.documents = self.process_documents(documents_raw)

        return self.documents

    def get_documents_by_course(self, course: str) -> List[Dict[str, Any]]:
        """
        Get documents filtered by course.

        Args:
            course: Course name to filter by

        Returns:
            Documents from the specified course
        """
        if not self.documents:
            self.load_documents()

        return [doc for doc in self.documents if doc["course"] == course]

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded documents.

        Returns:
            Dictionary with document statistics
        """
        if not self.documents:
            self.load_documents()

        courses: Dict[str, int] = {}
        for doc in self.documents:
            course = doc["course"]
            courses[course] = courses.get(course, 0) + 1

        return {"total_documents": len(self.documents), "documents_by_course": courses, "unique_courses": len(courses)}
