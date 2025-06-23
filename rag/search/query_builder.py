"""
Query builder for Elasticsearch searches.

Handles building search queries with optional course filtering.
"""

import logging
from typing import Any, Dict, Optional

from ..config import DEFAULT_BOOST_FACTOR, DEFAULT_NUM_RESULTS
from ..models.course import Course

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Builds Elasticsearch queries for document search."""

    def __init__(self, default_num_results: int = DEFAULT_NUM_RESULTS, default_boost: int = DEFAULT_BOOST_FACTOR):
        """
        Initialize the query builder.

        Args:
            default_num_results: Default number of results to return
            default_boost: Default boost factor for question field
        """
        self.default_num_results = default_num_results
        self.default_boost = default_boost

    def build_search_query(
        self, question: str, course_filter: Optional[Course] = None, num_results: int = None, boost: int = None
    ) -> Dict[str, Any]:
        """
        Build an Elasticsearch search query.

        Args:
            question: The question to search for
            course_filter: Optional course to filter by
            num_results: Number of results to return
            boost: Boost factor for the question field

        Returns:
            Elasticsearch search query dictionary
        """
        # Use defaults if not provided
        if num_results is None:
            num_results = self.default_num_results
        if boost is None:
            boost = self.default_boost

        # Build the base query structure
        query_structure = {
            "multi_match": {"query": question, "fields": [f"question^{boost}", "text", "section"], "type": "best_fields"}
        }

        # Conditionally add the course filter
        if course_filter is not None:
            search_query = {
                "size": num_results,
                "query": {"bool": {"must": query_structure, "filter": {"term": {"course": course_filter.value}}}},
            }
            logger.debug(f"Built query with course filter: {course_filter.value}")
        else:
            # No course filter - search across all courses
            search_query = {"size": num_results, "query": query_structure}
            logger.debug("Built query without course filter")

        return search_query

    def build_match_all_query(self, num_results: int = None) -> Dict[str, Any]:
        """
        Build a query that matches all documents.

        Args:
            num_results: Number of results to return

        Returns:
            Elasticsearch match_all query
        """
        if num_results is None:
            num_results = self.default_num_results

        return {"size": num_results, "query": {"match_all": {}}}

    def build_term_query(self, field: str, value: str, num_results: int = None) -> Dict[str, Any]:
        """
        Build a simple term query for exact matches.

        Args:
            field: Field to search in
            value: Value to search for
            num_results: Number of results to return

        Returns:
            Elasticsearch term query
        """
        if num_results is None:
            num_results = self.default_num_results

        return {"size": num_results, "query": {"term": {field: value}}}
