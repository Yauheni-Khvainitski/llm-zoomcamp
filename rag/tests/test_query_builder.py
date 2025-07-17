"""
Tests for the QueryBuilder class.

Contains comprehensive tests for query building functionality.
"""

import unittest

from ..models.course import Course
from ..search.query_builder import QueryBuilder


class TestQueryBuilder(unittest.TestCase):
    """Test suite for the QueryBuilder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.query_builder = QueryBuilder()

    def test_with_course_filter(self):
        """Test build_search_query with a course filter."""
        query = self.query_builder.build_search_query(
            question="How do I copy files to a Docker container?",
            course_filter=Course.MACHINE_LEARNING_ZOOMCAMP,
            num_results=3,
            boost=4,
        )

        # Test basic structure
        self.assertIsInstance(query, dict)
        self.assertIn("size", query)
        self.assertIn("query", query)

        # Test size parameter
        self.assertEqual(query["size"], 3)

        # Test query structure with course filter
        self.assertIn("bool", query["query"])
        bool_query = query["query"]["bool"]

        self.assertIn("must", bool_query)
        self.assertIn("filter", bool_query)

        # Test multi_match structure
        multi_match = bool_query["must"]["multi_match"]
        self.assertEqual(multi_match["query"], "How do I copy files to a Docker container?")
        self.assertIn("question^4", multi_match["fields"])
        self.assertIn("text", multi_match["fields"])
        self.assertIn("section", multi_match["fields"])
        self.assertEqual(multi_match["type"], "best_fields")

        # Test course filter
        course_filter = bool_query["filter"]["term"]
        self.assertEqual(course_filter["course"], "machine-learning-zoomcamp")

    def test_without_course_filter(self):
        """Test build_search_query without a course filter."""
        query = self.query_builder.build_search_query(
            question="How do I copy files to a Docker container?", num_results=5, boost=2
        )

        # Test basic structure
        self.assertIsInstance(query, dict)
        self.assertIn("size", query)
        self.assertIn("query", query)

        # Test size parameter
        self.assertEqual(query["size"], 5)

        # Test query structure without course filter
        self.assertIn("multi_match", query["query"])
        self.assertNotIn("bool", query["query"])

        # Test multi_match structure
        multi_match = query["query"]["multi_match"]
        self.assertEqual(multi_match["query"], "How do I copy files to a Docker container?")
        self.assertIn("question^2", multi_match["fields"])
        self.assertIn("text", multi_match["fields"])
        self.assertIn("section", multi_match["fields"])
        self.assertEqual(multi_match["type"], "best_fields")

    def test_different_courses(self):
        """Test build_search_query with different course enum values."""
        courses_to_test = [
            Course.DATA_ENGINEERING_ZOOMCAMP,
            Course.MACHINE_LEARNING_ZOOMCAMP,
            Course.MLOPS_ZOOMCAMP,
            Course.LLM_ZOOMCAMP,
        ]

        for course in courses_to_test:
            query = self.query_builder.build_search_query(question="Test question", course_filter=course)

            # Test that the correct course value is used
            expected_course = course.value
            actual_course = query["query"]["bool"]["filter"]["term"]["course"]
            self.assertEqual(actual_course, expected_course)

    def test_default_parameters(self):
        """Test build_search_query with default parameters."""
        query = self.query_builder.build_search_query(question="Test question")

        # Should use defaults: course_filter=None, num_results=5, boost=4
        self.assertEqual(query["size"], 5)

        # Should not have course filter (direct multi_match)
        self.assertIn("multi_match", query["query"])

        # Should use default boost of 4
        fields = query["query"]["multi_match"]["fields"]
        self.assertIn("question^4", fields)

    def test_edge_cases(self):
        """Test edge cases and parameter validation."""
        # Test with empty question
        query = self.query_builder.build_search_query(question="")
        self.assertEqual(query["query"]["multi_match"]["query"], "")

        # Test with very high boost
        query = self.query_builder.build_search_query(question="test", boost=100)
        self.assertIn("question^100", query["query"]["multi_match"]["fields"])

        # Test with very high num_results
        query = self.query_builder.build_search_query(question="test", num_results=1000)
        self.assertEqual(query["size"], 1000)

    def test_match_all_query(self):
        """Test build_match_all_query method."""
        query = self.query_builder.build_match_all_query(num_results=10)

        self.assertEqual(query["size"], 10)
        self.assertIn("match_all", query["query"])
        self.assertEqual(query["query"]["match_all"], {})

    def test_term_query(self):
        """Test build_term_query method."""
        query = self.query_builder.build_term_query(field="course", value="data-engineering-zoomcamp", num_results=3)

        self.assertEqual(query["size"], 3)
        self.assertIn("term", query["query"])
        self.assertEqual(query["query"]["term"]["course"], "data-engineering-zoomcamp")

    def test_custom_defaults(self):
        """Test QueryBuilder with custom defaults."""
        custom_builder = QueryBuilder(default_num_results=10, default_boost=2)

        query = custom_builder.build_search_query(question="Test")

        self.assertEqual(query["size"], 10)
        self.assertIn("question^2", query["query"]["multi_match"]["fields"])

    def test_with_string_course_filter(self):
        """Test build_search_query with a string course filter."""
        query = self.query_builder.build_search_query(
            question="How do I copy files to a Docker container?",
            course_filter="machine-learning-zoomcamp",
            num_results=3,
            boost=4,
        )

        # Test basic structure
        self.assertIsInstance(query, dict)
        self.assertIn("size", query)
        self.assertIn("query", query)

        # Test size parameter
        self.assertEqual(query["size"], 3)

        # Test query structure with course filter
        self.assertIn("bool", query["query"])
        bool_query = query["query"]["bool"]

        self.assertIn("must", bool_query)
        self.assertIn("filter", bool_query)

        # Test multi_match structure
        multi_match = bool_query["must"]["multi_match"]
        self.assertEqual(multi_match["query"], "How do I copy files to a Docker container?")
        self.assertIn("question^4", multi_match["fields"])
        self.assertIn("text", multi_match["fields"])
        self.assertIn("section", multi_match["fields"])
        self.assertEqual(multi_match["type"], "best_fields")

        # Test course filter - should use string value directly
        course_filter = bool_query["filter"]["term"]
        self.assertEqual(course_filter["course"], "machine-learning-zoomcamp")

    def test_different_string_courses(self):
        """Test build_search_query with different string course values."""
        courses_to_test = [
            "data-engineering-zoomcamp",
            "machine-learning-zoomcamp",
            "mlops-zoomcamp",
            "llm-zoomcamp",
        ]

        for course_string in courses_to_test:
            query = self.query_builder.build_search_query(question="Test question", course_filter=course_string)

            # Test that the correct course value is used
            actual_course = query["query"]["bool"]["filter"]["term"]["course"]
            self.assertEqual(actual_course, course_string)

    def test_course_filter_enum_vs_string_equivalence(self):
        """Test that Course enum and equivalent string produce the same query."""
        question = "Test question"

        # Query with Course enum
        query_enum = self.query_builder.build_search_query(question=question, course_filter=Course.MACHINE_LEARNING_ZOOMCAMP)

        # Query with equivalent string
        query_string = self.query_builder.build_search_query(question=question, course_filter="machine-learning-zoomcamp")

        # Both should produce identical queries
        self.assertEqual(query_enum, query_string)


def run_tests():
    """
    Run all QueryBuilder tests.

    Returns:
        True if all tests pass, False otherwise
    """
    print("🚀 Starting QueryBuilder validation tests...\n")

    try:
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestQueryBuilder)

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("\n🎉 ALL TESTS PASSED! The QueryBuilder is working correctly.")
            return True
        else:
            print(f"\n❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
            return False

    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        return False


if __name__ == "__main__":
    run_tests()
