"""
Tests for the Course enum model.
"""

import unittest

from ..models.course import Course


class TestCourse(unittest.TestCase):
    """Test suite for the Course enum."""

    def test_course_values(self):
        """Test that all course values are correct."""
        self.assertEqual(Course.DATA_ENGINEERING_ZOOMCAMP.value, "data-engineering-zoomcamp")
        self.assertEqual(Course.MACHINE_LEARNING_ZOOMCAMP.value, "machine-learning-zoomcamp")
        self.assertEqual(Course.MLOPS_ZOOMCAMP.value, "mlops-zoomcamp")
        self.assertEqual(Course.LLM_ZOOMCAMP.value, "llm-zoomcamp")

    def test_str_representation(self):
        """Test string representation of courses."""
        self.assertEqual(str(Course.DATA_ENGINEERING_ZOOMCAMP), "data-engineering-zoomcamp")
        self.assertEqual(str(Course.MACHINE_LEARNING_ZOOMCAMP), "machine-learning-zoomcamp")
        self.assertEqual(str(Course.MLOPS_ZOOMCAMP), "mlops-zoomcamp")
        self.assertEqual(str(Course.LLM_ZOOMCAMP), "llm-zoomcamp")

    def test_from_string_valid(self):
        """Test creating Course from valid string values."""
        self.assertEqual(Course.from_string("data-engineering-zoomcamp"), Course.DATA_ENGINEERING_ZOOMCAMP)
        self.assertEqual(Course.from_string("machine-learning-zoomcamp"), Course.MACHINE_LEARNING_ZOOMCAMP)
        self.assertEqual(Course.from_string("mlops-zoomcamp"), Course.MLOPS_ZOOMCAMP)
        self.assertEqual(Course.from_string("llm-zoomcamp"), Course.LLM_ZOOMCAMP)

    def test_from_string_invalid(self):
        """Test creating Course from invalid string values."""
        with self.assertRaises(ValueError) as context:
            Course.from_string("invalid-course")
        self.assertIn("Unknown course: invalid-course", str(context.exception))

        with self.assertRaises(ValueError):
            Course.from_string("")

        with self.assertRaises(ValueError):
            Course.from_string("data-engineering")  # Partial match

    def test_list_courses(self):
        """Test listing all available courses."""
        courses = Course.list_courses()
        expected_courses = ["data-engineering-zoomcamp", "machine-learning-zoomcamp", "mlops-zoomcamp", "llm-zoomcamp"]

        self.assertEqual(len(courses), 4)
        for course in expected_courses:
            self.assertIn(course, courses)

    def test_enum_iteration(self):
        """Test iterating over Course enum."""
        course_values = [course.value for course in Course]
        expected_values = ["data-engineering-zoomcamp", "machine-learning-zoomcamp", "mlops-zoomcamp", "llm-zoomcamp"]

        self.assertEqual(len(course_values), 4)
        for value in expected_values:
            self.assertIn(value, course_values)


if __name__ == "__main__":
    unittest.main()
