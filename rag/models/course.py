"""
Course enumeration for filtering search results.
"""

from enum import Enum


class Course(Enum):
    """Enum class for available course values."""

    DATA_ENGINEERING_ZOOMCAMP = "data-engineering-zoomcamp"
    MACHINE_LEARNING_ZOOMCAMP = "machine-learning-zoomcamp"
    MLOPS_ZOOMCAMP = "mlops-zoomcamp"
    LLM_ZOOMCAMP = "llm-zoomcamp"

    def __str__(self) -> str:
        """Return the string representation of the course."""
        return self.value

    @classmethod
    def from_string(cls, course_name: str) -> "Course":
        """Create Course enum from string value."""
        for course in cls:
            if course.value == course_name:
                return course
        raise ValueError(f"Unknown course: {course_name}")

    @classmethod
    def list_courses(cls) -> list[str]:
        """Return a list of all available course names."""
        return [course.value for course in cls]
