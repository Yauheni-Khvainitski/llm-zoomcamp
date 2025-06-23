"""
Tests for the ContextFormatter class.
"""

import unittest

from ..formatting.context import ContextFormatter


class TestContextFormatter(unittest.TestCase):
    """Test suite for the ContextFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ContextFormatter()
        self.sample_documents = [
            {
                "question": "What is Docker?",
                "text": "Docker is a containerization platform that allows you to package applications.",
            },
            {
                "question": "How to install Docker?",
                "text": "You can install Docker by downloading it from the official website.",
            },
        ]

    def test_init_default_templates(self):
        """Test initialization with default templates."""
        formatter = ContextFormatter()

        # Check default context template
        expected_context = "Q: {question}\nA: {text}"
        self.assertEqual(formatter.context_template, expected_context)

        # Check default prompt template contains expected elements
        self.assertIn("course teaching assistant", formatter.prompt_template)
        self.assertIn("QUESTION:", formatter.prompt_template)
        self.assertIn("CONTEXT:", formatter.prompt_template)

    def test_init_custom_templates(self):
        """Test initialization with custom templates."""
        custom_context = "Question: {question}\nAnswer: {text}"
        custom_prompt = "Custom prompt with {question} and {context}"

        formatter = ContextFormatter(context_template=custom_context, prompt_template=custom_prompt)

        self.assertEqual(formatter.context_template, custom_context)
        self.assertEqual(formatter.prompt_template, custom_prompt)

    def test_format_document_success(self):
        """Test successful document formatting."""
        doc = self.sample_documents[0]
        result = self.formatter.format_document(doc)

        expected = "Q: What is Docker?\nA: Docker is a containerization platform that allows you to package applications."
        self.assertEqual(result, expected)

    def test_format_document_missing_question(self):
        """Test document formatting with missing question field."""
        doc = {"text": "Some text"}

        with self.assertRaises(KeyError):
            self.formatter.format_document(doc)

    def test_format_document_missing_text(self):
        """Test document formatting with missing text field."""
        doc = {"question": "Some question"}

        with self.assertRaises(KeyError):
            self.formatter.format_document(doc)

    def test_format_document_empty_fields(self):
        """Test document formatting with empty fields."""
        doc = {"question": "", "text": ""}
        result = self.formatter.format_document(doc)

        expected = "Q: \nA: "
        self.assertEqual(result, expected)

    def test_format_context_multiple_documents(self):
        """Test formatting multiple documents into context."""
        result = self.formatter.format_context(self.sample_documents)

        expected = (
            "Q: What is Docker?\n"
            "A: Docker is a containerization platform that allows you to package applications.\n\n"
            "Q: How to install Docker?\n"
            "A: You can install Docker by downloading it from the official website."
        )

        self.assertEqual(result, expected)

    def test_format_context_single_document(self):
        """Test formatting single document into context."""
        result = self.formatter.format_context([self.sample_documents[0]])

        expected = "Q: What is Docker?\n" "A: Docker is a containerization platform that allows you to package applications."

        self.assertEqual(result, expected)

    def test_format_context_empty_list(self):
        """Test formatting empty document list."""
        result = self.formatter.format_context([])
        self.assertEqual(result, "")

    def test_format_context_strips_whitespace(self):
        """Test that context formatting strips leading/trailing whitespace."""
        docs = [{"question": "Q1", "text": "A1"}]
        result = self.formatter.format_context(docs)

        # Should not have leading or trailing whitespace
        self.assertEqual(result, result.strip())

    def test_build_prompt_success(self):
        """Test successful prompt building."""
        question = "What is containerization?"
        context = "Q: What is Docker?\nA: Docker is a containerization platform."

        result = self.formatter.build_prompt(question, context)

        self.assertIn("course teaching assistant", result)
        self.assertIn("QUESTION: What is containerization?", result)
        self.assertIn("CONTEXT:", result)
        self.assertIn("Q: What is Docker?", result)
        self.assertIn("A: Docker is a containerization platform.", result)

    def test_build_prompt_empty_context(self):
        """Test prompt building with empty context."""
        question = "What is containerization?"
        context = ""

        result = self.formatter.build_prompt(question, context)

        self.assertIn("QUESTION: What is containerization?", result)
        self.assertIn("CONTEXT:\n", result)

    def test_build_prompt_empty_question(self):
        """Test prompt building with empty question."""
        question = ""
        context = "Some context"

        result = self.formatter.build_prompt(question, context)

        self.assertIn("QUESTION: ", result)
        self.assertIn("CONTEXT:\nSome context", result)

    def test_build_prompt_from_documents(self):
        """Test building prompt directly from documents."""
        question = "What is containerization?"

        result = self.formatter.build_prompt_from_documents(question, self.sample_documents)

        self.assertIn("QUESTION: What is containerization?", result)
        self.assertIn("Q: What is Docker?", result)
        self.assertIn("Q: How to install Docker?", result)

    def test_build_prompt_from_empty_documents(self):
        """Test building prompt from empty document list."""
        question = "What is containerization?"

        result = self.formatter.build_prompt_from_documents(question, [])

        self.assertIn("QUESTION: What is containerization?", result)
        self.assertIn("CONTEXT:\n", result)

    def test_set_context_template(self):
        """Test setting custom context template."""
        new_template = "Question: {question}\nAnswer: {text}"
        self.formatter.set_context_template(new_template)

        self.assertEqual(self.formatter.context_template, new_template)

        # Test that new template is used
        doc = self.sample_documents[0]
        result = self.formatter.format_document(doc)

        expected = (
            "Question: What is Docker?\nAnswer: Docker is a containerization platform that allows you to package applications."
        )
        self.assertEqual(result, expected)

    def test_set_prompt_template(self):
        """Test setting custom prompt template."""
        new_template = "Answer this: {question}\nUsing: {context}"
        self.formatter.set_prompt_template(new_template)

        self.assertEqual(self.formatter.prompt_template, new_template)

        # Test that new template is used
        result = self.formatter.build_prompt("Test question", "Test context")

        expected = "Answer this: Test question\nUsing: Test context"
        self.assertEqual(result, expected)

    def test_get_context_stats(self):
        """Test getting context statistics."""
        context = (
            "Q: What is Docker?\n"
            "A: Docker is a containerization platform.\n\n"
            "Q: How to install Docker?\n"
            "A: You can install Docker by downloading it."
        )

        stats = self.formatter.get_context_stats(context)

        self.assertEqual(stats["total_characters"], len(context))
        self.assertEqual(stats["total_lines"], 5)  # Including empty line
        self.assertEqual(stats["non_empty_lines"], 4)
        self.assertEqual(stats["q_count"], 2)
        self.assertEqual(stats["a_count"], 2)

    def test_get_context_stats_empty(self):
        """Test getting statistics for empty context."""
        stats = self.formatter.get_context_stats("")

        self.assertEqual(stats["total_characters"], 0)
        self.assertEqual(stats["total_lines"], 1)  # Empty string has 1 line
        self.assertEqual(stats["non_empty_lines"], 0)
        self.assertEqual(stats["q_count"], 0)
        self.assertEqual(stats["a_count"], 0)

    def test_get_context_stats_no_qa(self):
        """Test getting statistics for context without Q: and A: markers."""
        context = "This is just plain text\nwith multiple lines\nbut no question or answer markers"

        stats = self.formatter.get_context_stats(context)

        self.assertEqual(stats["total_characters"], len(context))
        self.assertEqual(stats["total_lines"], 3)
        self.assertEqual(stats["non_empty_lines"], 3)
        self.assertEqual(stats["q_count"], 0)
        self.assertEqual(stats["a_count"], 0)

    def test_custom_context_template_with_extra_fields(self):
        """Test custom context template that uses additional document fields."""
        custom_template = "Section: {section}\nQ: {question}\nA: {text}"
        formatter = ContextFormatter(context_template=custom_template)

        doc = {"question": "What is Docker?", "text": "Docker is a platform.", "section": "Containerization"}

        result = formatter.format_document(doc)
        expected = "Section: Containerization\nQ: What is Docker?\nA: Docker is a platform."
        self.assertEqual(result, expected)

    def test_format_document_with_special_characters(self):
        """Test document formatting with special characters."""
        doc = {
            "question": "What is 'Docker' & how does it work?",
            "text": 'Docker uses "containers" to package apps & their dependencies.',
        }

        result = self.formatter.format_document(doc)

        self.assertIn("What is 'Docker' & how does it work?", result)
        self.assertIn('Docker uses "containers" to package apps & their dependencies.', result)


if __name__ == "__main__":
    unittest.main()
