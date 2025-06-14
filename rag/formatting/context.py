"""
Context formatting for RAG system.

Handles formatting documents into context strings and building prompts for LLMs.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ContextFormatter:
    """Formats documents into context strings and builds prompts for LLMs."""

    def __init__(self, context_template: str = None, prompt_template: str = None):
        """
        Initialize the context formatter.

        Args:
            context_template: Template for formatting individual documents
            prompt_template: Template for the final prompt
        """
        self.context_template = context_template or self._get_default_context_template()
        self.prompt_template = prompt_template or self._get_default_prompt_template()

    def _get_default_context_template(self) -> str:
        """Get the default context template for Q&A formatting."""
        return """Q: {question}
A: {text}""".strip()

    def _get_default_prompt_template(self) -> str:
        """Get the default prompt template for the LLM."""
        return """You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}""".strip()

    def format_document(self, document: Dict[str, Any]) -> str:
        """
        Format a single document using the context template.

        Args:
            document: Document with 'question' and 'text' fields

        Returns:
            Formatted document string
        """
        try:
            return self.context_template.format(question=document["question"], text=document["text"])
        except KeyError as e:
            logger.error(f"Missing field in document: {e}")
            raise
        except Exception as e:
            logger.error(f"Error formatting document: {e}")
            raise

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format multiple documents into a context string.

        Args:
            documents: List of documents to format

        Returns:
            Formatted context string with all documents
        """
        if not documents:
            logger.warning("No documents provided for context formatting")
            return ""

        formatted_docs = []
        for doc in documents:
            formatted_doc = self.format_document(doc)
            formatted_docs.append(formatted_doc)

        # Join documents with double newlines
        context = "\n\n".join(formatted_docs)

        logger.debug(f"Formatted {len(documents)} documents into context")
        return context.strip()

    def build_prompt(self, question: str, context: str) -> str:
        """
        Build a complete prompt for the LLM.

        Args:
            question: The user's question
            context: The formatted context from documents

        Returns:
            Complete prompt for the LLM
        """
        try:
            prompt = self.prompt_template.format(question=question, context=context)
            logger.debug(f"Built prompt with {len(context)} context characters")
            return prompt
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            raise

    def build_prompt_from_documents(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """
        Build a complete prompt from question and documents.

        Args:
            question: The user's question
            documents: List of documents to use as context

        Returns:
            Complete prompt for the LLM
        """
        context = self.format_context(documents)
        return self.build_prompt(question, context)

    def set_context_template(self, template: str) -> None:
        """
        Set a custom context template.

        Args:
            template: New context template with {question} and {text} placeholders
        """
        self.context_template = template
        logger.info("Updated context template")

    def set_prompt_template(self, template: str) -> None:
        """
        Set a custom prompt template.

        Args:
            template: New prompt template with {question} and {context} placeholders
        """
        self.prompt_template = template
        logger.info("Updated prompt template")

    def get_context_stats(self, context: str) -> Dict[str, Any]:
        """
        Get statistics about a formatted context.

        Args:
            context: Formatted context string

        Returns:
            Dictionary with context statistics
        """
        lines = context.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        return {
            "total_characters": len(context),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "q_count": context.count("Q:"),
            "a_count": context.count("A:"),
        }
