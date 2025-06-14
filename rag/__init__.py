"""
RAG (Retrieval-Augmented Generation) Package

A comprehensive RAG system for Q&A over course documents using Elasticsearch and OpenAI.
"""

__version__ = "1.0.0"
__author__ = "DataTalks.Club"

from .models.course import Course
from .pipeline.rag import RAGPipeline

__all__ = ["Course", "RAGPipeline"]
