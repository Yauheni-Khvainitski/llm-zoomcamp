#!/usr/bin/env python3
"""
Example usage of the RAG system.

This script demonstrates:
1. Full RAG pipeline with Elasticsearch (including question answering)
2. Basic vector store operations with Qdrant
"""

import logging
from dotenv import load_dotenv

from rag import RAGPipeline
from rag.data.vector_store import VectorSearcher, QdrantVectorLoader
from rag.search.qdrant_client_custom import QdrantClientCustom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_elasticsearch_rag():
    """Demonstrate full RAG pipeline with Elasticsearch - including question answering."""
    print("\n🤖 Full RAG Pipeline Demonstration (Elasticsearch)")
    print("=" * 60)
    
    rag = RAGPipeline()
    
    # Health check
    print("1. Health Check")
    print("-" * 30)
    health = rag.health_check()
    print(f"Status: {health}")
    
    if not health["elasticsearch"]: 
        print("❌ Elasticsearch not available - skipping RAG demo")
        print("   Run: docker-compose up -d")
        return
    
    # Setup and load documents
    print("\n2. Document Loading")
    print("-" * 30)
    setup_result = rag.setup_index()
    print(f"✅ Documents loaded: {setup_result['documents_loaded']}")
    
    # Interactive question answering
    print("\n3. Question Answering Demo")
    print("-" * 30)
    
    # Sample questions
    sample_question = "How can I join the course?"
    
    try:
        # Get answer from RAG pipeline
        answer = rag.ask(sample_question, search_engine="elasticsearch")

        print(f"\n🤖 Answer:")
        print("-" * 30)
        print(answer)
            
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        print(f"❌ Error: {e}")
    
    print("\n✅ Elasticsearch RAG demonstration completed!")

def demonstrate_vector_store():
    """Demonstrate basic vector store operations with Qdrant."""
    print("\n🔍 Vector Store Operations (Qdrant)")
    print("=" * 60)

    rag = RAGPipeline()

    # Health check
    print("1. Health Check")
    print("-" * 30)
    health = rag.health_check()
    print(f"Status: {health}")

    if not health:
        print("❌ Qdrant not available - skipping RAG demo")
        print("   Run: docker-compose up -d")
        return
    
    # Setup and load documents
    print("\n2. Document Loading")
    print("-" * 30)
    test_collection = "test-collection"
    qdrant_loader = QdrantVectorLoader()
    result = qdrant_loader.setup_collection(
        collection_name=test_collection,
        course_filter=None,
        delete_if_exists=True
    )
    
    print(f"✅ Test collection created:")
    print(f"   Collection: {result['collection_name']}")
    print(f"   Documents: {result['documents_loaded']}")
    print(f"   Points uploaded: {result['points_uploaded']}")

    # Interactive question answering
    print("\n3. Question Answering Demo")
    print("-" * 30)

    # Sample questions
    sample_question = "How can I join the course?"

    try:
        # Get answer from RAG pipeline
        answer = rag.ask(sample_question, search_engine="qdrant", qdrant_collection_name=test_collection)
        
        print(f"\n🤖 Answer:")
        print("-" * 30)
        print(answer)

    except Exception as e:
        logger.error(f"Error in getting answer: {e}")
        print(f"❌ Error: {e}")

    print("\n✅ Vector store operations completed!")

def main():
    """Main example function."""
    print("🚀 RAG System - Complete Demonstration")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Part 1: Full RAG Pipeline with Elasticsearch
        demonstrate_elasticsearch_rag()
        
        # Part 2: Vector Store Operations with Qdrant
        demonstrate_vector_store()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed!")
        print("\nWhat was demonstrated:")
        print("• Full RAG pipeline with question answering (Elasticsearch)")
        print("• Full RAG pipeline with question answering (Qdrant)")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
