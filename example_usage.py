#!/usr/bin/env python3
"""
Example usage of the RAG system.

This script demonstrates:
1. Full RAG pipeline with Elasticsearch (including question answering)
2. Basic vector store operations with Qdrant
"""

import logging
import os
from dotenv import load_dotenv

from rag import RAGPipeline
from rag.data.vector_store import VectorStoreLoader, QdrantVectorLoader
from rag.search.qdrant_client_custom import QdrantClientCustom

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demonstrate_full_rag():
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
    sample_questions = [
        "What is the difference between supervised and unsupervised learning?",
        "How do I prepare data for machine learning?",
        "What are the steps in a typical ML workflow?",
        "How do I evaluate a machine learning model?",
        "What is feature engineering?"
    ]
    
    print("Sample questions available:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    
    print("\nYou can:")
    print("• Enter a number (1-5) to ask a sample question")
    print("• Type your own question")
    print("• Type 'quit' to exit")
    
    while True:
        print("\n" + "─" * 50)
        user_input = input("\n🤔 Your question (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Handle numbered questions
        if user_input.isdigit():
            try:
                question_num = int(user_input)
                if 1 <= question_num <= len(sample_questions):
                    question = sample_questions[question_num - 1]
                else:
                    print("❌ Please enter a number between 1 and 5")
                    continue
            except ValueError:
                print("❌ Invalid input")
                continue
        else:
            question = user_input
        
        if not question:
            print("❌ Please enter a question")
            continue
        
        print(f"\n🤔 Question: {question}")
        print("🔍 Searching for relevant documents...")
        
        try:
            # Get answer from RAG pipeline
            answer = rag.ask(question)
            
            print(f"\n🤖 Answer:")
            print("-" * 30)
            print(answer)
            
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            print(f"❌ Error: {e}")
    
    print("\n✅ RAG demonstration completed!")


def demonstrate_vector_store():
    """Demonstrate basic vector store operations with Qdrant."""
    print("\n🔍 Vector Store Operations (Qdrant)")
    print("=" * 50)
    
    # Check Qdrant availability
    try:
        qdrant_client = QdrantClientCustom()
        print(f"✅ Qdrant connected: {qdrant_client.qdrant_url}")
    except Exception as e:
        print(f"❌ Qdrant not available: {e}")
        print("   Run: docker-compose up -d")
        return
    
    # Create test collection name
    test_collection = "test-collection"
    
    try:
        # Use QdrantVectorLoader for operations
        qdrant_loader = QdrantVectorLoader()
        
        print("\n1. Creating Test Collection")
        print("-" * 30)
        
        # Create test collection with ML course documents only (smaller dataset)
        result = qdrant_loader.setup_collection(
            collection_name=test_collection,
            course_filter="machine-learning-zoomcamp",
            delete_if_exists=True
        )
        
        print(f"✅ Test collection created:")
        print(f"   Collection: {result['collection_name']}")
        print(f"   Documents: {result['documents_loaded']}")
        print(f"   Points uploaded: {result['points_uploaded']}")
        
        print("\n2. Collection Statistics")
        print("-" * 30)
        
        collections_info = qdrant_client.qdrant.get_collections()
        collections = [collection.name for collection in collections_info.collections]
        print(f"📊 Available collections: {len(collections)}")
        
        for collection in collections:
            info = qdrant_client.qdrant.get_collection(collection)
            print(f"   • {collection}: {info.points_count} points")
        
        print("\n3. Cleaning Up")
        print("-" * 30)
        
        # Delete test collection
        qdrant_client.delete_collection(test_collection)
        print(f"✅ Test collection '{test_collection}' deleted")
        
        print("\n✅ Vector store operations completed!")
        
    except Exception as e:
        logger.error(f"Error in vector store operations: {e}")
        print(f"❌ Error: {e}")


def main():
    """Main example function."""
    print("🚀 RAG System - Complete Demonstration")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    
    try:
        # Part 1: Full RAG Pipeline with Elasticsearch
        demonstrate_full_rag()
        
        # Part 2: Vector Store Operations with Qdrant
        demonstrate_vector_store()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed!")
        print("\nWhat was demonstrated:")
        print("• Full RAG pipeline with question answering (Elasticsearch)")
        print("• Vector store operations: create, load, delete (Qdrant)")
        print("\nNext steps:")
        print("• Integrate Qdrant with RAG pipeline for similarity search")
        print("• Test with different embedding models")
        print("• Build production-ready applications")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
