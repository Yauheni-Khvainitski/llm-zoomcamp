#!/usr/bin/env python3
"""
Example usage of the RAG system.

This script demonstrates how to use the structured RAG system
that was extracted from the rag.ipynb notebook.
"""

import logging
import os

from dotenv import load_dotenv

from rag import Course, RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("🚀 RAG System Example Usage")
    print("=" * 50)

    # Load environment variables
    load_dotenv()

    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Create a .env file with your API key to test LLM functionality")

    try:
        # Initialize the RAG pipeline
        print("\n1. Initializing RAG Pipeline...")
        rag = RAGPipeline()

        # Perform health check
        print("\n2. Performing Health Check...")
        health = rag.health_check()
        print(f"Health Status: {health}")

        if not health["elasticsearch"]:
            print("❌ Elasticsearch is not available. Please start Elasticsearch first.")
            print("   Run: docker-compose up -d")
            return

        # Set up the index and load documents
        print("\n3. Setting up Index and Loading Documents...")
        setup_result = rag.setup_index()
        print("✅ Setup complete:")
        print(f"   - Documents loaded: {setup_result['documents_loaded']}")
        print(f"   - Documents indexed: {setup_result['documents_indexed']}")

        # Get system statistics
        print("\n4. System Statistics...")
        stats = rag.get_stats()
        print(f"📊 Stats: {stats}")

        # Example 1: Basic question without course filter
        print("\n5. Example 1: Basic Question (All Courses)")
        print("-" * 30)
        question1 = "How do I copy files to a Docker container?"
        print(f"Question: {question1}")

        try:
            response1 = rag.ask(question1)
            print(f"Response: {response1}")
        except Exception as e:
            print(f"Error: {e}")

        # Example 2: Question with course filter
        print("\n6. Example 2: Question with Course Filter")
        print("-" * 30)
        question2 = "How do I debug a Docker container?"
        course_filter = Course.MACHINE_LEARNING_ZOOMCAMP
        print(f"Question: {question2}")
        print(f"Course Filter: {course_filter.value}")

        try:
            response2 = rag.ask(question2, course_filter=course_filter)
            print(f"Response: {response2}")
        except Exception as e:
            print(f"Error: {e}")

        # Example 3: Detailed response with metadata
        print("\n7. Example 3: Detailed Response with Metadata")
        print("-" * 30)
        question3 = "What is the difference between Docker and Kubernetes?"

        try:
            result = rag.ask_with_details(question=question3, course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=3)

            print(f"Question: {result['question']}")
            print(f"Response: {result['response']}")
            print(f"Total Hits: {result['search_results']['total_hits']}")
            print(f"Max Score: {result['search_results']['max_score']}")
            print(f"Documents Used: {len(result['search_results']['documents'])}")
            print(f"Model Used: {result['metadata']['model']}")

        except Exception as e:
            print(f"Error: {e}")

        # Example 4: Using individual components
        print("\n8. Example 4: Using Individual Components")
        print("-" * 30)

        # Search only (without LLM)
        documents = rag.search(question="How to set up Docker?", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=2)

        print(f"Search Results: {len(documents)} documents found")
        for i, doc in enumerate(documents, 1):
            print(f"  {i}. {doc['question'][:60]}...")

        # Example 5: Test homework questions from the notebook
        print("\n9. Example 5: Homework Questions")
        print("-" * 30)

        # Q3: Searching
        homework_q3 = "How do execute a command on a Kubernetes pod?"
        search_raw = rag.search(homework_q3, return_raw=True)
        max_score = search_raw["hits"]["max_score"]
        print(f"Q3 - Max Score: {max_score}")

        # Q4: Filtering
        homework_q4 = "How do copy a file to a Docker container?"
        filtered_docs = rag.search(homework_q4, course_filter=Course.MACHINE_LEARNING_ZOOMCAMP, num_results=3)
        if len(filtered_docs) >= 3:
            third_doc_text = filtered_docs[2]["text"][:100] + "..."
            print(f"Q4 - Third document text: {third_doc_text}")

        print("\n✅ Example usage completed successfully!")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")


def test_components():
    """Test individual components."""
    print("\n🧪 Testing Individual Components")
    print("=" * 50)

    # Test QueryBuilder
    from rag.models import Course
    from rag.search import QueryBuilder

    print("\n1. Testing QueryBuilder...")
    query_builder = QueryBuilder()

    # Test with course filter
    query = query_builder.build_search_query(
        question="Test question", course_filter=Course.DATA_ENGINEERING_ZOOMCAMP, num_results=5, boost=6
    )
    print(f"✅ Query with filter: {query['size']} results, course filter present")

    # Test without course filter
    query_no_filter = query_builder.build_search_query(question="Test question", num_results=10)
    print(f"✅ Query without filter: {query_no_filter['size']} results, no filter")

    # Test ContextFormatter
    from rag.formatting import ContextFormatter

    print("\n2. Testing ContextFormatter...")
    formatter = ContextFormatter()

    sample_docs = [
        {"question": "What is Docker?", "text": "Docker is a containerization platform."},
        {"question": "What is Kubernetes?", "text": "Kubernetes is a container orchestration system."},
    ]

    context = formatter.format_context(sample_docs)
    print(f"✅ Formatted context ({len(context)} chars):")
    print(context[:100] + "..." if len(context) > 100 else context)

    prompt = formatter.build_prompt("What is containerization?", context)
    print(f"✅ Built prompt ({len(prompt)} chars)")

    print("\n✅ Component testing completed!")


if __name__ == "__main__":
    print("RAG System - Structured Implementation")
    print("Based on rag.ipynb notebook")
    print("=" * 60)

    # Run main examples
    main()

    # Test individual components
    test_components()

    print("\n" + "=" * 60)
    print("For more examples, see the README.md file")
    print("To run tests: python -m rag.tests.test_query_builder")
