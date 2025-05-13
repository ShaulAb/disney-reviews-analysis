#!/usr/bin/env python3
"""
Test script for ChromaDB filtering capabilities.

This script tests the ChromaDB filtering logic in isolation to verify
that compound filters work correctly. It connects to the existing ChromaDB
and runs a series of test queries with different filter combinations.

Usage:
    python test_chromadb_filters.py
"""
import os
import sys
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from llm_qa.document_store import DisneyReviewsDocumentStore
import config
from utils.logger import setup_logger

# Setup logger
logger = setup_logger("chromadb_test")

def check_environment():
    """Verify the environment is correctly set up."""
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is not set in environment or .env file")
        return False
    
    chroma_dir = os.path.join(config.CACHE_DIR, "chroma")
    if not os.path.exists(chroma_dir):
        print(f"Error: ChromaDB directory not found at {chroma_dir}")
        print("Please run initialize_vector_db.py first")
        return False
    
    return True

def run_test_query(
    doc_store: DisneyReviewsDocumentStore,
    query: str,
    filters: Dict[str, Any],
    k: int = 3,
    test_name: str = "Test"
):
    """
    Run a test query with the specified filters.
    
    Args:
        doc_store: Document store to query
        query: Query string
        filters: Dictionary of filters to apply
        k: Number of results to return
        test_name: Name of the test for output
    """
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_name}")
    print(f"{'=' * 80}")
    print(f"Query: '{query}'")
    print(f"Filters: {filters}")
    print(f"k: {k}")
    
    try:
        # Build keyword arguments for get_relevant_reviews
        kwargs = {}
        if 'branch' in filters:
            kwargs['branch'] = filters['branch']
        if 'reviewer_location' in filters:
            kwargs['reviewer_location'] = filters['reviewer_location']
        if 'rating' in filters:
            kwargs['rating'] = filters['rating']
        if 'year_month' in filters:
            kwargs['year_month'] = filters['year_month']
        if 'sentiment' in filters:
            kwargs['sentiment'] = filters['sentiment']
        
        # Execute query
        results = doc_store.get_relevant_reviews(query, k=k, **kwargs)
        
        # Print results
        print(f"\nResults: {len(results)}")
        
        if len(results) > 0:
            for i, result in enumerate(results):
                print(f"\n{i+1}. Content: {result.page_content[:150]}...")
                print(f"   Metadata: {result.metadata}")
        else:
            print("\nNo results found.")
        
        return len(results) > 0
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def run_test_search(
    doc_store: DisneyReviewsDocumentStore,
    query: str,
    metadata_filter: Dict[str, Any],
    k: int = 3,
    test_name: str = "Search Test"
):
    """
    Run a test using direct search method with metadata_filter.
    
    Args:
        doc_store: Document store to query
        query: Query string
        metadata_filter: Filter dict to pass directly to search
        k: Number of results to return
        test_name: Name of the test for output
    """
    print(f"\n{'=' * 80}")
    print(f"DIRECT SEARCH TEST: {test_name}")
    print(f"{'=' * 80}")
    print(f"Query: '{query}'")
    print(f"metadata_filter: {metadata_filter}")
    print(f"k: {k}")
    
    try:
        # Execute direct search
        results = doc_store.search(query, k=k, metadata_filter=metadata_filter)
        
        # Print results
        print(f"\nResults: {len(results)}")
        
        if len(results) > 0:
            for i, result in enumerate(results):
                print(f"\n{i+1}. Content: {result.page_content[:150]}...")
                print(f"   Metadata: {result.metadata}")
        else:
            print("\nNo results found.")
        
        return len(results) > 0
    
    except Exception as e:
        print(f"\nERROR in search: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run ChromaDB filter tests."""
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print(f"{'=' * 80}")
    print(f"CHROMADB FILTER TESTS")
    print(f"{'=' * 80}")
    
    # Connect to existing ChromaDB
    try:
        chroma_dir = os.path.join(config.CACHE_DIR, "chroma")
        print(f"Connecting to ChromaDB at {chroma_dir}...")
        doc_store = DisneyReviewsDocumentStore(persist_directory=chroma_dir)
        
        # Get collection info
        collection = doc_store.vector_store._collection
        doc_count = collection.count()
        print(f"Connected successfully! Found {doc_count} documents.")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {str(e)}")
        sys.exit(1)
    
    # Run test queries
    
    # Test 1: Single filter
    run_test_query(
        doc_store,
        "What do people think of the rides?",
        {"branch": "Disneyland_California"},
        test_name="Single Filter - Branch"
    )
    
    # Test 2: Two filters
    run_test_query(
        doc_store,
        "What was your experience?",
        {"branch": "Disneyland_HongKong", "reviewer_location": "Australia"},
        test_name="Two Filters - Branch and Reviewer Location"
    )
    
    # Test 3: Three filters
    run_test_query(
        doc_store,
        "What did you enjoy?",
        {"branch": "Disneyland_HongKong", "reviewer_location": "Australia", "rating": 5},
        test_name="Three Filters - Branch, Reviewer Location, and Rating"
    )
    
    # Test 4: Filter with sentiment
    run_test_query(
        doc_store,
        "What do people say about their visit?",
        {"branch": "Disneyland_California", "sentiment": "positive"},
        test_name="Filter with Sentiment"
    )
    
    # Test 5: Non-existent value
    run_test_query(
        doc_store,
        "What do people think?",
        {"branch": "NonExistentBranch"},
        test_name="Non-existent Branch (should return no results)"
    )
    
    # Test 6: Direct search with single filter
    run_test_search(
        doc_store,
        "How is the food?",
        {"Branch": "Disneyland_California"},
        test_name="Direct Search - Single Filter"
    )
    
    # Test 7: Direct search with multiple filters
    run_test_search(
        doc_store,
        "What do visitors say?",
        {"Branch": "Disneyland_HongKong", "Reviewer_Location": "Australia"},
        test_name="Direct Search - Multiple Filters"
    )
    
    print(f"\n{'=' * 80}")
    print("Tests completed!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 