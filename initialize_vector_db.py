"""
Script to initialize the vector database for Disney Reviews Analysis.

This script loads the Disney reviews dataset, processes it, and creates
a vector database (ChromaDB) for use by the QA system. It processes the data
in batches to avoid exceeding API limits.

Usage:
    python initialize_vector_db.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (required)
    DATASET_PATH: Path to the Disney reviews CSV (optional)
    CACHE_DIR: Directory to store the vector database (optional)
"""
import os
import sys
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from data_ingestion.data_loader import load_and_preprocess
from analysis.sentiment import analyze_reviews_sentiment
from llm_qa.document_store import create_and_populate_document_store, DisneyReviewsDocumentStore
from utils.logger import setup_logger
import config

# Load environment variables
load_dotenv()

# Setup logger
logger = setup_logger("vector_db_init")

def check_environment():
    """Check if the environment is properly set up."""
    # Check for OpenAI API key
    if not config.OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY is not set. Please set it in .env file or environment variables.")
        return False
    
    # Check for dataset
    if not os.path.exists(config.DATASET_PATH):
        logger.error(f"Dataset not found at {config.DATASET_PATH}")
        return False
    
    # Check for cache directory
    if not os.path.exists(config.CACHE_DIR):
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        logger.info(f"Created cache directory at {config.CACHE_DIR}")
    
    return True

def initialize_vector_database(force_recreate=False):
    """
    Initialize the vector database for QA.
    
    Args:
        force_recreate (bool): Whether to force recreation of the database even if it exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Validate environment
    if not check_environment():
        return False
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        df, stats = load_and_preprocess()
        logger.info(f"Loaded {len(df)} reviews from {config.DATASET_PATH}")
        
        # Check if we need to analyze sentiment
        if 'sentiment_label' not in df.columns:
            logger.info("Analyzing sentiment...")
            df = analyze_reviews_sentiment(df)
            logger.info("Sentiment analysis complete")
        
        # Check if vector store exists
        chroma_dir = os.path.join(config.CACHE_DIR, "chroma")
        store_exists = os.path.exists(chroma_dir) and os.listdir(chroma_dir)
        
        if store_exists and not force_recreate:
            logger.info(f"Vector store already exists at {chroma_dir}. Use force_recreate=True to recreate.")
            
            # Quick validation
            try:
                doc_store = DisneyReviewsDocumentStore(persist_directory=chroma_dir)
                collection = doc_store.vector_store._collection
                doc_count = collection.count()
                logger.info(f"Existing vector store contains {doc_count} documents")
                
                if doc_count == 0:
                    logger.warning("Existing vector store is empty. Consider using force_recreate=True")
                return True
            except Exception as e:
                logger.error(f"Error validating existing vector store: {str(e)}")
                return False
        
        # Create and populate the vector store
        logger.info(f"Creating vector store at {chroma_dir} (force_recreate={force_recreate})")
        doc_store = create_and_populate_document_store(df, force_recreate=force_recreate)
        
        # Verify collection
        collection = doc_store.vector_store._collection
        doc_count = collection.count()
        logger.info(f"Vector store created successfully with {doc_count} documents")
        
        # Test with a simple query
        query = "How is the food at Disneyland?"
        logger.info(f"Testing vector store with query: '{query}'")
        results = doc_store.search(query, k=2)
        logger.info(f"Test query returned {len(results)} results")
        
        # Log completion time
        end_time = time.time()
        logger.info(f"Vector database initialization completed in {end_time - start_time:.2f} seconds")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Initialize vector database for Disney Reviews Analysis")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force recreation of the vector database even if it exists"
    )
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("Disney Reviews Vector Database Initialization")
    print("=" * 80)
    
    # Initialize vector database
    print(f"Starting initialization (force_recreate={args.force})...")
    result = initialize_vector_database(force_recreate=args.force)
    
    if result:
        print("\nVector database initialization completed successfully!")
        print(f"Vector database location: {os.path.join(config.CACHE_DIR, 'chroma')}")
    else:
        print("\nVector database initialization failed. Check the logs for details.")
        sys.exit(1) 