"""
Main script to run the Disney Reviews Analysis application.
"""
import os
import sys
import argparse
import subprocess
from utils.logger import setup_logger

# Setup logger for the runner
logger = setup_logger("runner")

def run_streamlit_app():
    """Run the Streamlit application."""
    logger.info("Starting Streamlit application")
    
    # Get the directory of the run.py file
    file_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the absolute path to the Streamlit app
    streamlit_app_path = os.path.join(file_dir, "ui", "streamlit_app.py")
    
    # Run Streamlit app using subprocess
    logger.info(f"Running Streamlit app from: {streamlit_app_path}")
    subprocess.run([
        "streamlit", "run", streamlit_app_path,
        "--browser.serverAddress=0.0.0.0",
        "--server.port=8501"
    ])
    
def initialize_vector_db():
    """Initialize the vector database."""
    logger.info("Starting vector database initialization")
    
    # Import here to avoid circular imports
    from data_ingestion.data_loader import load_and_preprocess
    from analysis.sentiment import analyze_reviews_sentiment
    from llm_qa.document_store import create_and_populate_document_store
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        df, stats = load_and_preprocess()
        logger.info(f"Loaded {len(df)} reviews")
        
        # Analyze sentiment
        logger.info("Analyzing sentiment")
        sentiment_df = analyze_reviews_sentiment(df)
        
        # Create and populate document store (force recreate)
        logger.info("Creating and populating vector database")
        doc_store = create_and_populate_document_store(sentiment_df, force_recreate=True)
        
        # Log success
        logger.info("Vector database initialization complete")
        print("Vector database initialization complete!")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        print(f"Error initializing vector database: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Disney Reviews Analysis")
    parser.add_argument(
        "--init-db", 
        action="store_true", 
        help="Initialize the vector database"
    )
    args = parser.parse_args()
    
    # Initialize vector database if requested
    if args.init_db:
        initialize_vector_db()
    else:
        # Run the Streamlit app
        run_streamlit_app() 