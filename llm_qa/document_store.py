"""
Document store module for the Disney Reviews Q&A system.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import uuid
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

# Setup logger
logger = setup_logger("document_store")

# Maximum batch size for vector store operations to avoid API limits
MAX_BATCH_SIZE = 5000  # Set below the 5461 limit to leave some buffer

class DisneyReviewsDocumentStore:
    """Document store for Disney reviews using ChromaDB and LangChain."""
    
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the document store.
        
        Args:
            persist_directory (str, optional): Directory to persist the vector store.
                If None, uses {config.CACHE_DIR}/chroma
            embedding_model (str, optional): Embedding model to use.
                If None, uses OpenAI embeddings or the model specified in config.
        """
        self.persist_directory = persist_directory or os.path.join(config.CACHE_DIR, "chroma")
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize embedding model
        embedding_model = embedding_model or config.EMBEDDING_MODEL
        logger.info(f"Using embedding model: {embedding_model if embedding_model else 'OpenAI'}")
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=config.OPENAI_API_KEY
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        # Vector store
        self._vector_store = None
    
    @property
    def vector_store(self) -> Chroma:
        """
        Get the vector store, initializing it if necessary.
        
        Returns:
            Chroma: Vector store
        """
        if self._vector_store is None:
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                logger.info(f"Loading vector store from {self.persist_directory}")
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                logger.info(f"Initializing new vector store at {self.persist_directory}")
                self._vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
        
        return self._vector_store
    
    def add_reviews_from_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'Review_Text',
        metadata_columns: Optional[List[str]] = None
    ) -> None:
        """
        Add reviews from a DataFrame to the document store.
        
        The reviews are processed in batches to avoid exceeding API limits.
        Each batch contains at most MAX_BATCH_SIZE documents.
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text
            metadata_columns (List[str], optional): List of column names to include as metadata
        """
        logger.info(f"Adding {len(df)} reviews to the document store")
        
        # Default metadata columns if none provided
        if metadata_columns is None:
            metadata_columns = [
                'Review_ID', 'Rating', 'Year_Month', 'Reviewer_Location', 'Branch'
            ]
        
        # Create documents
        documents = []
        for _, row in df.iterrows():
            # Get text
            text = str(row[text_column])
            
            # Skip empty texts
            if not text.strip():
                continue
            
            # Create metadata
            metadata = {col: str(row[col]) for col in metadata_columns if col in row}
            
            # Add sentiment if available
            if 'sentiment_label' in row:
                metadata['sentiment_label'] = str(row['sentiment_label'])
            
            # Add document
            documents.append(
                Document(page_content=text, metadata=metadata)
            )
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process chunks in batches to avoid exceeding API limits
        total_chunks = len(chunks)
        batch_count = (total_chunks + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE  # Ceiling division
        
        logger.info(f"Processing {total_chunks} chunks in {batch_count} batches (max {MAX_BATCH_SIZE} per batch)")
        
        for i in range(0, total_chunks, MAX_BATCH_SIZE):
            batch = chunks[i:i+MAX_BATCH_SIZE]
            batch_num = (i // MAX_BATCH_SIZE) + 1
            logger.info(f"Processing batch {batch_num}/{batch_count} with {len(batch)} chunks")
            
            # Add batch to vector store
            self.vector_store.add_documents(batch)
            logger.info(f"Completed batch {batch_num}/{batch_count}")
        
        # Persist once after all batches are processed
        logger.info("Persisting vector store to disk")
        self.vector_store.persist()
        
        logger.info(f"Successfully added {total_chunks} chunks to the vector store")
    
    def search(
        self, 
        query: str, 
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            metadata_filter (Dict, optional): Filter for metadata fields
        
        Returns:
            List[Document]: List of similar documents
        
        Note:
            ChromaDB expects filters in the form:
            {"where": {"FieldName": {"$eq": value}, ...}}
        """
        logger.info(f"Searching for: {query}")

        filter_dict = None
        if metadata_filter:
            if len(metadata_filter) == 1:
                # Single filter: just use {field: {"$eq": value}}
                key, value = next(iter(metadata_filter.items()))
                filter_dict = {key: {"$eq": value}}
            else:
                # Multiple filters: use $and
                filter_dict = {
                    "$and": [
                        {key: {"$eq": value}} for key, value in metadata_filter.items()
                    ]
                }
            logger.info(f"Using filter: {filter_dict}")

        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter_dict
        )
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_relevant_reviews(
        self, 
        query: str, 
        k: int = 5,
        branch: Optional[str] = None,
        reviewer_location: Optional[str] = None,
        rating: Optional[int] = None,
        year_month: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> List[Document]:
        """
        Get relevant reviews based on query and filters.
        
        Args:
            query (str): Query string
            k (int): Number of results to return
            branch (str, optional): Filter by branch
            reviewer_location (str, optional): Filter by reviewer location
            rating (int, optional): Filter by rating
            year_month (str, optional): Filter by year_month
            sentiment (str, optional): Filter by sentiment
        
        Returns:
            List[Document]: List of relevant reviews
        """
        # Build metadata filter
        metadata_filter = {}
        
        if branch:
            metadata_filter['Branch'] = branch
        
        if reviewer_location:
            metadata_filter['Reviewer_Location'] = reviewer_location
        
        if rating:
            metadata_filter['Rating'] = str(rating)
        
        if year_month:
            metadata_filter['Year_Month'] = year_month
        
        if sentiment:
            metadata_filter['sentiment_label'] = sentiment
        
        # Search
        return self.search(query, k=k, metadata_filter=metadata_filter if metadata_filter else None)

def create_and_populate_document_store(
    df: pd.DataFrame,
    force_recreate: bool = False
) -> DisneyReviewsDocumentStore:
    """
    Create and populate a document store with reviews.
    
    Args:
        df (pd.DataFrame): DataFrame containing reviews
        force_recreate (bool): Whether to recreate the store even if it exists
    
    Returns:
        DisneyReviewsDocumentStore: Document store
    """
    persist_directory = os.path.join(config.CACHE_DIR, "chroma")
    
    # Check if store already exists
    store_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)
    
    if store_exists and not force_recreate:
        logger.info(f"Document store already exists at {persist_directory}")
        return DisneyReviewsDocumentStore(persist_directory=persist_directory)
    
    # Create new store
    logger.info(f"Creating new document store at {persist_directory}")
    
    # Delete existing store if it exists
    if store_exists:
        import shutil
        logger.info(f"Deleting existing store at {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Create store and add reviews
    doc_store = DisneyReviewsDocumentStore(persist_directory=persist_directory)
    doc_store.add_reviews_from_dataframe(df)
    
    return doc_store

if __name__ == "__main__":
    # Test the document store
    from data_ingestion.data_loader import load_and_preprocess
    from analysis.sentiment import analyze_reviews_sentiment
    
    # Load and preprocess the dataset
    df, _ = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Create and populate document store
    doc_store = create_and_populate_document_store(sentiment_df, force_recreate=True)
    
    # Test search
    query = "Are the rides fun for children?"
    results = doc_store.search(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"Results ({len(results)}):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Content: {result.page_content[:100]}...")
        print(f"   Metadata: {result.metadata}")
    
    # Test search with filters
    query = "How is the food?"
    branch = "Disneyland_California"
    results = doc_store.get_relevant_reviews(query, k=3, branch=branch)
    
    print(f"\nQuery: {query} (Branch: {branch})")
    print(f"Results ({len(results)}):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Content: {result.page_content[:100]}...")
        print(f"   Metadata: {result.metadata}")
    
    # Test search with multiple filters
    query = "How was your experience?"
    branch = "Disneyland_HongKong"
    reviewer_location = "Australia"
    
    print(f"\nTesting compound filter query...")
    print(f"Query: {query} (Branch: {branch}, Reviewer Location: {reviewer_location})")
    
    results = doc_store.get_relevant_reviews(
        query, 
        k=3, 
        branch=branch, 
        reviewer_location=reviewer_location
    )
    
    print(f"Results ({len(results)}):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. Content: {result.page_content[:100]}...")
        print(f"   Metadata: {result.metadata}") 