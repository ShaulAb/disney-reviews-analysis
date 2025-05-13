"""
Data loading and processing module for Disney Reviews dataset.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

# Setup logger
logger = setup_logger("data_loader")

def load_disney_reviews(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Disney reviews dataset from a CSV file.
    
    Args:
        file_path (str, optional): Path to the CSV file. Defaults to config.DATASET_PATH.
    
    Returns:
        pd.DataFrame: DataFrame containing the reviews data
    """
    file_path = file_path or config.DATASET_PATH
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding="ISO-8859-1")
        logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the reviews data.
    
    Args:
        df (pd.DataFrame): Raw reviews DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame
    """
    logger.info("Preprocessing reviews data")
    
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    processed_df.loc[df["Year_Month"] == "missing", "Year_Month"] = '2009-10'
    processed_df = processed_df.fillna({
        'Review_Text': '',
        'Reviewer_Location': 'Unknown',
        'Branch': 'Unknown',
    })
    
    # Extract year and month from Year_Month column and add as separate columns
    try:
        processed_df[['Year', 'Month']] = processed_df['Year_Month'].str.split('-', expand=True)
        processed_df['Year'] = processed_df['Year'].astype(int)
        processed_df['Month'] = processed_df['Month'].astype(int)
    except Exception as e:
        logger.warning(f"Error processing Year_Month column: {str(e)}")
    
    # Convert Review_ID to string if it's not already
    processed_df['Review_ID'] = processed_df['Review_ID'].astype(str)
    
    # Ensure Rating is numeric
    processed_df['Rating'] = pd.to_numeric(processed_df['Rating'], errors='coerce')
    
    logger.info("Preprocessing complete")
    return processed_df

def add_review_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add column with review text length.
    
    Args:
        df (pd.DataFrame): Reviews DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with added review length column
    """
    df['Review_Length'] = df['Review_Text'].str.len()
    return df

def get_review_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for the reviews dataset.
    
    Args:
        df (pd.DataFrame): Processed reviews DataFrame
    
    Returns:
        Dict: Dictionary containing summary statistics
    """
    stats = {
        'total_reviews': len(df),
        'average_rating': df['Rating'].mean(),
        'rating_distribution': df['Rating'].value_counts().to_dict(),
        'reviews_per_branch': df['Branch'].value_counts().to_dict(),
        'top_reviewer_locations': df['Reviewer_Location'].value_counts().head(10).to_dict(),
        'time_range': (df['Year_Month'].min(), df['Year_Month'].max())
    }
    
    return stats

def load_and_preprocess() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and preprocess the dataset and return with summary statistics.
    
    Returns:
        Tuple: (processed_df, stats_dict)
    """
    raw_df = load_disney_reviews()
    processed_df = preprocess_reviews(raw_df)
    processed_df = add_review_length(processed_df)
    stats = get_review_summary_stats(processed_df)
    
    return processed_df, stats

if __name__ == "__main__":
    # Test the data loading and preprocessing
    df, stats = load_and_preprocess()
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\nDataset Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head()) 