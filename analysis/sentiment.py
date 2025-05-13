"""
Sentiment analysis module for Disney Reviews.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from textblob import TextBlob
import re

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

# Setup logger
logger = setup_logger("sentiment_analysis")

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for sentiment analysis.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_sentiment_textblob(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text (str): Input text
    
    Returns:
        Dict: Dictionary with polarity and subjectivity scores
    """
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        return {"polarity": 0.0, "subjectivity": 0.0}
    
    analysis = TextBlob(cleaned_text)
    return {
        "polarity": analysis.sentiment.polarity,
        "subjectivity": analysis.sentiment.subjectivity
    }

def get_sentiment_label(polarity: float) -> str:
    """
    Convert polarity score to sentiment label.
    
    Args:
        polarity (float): Sentiment polarity score
    
    Returns:
        str: Sentiment label (negative, neutral, positive)
    """
    if polarity <= -0.1:
        return "negative"
    elif polarity >= 0.1:
        return "positive"
    else:
        return "neutral"

def analyze_reviews_sentiment(df: pd.DataFrame, text_column: str = 'Review_Text') -> pd.DataFrame:
    """
    Analyze sentiment for all reviews in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing reviews
        text_column (str): Name of the column containing review text
    
    Returns:
        pd.DataFrame: DataFrame with added sentiment columns
    """
    logger.info("Analyzing sentiment for reviews")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Apply sentiment analysis to each review
    sentiments = [analyze_sentiment_textblob(text) for text in result_df[text_column]]
    
    # Add sentiment columns
    result_df['sentiment_polarity'] = [s['polarity'] for s in sentiments]
    result_df['sentiment_subjectivity'] = [s['subjectivity'] for s in sentiments]
    result_df['sentiment_label'] = [get_sentiment_label(p) for p in result_df['sentiment_polarity']]
    
    logger.info("Sentiment analysis complete")
    
    return result_df

def get_sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for sentiment analysis.
    
    Args:
        df (pd.DataFrame): DataFrame with sentiment columns
    
    Returns:
        Dict: Dictionary with sentiment summary statistics
    """
    if 'sentiment_label' not in df.columns:
        logger.error("Sentiment columns not found in DataFrame")
        return {}
    
    summary = {
        'sentiment_distribution': df['sentiment_label'].value_counts().to_dict(),
        'average_polarity': df['sentiment_polarity'].mean(),
        'average_subjectivity': df['sentiment_subjectivity'].mean(),
        'sentiment_by_branch': df.groupby('Branch')['sentiment_polarity'].mean().to_dict(),
        'sentiment_by_rating': df.groupby('Rating')['sentiment_polarity'].mean().to_dict()
    }
    
    return summary

if __name__ == "__main__":
    # Test the sentiment analysis
    from data_ingestion.data_loader import load_and_preprocess
    
    # Load and preprocess the dataset
    df, _ = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Get sentiment summary
    summary = get_sentiment_summary(sentiment_df)
    
    print("\nSentiment Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nSample data with sentiment:")
    print(sentiment_df[['Review_Text', 'sentiment_polarity', 'sentiment_label']].head()) 