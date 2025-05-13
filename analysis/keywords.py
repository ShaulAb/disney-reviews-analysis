"""
Keywords extraction module for Disney Reviews.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

# Setup logger
logger = setup_logger("keywords_extraction")

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading required NLTK resources")
    nltk.download('punkt')
    nltk.download('stopwords')

# Get English stopwords
STOPWORDS = set(stopwords.words('english'))
# Add common words that aren't useful for analysis
CUSTOM_STOPWORDS = {
    'disney', 'disneyland', 'park', 'parks', 'world', 'day', 'days', 'time',
    'trip', 'visit', 'visited', 'visiting', 'place', 'people', 'lot', 'lots',
    'really', 'go', 'went', 'going', 'get', 'got', 'getting', 'make', 'made',
    'making', 'us', 'use', 'used', 'using', 'one', 'two', 'three', 'see', 'saw',
    'seeing', 'take', 'took', 'taking', 'come', 'came', 'coming', 'year', 'years',
    'month', 'months', 'week', 'weeks'
}
STOPWORDS.update(CUSTOM_STOPWORDS)

def preprocess_text_for_keywords(text: str) -> str:
    """
    Preprocess text for keyword extraction.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters except apostrophes for contractions
    text = re.sub(r'[^\w\s\']', ' ', text)
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract keywords from text.
    
    Args:
        text (str): Input text
        top_n (int): Number of top keywords to return
    
    Returns:
        List[str]: List of keywords
    """
    if not text:
        return []
    
    # Preprocess text
    processed_text = preprocess_text_for_keywords(text)
    
    # Tokenize text
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords and very short words
    filtered_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    
    # Count token frequencies
    token_counts = Counter(filtered_tokens)
    
    # Get top keywords
    top_keywords = [keyword for keyword, _ in token_counts.most_common(top_n)]
    
    return top_keywords

def extract_ngrams(text: str, n: int = 2, top_n: int = 10) -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text (str): Input text
        n (int): n-gram size
        top_n (int): Number of top n-grams to return
    
    Returns:
        List[str]: List of n-grams
    """
    if not text:
        return []
    
    # Preprocess text
    processed_text = preprocess_text_for_keywords(text)
    
    # Tokenize text
    tokens = word_tokenize(processed_text)
    
    # Remove stopwords and very short words
    filtered_tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 2]
    
    # Generate n-grams
    n_grams = list(ngrams(filtered_tokens, n))
    
    # Convert n-grams to strings
    n_gram_strings = [' '.join(gram) for gram in n_grams]
    
    # Count n-gram frequencies
    n_gram_counts = Counter(n_gram_strings)
    
    # Get top n-grams
    top_n_grams = [n_gram for n_gram, _ in n_gram_counts.most_common(top_n)]
    
    return top_n_grams

def extract_all_review_keywords(df: pd.DataFrame, text_column: str = 'Review_Text') -> pd.DataFrame:
    """
    Extract keywords from all reviews in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing reviews
        text_column (str): Name of the column containing review text
    
    Returns:
        pd.DataFrame: DataFrame with added keywords column
    """
    logger.info("Extracting keywords from reviews")
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Extract keywords from each review
    result_df['keywords'] = result_df[text_column].apply(lambda x: extract_keywords(x, top_n=5))
    
    # Extract bigrams from each review
    result_df['bigrams'] = result_df[text_column].apply(lambda x: extract_ngrams(x, n=2, top_n=5))
    
    logger.info("Keyword extraction complete")
    
    return result_df

def get_top_keywords_overall(df: pd.DataFrame, top_n: int = 20) -> Dict[str, int]:
    """
    Get the most common keywords across all reviews.
    
    Args:
        df (pd.DataFrame): DataFrame with extracted keywords
        top_n (int): Number of top keywords to return
    
    Returns:
        Dict[str, int]: Dictionary with keyword and count
    """
    if 'keywords' not in df.columns:
        logger.error("Keywords column not found in DataFrame")
        return {}
    
    # Flatten the list of keywords for all reviews
    all_keywords = [keyword for keywords in df['keywords'] for keyword in keywords]
    
    # Count keyword frequencies
    keyword_counts = Counter(all_keywords)
    
    # Get top keywords
    top_keywords = dict(keyword_counts.most_common(top_n))
    
    return top_keywords

def get_top_bigrams_overall(df: pd.DataFrame, top_n: int = 20) -> Dict[str, int]:
    """
    Get the most common bigrams across all reviews.
    
    Args:
        df (pd.DataFrame): DataFrame with extracted bigrams
        top_n (int): Number of top bigrams to return
    
    Returns:
        Dict[str, int]: Dictionary with bigram and count
    """
    if 'bigrams' not in df.columns:
        logger.error("Bigrams column not found in DataFrame")
        return {}
    
    # Flatten the list of bigrams for all reviews
    all_bigrams = [bigram for bigrams in df['bigrams'] for bigram in bigrams]
    
    # Count bigram frequencies
    bigram_counts = Counter(all_bigrams)
    
    # Get top bigrams
    top_bigrams = dict(bigram_counts.most_common(top_n))
    
    return top_bigrams

def get_keywords_by_sentiment(df: pd.DataFrame, top_n: int = 10) -> Dict[str, Dict[str, int]]:
    """
    Get top keywords for each sentiment category.
    
    Args:
        df (pd.DataFrame): DataFrame with keywords and sentiment columns
        top_n (int): Number of top keywords per sentiment to return
    
    Returns:
        Dict: Dictionary with sentiment category and top keywords
    """
    if 'keywords' not in df.columns or 'sentiment_label' not in df.columns:
        logger.error("Required columns not found in DataFrame")
        return {}
    
    # Get unique sentiment labels
    sentiment_labels = df['sentiment_label'].unique()
    
    # Get top keywords for each sentiment
    keywords_by_sentiment = {}
    for sentiment in sentiment_labels:
        # Get reviews with this sentiment
        sentiment_reviews = df[df['sentiment_label'] == sentiment]
        
        # Flatten the list of keywords for these reviews
        all_keywords = [keyword for keywords in sentiment_reviews['keywords'] for keyword in keywords]
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Get top keywords for this sentiment
        top_keywords = dict(keyword_counts.most_common(top_n))
        
        keywords_by_sentiment[sentiment] = top_keywords
    
    return keywords_by_sentiment

if __name__ == "__main__":
    # Test the keyword extraction
    from data_ingestion.data_loader import load_and_preprocess
    from analysis.sentiment import analyze_reviews_sentiment
    
    # Load and preprocess the dataset
    df, _ = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Extract keywords
    keywords_df = extract_all_review_keywords(sentiment_df)
    
    # Get top keywords overall
    top_keywords = get_top_keywords_overall(keywords_df)
    print("\nTop Keywords Overall:")
    for keyword, count in top_keywords.items():
        print(f"{keyword}: {count}")
    
    # Get top bigrams overall
    top_bigrams = get_top_bigrams_overall(keywords_df)
    print("\nTop Bigrams Overall:")
    for bigram, count in top_bigrams.items():
        print(f"{bigram}: {count}")
    
    # Get keywords by sentiment
    keywords_by_sentiment = get_keywords_by_sentiment(keywords_df)
    print("\nTop Keywords by Sentiment:")
    for sentiment, keywords in keywords_by_sentiment.items():
        print(f"\n{sentiment.upper()}:")
        for keyword, count in keywords.items():
            print(f"{keyword}: {count}")
    
    print("\nSample data with keywords:")
    print(keywords_df[['Review_Text', 'keywords', 'bigrams']].head()) 