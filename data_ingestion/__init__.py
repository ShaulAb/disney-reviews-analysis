"""
Data ingestion package for Disney Reviews Analysis project.
"""
from data_ingestion.data_loader import (
    load_disney_reviews, 
    preprocess_reviews, 
    add_review_length, 
    get_review_summary_stats,
    load_and_preprocess
)

__all__ = [
    "load_disney_reviews", 
    "preprocess_reviews", 
    "add_review_length", 
    "get_review_summary_stats",
    "load_and_preprocess"
] 