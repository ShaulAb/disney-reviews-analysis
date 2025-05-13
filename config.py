"""
Configuration settings for the Disney Reviews Analysis project.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Data paths
DATASET_PATH = os.getenv("DATASET_PATH", "data/DisneylandReviews.csv")
CACHE_DIR = os.getenv("CACHE_DIR", ".cache")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# LLM settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "")  # Empty means use OpenAI's embeddings

# Analysis settings
SENTIMENT_ANALYZER = "textblob"  # Options: 'textblob', 'vader', etc.
TOP_N_INSIGHTS = 10  # Number of top insights to display

# UI settings
APP_TITLE = "Disney Parks Review Analysis"
APP_SUBTITLE = "Extract insights and ask questions about Disney park reviews"
THEME_COLOR = "#1E90FF"  # Dodger blue 