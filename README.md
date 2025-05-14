# Disney Parks Review Analysis

A data analysis and Q&A system for Disney park reviews. This application processes and analyzes Disney park reviews, extracting insights and providing a chat interface to answer questions about the reviews.

## Features

- **Data Loading and Preprocessing**: Load and preprocess Disney park reviews from CSV files.
- **Sentiment Analysis**: Analyze the sentiment of reviews using TextBlob.
- **Keyword Extraction**: Extract important keywords from reviews.
- **Insight Generation**: Generate insights and recommendations from the reviews.
- **Question Answering**: Ask questions about the reviews and get answers based on the reviews.
- **Interactive Dashboard**: View visualizations and insights in a Streamlit dashboard.

## Project Structure

The project is organized into several modules:

- `data_ingestion/`: Data loading and preprocessing.
- `analysis/`: Sentiment analysis and keyword extraction.
- `insights/`: Insight generation from reviews.
- `llm_qa/`: Question answering system based on reviews.
- `ui/`: Streamlit user interface.
- `utils/`: Utility functions.

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:ShaulAb/disney-reviews-analysis.git
   cd disney-reviews-analysis
   ```

2. Install the required packages:
   ```
   uv pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your-openai-api-key
   DATASET_PATH=path/to/DisneylandReviews.csv
   ```

## Vector Database Initialization

Before using the Q&A feature, you need to initialize the vector database:

1. Run the initialization script:
   ```
   python initialize_vector_db.py
   ```

2. To force recreation of an existing database:
   ```
   python initialize_vector_db.py --force
   ```

This process:
- Loads and preprocesses the Disney reviews dataset
- Performs sentiment analysis
- Creates embeddings for each review (processed in batches)
- Stores the embeddings in a ChromaDB database

**Note**: This initialization is resource-intensive and requires an OpenAI API key for generating embeddings. The system uses batch processing to handle large datasets while respecting API rate limits.

## Usage

To run the application:

```
python run.py
```

The Streamlit UI will be available at `http://localhost:8501`.

The application has three main pages:

1. **Dashboard**: View visualizations of the review data.
2. **Insights & Recommendations**: View insights and recommendations generated from the reviews.
3. **Chat Q&A**: Ask questions about the reviews.

## Chat Q&A System

The Chat Q&A system allows you to ask natural language questions about the Disney park reviews. For example:

- "What do people like most about Disneyland?"
- "Are the rides appropriate for small children?"
- "What are the most common complaints about food?"

The system searches through the reviews to find relevant information and generates answers based on those reviews.

## Requirements

- Python 3.12
- OpenAI API key
- Pandas
- Streamlit
- LangChain
- TextBlob
- NLTK
- ChromaDB

## License

[MIT License](LICENSE)
