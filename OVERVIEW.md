[MODE: EXECUTE]

# Disney Parks Review Analysis – Comprehensive Documentation

---

## Table of Contents

- [Disney Parks Review Analysis – Comprehensive Documentation](#disney-parks-review-analysis--comprehensive-documentation)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Data Ingestion](#data-ingestion)
  - [Data Preprocessing](#data-preprocessing)
    - [Sentiment Analysis](#sentiment-analysis)
    - [Keyword Extraction](#keyword-extraction)
    - [Other NLP Transformations](#other-nlp-transformations)
  - [Vector Database (ChromaDB)](#vector-database-chromadb)
    - [Why Use a Vector DB?](#why-use-a-vector-db)
    - [How ChromaDB is Used](#how-chromadb-is-used)
    - [Batching and Filtering](#batching-and-filtering)
  - [Insights \& Recommendations](#insights--recommendations)
    - [Insight Extraction Process](#insight-extraction-process)
    - [Categories and Ranking](#categories-and-ranking)
  - [Chatbot Capabilities](#chatbot-capabilities)
    - [Implementation](#implementation)
    - [Supported Models](#supported-models)
    - [Filtering and Q\&A Logic](#filtering-and-qa-logic)
  - [User Interface](#user-interface)
    - [Streamlit Dashboard](#streamlit-dashboard)
    - [Chat Interface](#chat-interface)
    - [Export and Controls](#export-and-controls)
  - [Project Structure](#project-structure)
  - [How to Run](#how-to-run)
  - [Technical Notes \& References](#technical-notes--references)
  - [Acknowledgements](#acknowledgements)

---

## Project Overview

This project analyzes Disney park reviews to extract actionable insights and provide a powerful, user-friendly Q&A system. It leverages modern NLP, LLMs, and vector search to allow users to explore, visualize, and query thousands of real customer reviews.

---

## Data Ingestion

- **Source:** The primary dataset is a CSV file (`DisneylandReviews.csv`) containing fields such as `Review_ID`, `Rating`, `Year_Month`, `Reviewer_Location`, `Review_Text`, and `Branch`.
- **Loader:** The `data_ingestion/data_loader.py` module handles reading the CSV, parsing dates, and basic cleaning.
- **Preprocessing:** Initial steps include removing empty reviews, handling missing values, and standardizing text fields.

---

## Data Preprocessing

### Sentiment Analysis

- **Purpose:** To enrich each review with a sentiment label (`positive`, `neutral`, `negative`).
- **Implementation:** The `analysis/sentiment.py` module uses TextBlob for sentiment scoring. Each review is assigned a label based on polarity thresholds.
- **Usage:** Sentiment labels are used for filtering, analytics, and insight generation.

### Keyword Extraction

- **Purpose:** To identify key themes and topics in the reviews.
- **Implementation:** The `analysis/keywords.py` module uses NLP techniques (e.g., TF-IDF, RAKE, or custom heuristics) to extract keywords from each review.
- **Usage:** Extracted keywords are used for insight generation and to support advanced search/filtering.

### Other NLP Transformations

- **Text Normalization:** Lowercasing, punctuation removal, and tokenization are applied as needed.
- **Chunking:** Reviews are split into manageable chunks for embedding and vector storage.

---

## Vector Database (ChromaDB)

### Why Use a Vector DB?

- **Semantic Search:** Traditional keyword search is limited. Vector DBs enable semantic similarity search, allowing the system to find reviews relevant to a user's query even if the wording is different.
- **Scalability:** Efficiently handles tens of thousands of reviews.
- **Filtering:** Supports metadata-based filtering (e.g., by branch, sentiment, reviewer location).

### How ChromaDB is Used

- **Embeddings:** Each review (or chunk) is embedded using OpenAI's `text-embedding-ada-002` model.
- **Storage:** Embeddings and metadata are stored in ChromaDB, a high-performance vector database.
- **Batching:** To respect API limits, embeddings are generated in batches (max 5,000 per batch).
- **Persistence:** The vector DB is persisted to disk for fast reloads and sharing.

### Batching and Filtering

- **Batching:** Implemented in `llm_qa/document_store.py` to avoid API errors and improve reliability.
- **Filtering:** Supports single and multi-field filters using ChromaDB's `$and` operator (see [ChromaDB Filtering Docs](https://docs.trychroma.com/usage-guide#filtering)).  
  Example:
  ```python
  filter = {
      "$and": [
          {"Branch": {"$eq": "Disneyland_HongKong"}},
          {"Reviewer_Location": {"$eq": "Australia"}}
      ]
  }
  ```

---

## Insights & Recommendations

### Insight Extraction Process

- **Keyword Aggregation:** Extracted keywords are aggregated and analyzed for frequency and co-occurrence.
- **Sentiment Aggregation:** Sentiment scores are aggregated by branch, time, and topic.
- **Insight Generation:** The `insights/insight_generator.py` module combines keyword and sentiment data to generate actionable insights and recommendations.

### Categories and Ranking

- **Categories:** Insights are grouped into categories such as:
    - Staff and Service
    - Rides and Attractions
    - Food and Dining
    - Cleanliness and Maintenance
    - Wait Times and Crowds
    - Value for Money
    - Overall Experience
    - Logistics and Operations
    - Facilities and Amenities
- **Ranking:** Insights are ranked by frequency, sentiment impact, and relevance to user queries.

---

## Chatbot Capabilities

### Implementation

- **LLM Integration:** Uses OpenAI's GPT models (configurable, e.g., `gpt-4o-mini`) via the LangChain framework.
- **Q&A Chain:** The `llm_qa/qa_chain.py` module implements a retrieval-augmented generation (RAG) pipeline:
    1. User query is embedded.
    2. ChromaDB is searched for semantically similar reviews, with optional metadata filters.
    3. The most relevant reviews are passed to the LLM to generate a natural language answer.

### Supported Models

- **Embeddings:** `text-embedding-ada-002` (OpenAI)
- **LLM:** Configurable via `config.py` (default: OpenAI GPT-4o-mini)

### Filtering and Q&A Logic

- **Metadata Filtering:** Users can filter by branch, reviewer location, rating, sentiment, and more.
- **Multi-Field Filtering:** Implemented using ChromaDB's `$and` operator for robust multi-condition queries.
- **Post-Processing:** The LLM is prompted to cite supporting reviews and provide references.

---

## User Interface

### Streamlit Dashboard

- **Overview:** Displays total reviews, average rating, and number of branches.
- **Visualizations:** Uses Plotly Express for interactive charts (rating distribution, sentiment, reviews over time, top locations).
- **Branch and Sentiment Analysis:** Pie charts and bar charts for quick insights.

### Chat Interface

- **Chat Q&A:** Users can ask natural language questions about the reviews.
- **References:** Answers include supporting review excerpts and metadata.
- **Controls:** Includes clear chat, export chat, and timeout settings.

### Export and Controls

- **Export:** Chat history can be exported as a `.txt` file.
- **Clear Chat:** Resets the chat session (now uses `st.rerun()` for compatibility).

---

## Project Structure

```
.
├── analysis/
│   ├── keywords.py
│   └── sentiment.py
├── data_ingestion/
│   └── data_loader.py
├── insights/
│   └── insight_generator.py
├── llm_qa/
│   ├── document_store.py
│   └── qa_chain.py
├── ui/
│   └── streamlit_app.py
├── utils/
│   └── logger.py
├── initialize_vector_db.py
├── test_chromadb_filters.py
├── config.py
├── requirements.txt
├── README.md
└── ...
```

---

## How to Run

1. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   - Create a `.env` file with:
     ```
     OPENAI_API_KEY=your-openai-key
     DATASET_PATH=data/DisneylandReviews.csv
     ```

3. **Initialize the vector database:**
   ```bash
   uv run initialize_vector_db.py
   ```

4. **Run the Streamlit app:**
   ```bash
   uv run run.py
   ```
   - The app will be available at [http://localhost:8501](http://localhost:8501).

---

## Technical Notes & References

- **ChromaDB Filtering:** Uses `$and` operator for multi-field filters ([ChromaDB Docs](https://docs.trychroma.com/usage-guide#filtering), [LangChain Issue #15417](https://github.com/langchain-ai/langchain/issues/15417)).
- **Batch Processing:** Embeddings are generated in batches to avoid API limits.
- **LLM Q&A:** Retrieval-augmented generation (RAG) pipeline for accurate, reference-backed answers.
- **Streamlit UI:** Modern, interactive, and user-friendly interface with advanced controls.

---

## Acknowledgements

- [ChromaDB](https://www.trychroma.com/)
- [LangChain](https://www.langchain.com/)
- [OpenAI](https://openai.com/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/python/)

---

**This documentation provides a comprehensive overview of the Disney Parks Review Analysis project, covering every major component and design decision.**  
For further details, see the codebase and in-line comments.
