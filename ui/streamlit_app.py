"""
Streamlit UI for Disney Reviews Analysis.
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional
import datetime
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingestion.data_loader import load_and_preprocess
from analysis.sentiment import analyze_reviews_sentiment
from analysis.keywords import extract_all_review_keywords
from insights.insight_generator import generate_insights, InsightCategory
from llm_qa.document_store import DisneyReviewsDocumentStore
from llm_qa.qa_chain import create_qa_chain
from utils.logger import setup_logger
import config

# Setup logger for the UI
logger = setup_logger("streamlit_ui")

# Set page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to verify ChromaDB
def verify_chromadb():
    """
    Verify that ChromaDB exists and has documents.
    
    Returns:
        Tuple[bool, str, int]: (success, message, document_count)
    """
    logger.info("Verifying ChromaDB")
    
    # Check if ChromaDB directory exists
    chroma_dir = os.path.join(config.CACHE_DIR, "chroma")
    if not os.path.exists(chroma_dir):
        error_msg = f"ChromaDB directory not found at {chroma_dir}"
        logger.error(error_msg)
        return False, error_msg, 0
    
    try:
        # Try to load the existing ChromaDB
        doc_store = DisneyReviewsDocumentStore(persist_directory=chroma_dir)
        
        # Check if the vector store has documents
        collection = doc_store.vector_store._collection
        doc_count = collection.count()
        
        if doc_count == 0:
            error_msg = "ChromaDB exists but contains no documents"
            logger.error(error_msg)
            return False, error_msg, 0
        
        logger.info(f"ChromaDB verification successful: found {doc_count} documents")
        return True, f"ChromaDB verified: {doc_count} documents found", doc_count
        
    except Exception as e:
        error_msg = f"Error verifying ChromaDB: {str(e)}"
        logger.error(error_msg)
        return False, error_msg, 0

# Cache data loading
@st.cache_data
def load_data():
    """Load and process data."""
    logger.info("Loading and processing data")
    
    # Load and preprocess data
    df, stats = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Extract keywords
    keywords_df = extract_all_review_keywords(sentiment_df)
    
    # Generate insights
    recommendations, top_insights = generate_insights(keywords_df)
    
    logger.info("Data loading and processing complete")
    return keywords_df, stats, recommendations, top_insights

@st.cache_resource
def init_qa_system(df):
    """
    Initialize the QA system if ChromaDB is available and valid.
    
    Args:
        df: DataFrame with review data
        
    Returns:
        QAChain or None: Returns QA chain if successful, None otherwise
    """
    logger.info("Initializing QA system")
    
    # Verify ChromaDB first
    is_valid, message, doc_count = verify_chromadb()
    
    if not is_valid:
        logger.warning(f"QA system initialization skipped: {message}")
        return None
    
    try:
        # Create document store (will use existing ChromaDB)
        doc_store = DisneyReviewsDocumentStore()
        
        # Create QA chain
        qa_chain = create_qa_chain(doc_store)
        
        logger.info(f"QA system initialization complete with {doc_count} documents")
        return qa_chain
    
    except Exception as e:
        error_msg = f"Error initializing QA system: {str(e)}"
        logger.error(error_msg)
        return None

# Main app
def main():
    """Main application."""
    # Display title
    st.title(config.APP_TITLE)
    st.write(config.APP_SUBTITLE)
    
    # Load data
    with st.spinner("Loading data..."):
        df, stats, recommendations, top_insights = load_data()
    
    # Initialize QA system
    with st.spinner("Initializing QA system..."):
        qa_chain = init_qa_system(df)
        qa_system_available = qa_chain is not None
    
    # Create sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "Insights & Recommendations", "Chat Q&A"]
    )
    
    # Display QA system status in sidebar
    if not qa_system_available:
        st.sidebar.error("⚠️ Chat Q&A system is unavailable. Please run vector database initialization first.")
    
    # Display pages
    if page == "Dashboard":
        display_dashboard(df, stats)
    elif page == "Insights & Recommendations":
        display_insights(recommendations, top_insights)
    elif page == "Chat Q&A":
        display_chat_qa(qa_chain)

def display_dashboard(df, stats):
    """Display dashboard."""
    st.header("Disney Parks Review Dashboard")
    
    # Display basic stats
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", stats['total_reviews'])
    
    with col2:
        st.metric("Average Rating", f"{stats['average_rating']:.2f}/5.0")
    
    with col3:
        branches = len(stats['reviews_per_branch'])
        st.metric("Disney Branches", branches)
    
    # Display rating distribution
    st.subheader("Rating Distribution")
    rating_df = pd.DataFrame({
        'Rating': list(stats['rating_distribution'].keys()),
        'Count': list(stats['rating_distribution'].values())
    })
    fig = px.bar(
        rating_df, 
        x='Rating', 
        y='Count',
        labels={'Count': 'Number of Reviews', 'Rating': 'Rating (1-5)'},
        color='Rating',
        color_continuous_scale=px.colors.sequential.Blues
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display reviews by branch
    st.subheader("Reviews by Branch")
    branch_df = pd.DataFrame({
        'Branch': list(stats['reviews_per_branch'].keys()),
        'Count': list(stats['reviews_per_branch'].values())
    })
    fig = px.pie(
        branch_df, 
        values='Count', 
        names='Branch',
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display sentiment distribution
    if 'sentiment_label' in df.columns:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment_label'].value_counts()
        sentiment_df = pd.DataFrame({
            'Sentiment': sentiment_counts.index,
            'Count': sentiment_counts.values
        })
        
        fig = px.pie(
            sentiment_df,
            values='Count',
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'positive': 'green',
                'neutral': 'grey',
                'negative': 'red'
            },
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display reviews over time
    if 'Year' in df.columns and 'Month' in df.columns:
        st.subheader("Reviews Over Time")
        
        # Create date column
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
        
        # Group by date
        time_df = df.groupby('Date').size().reset_index(name='Count')
        
        fig = px.line(
            time_df,
            x='Date',
            y='Count',
            labels={'Count': 'Number of Reviews', 'Date': 'Date'},
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display top reviewer locations
    st.subheader("Top Reviewer Locations")
    top_locations = pd.DataFrame({
        'Location': list(stats['top_reviewer_locations'].keys()),
        'Count': list(stats['top_reviewer_locations'].values())
    }).sort_values('Count', ascending=False)
    
    fig = px.bar(
        top_locations,
        x='Location',
        y='Count',
        labels={'Count': 'Number of Reviews', 'Location': 'Reviewer Location'},
        color='Count',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig, use_container_width=True)

def display_insights(recommendations, top_insights):
    """Display insights and recommendations."""
    st.header("Insights & Recommendations")
    
    # Create tabs for each category
    tabs = st.tabs([category for category in InsightCategory.ALL_CATEGORIES])
    
    for i, category in enumerate(InsightCategory.ALL_CATEGORIES):
        with tabs[i]:
            st.subheader(f"{category} Insights")
            
            # Display recommendations
            if category in recommendations and recommendations[category]:
                st.markdown("### Recommendations")
                for j, recommendation in enumerate(recommendations[category]):
                    st.markdown(f"**{j+1}. {recommendation}**")
            else:
                st.info("No specific recommendations for this category.")
            
            # Display insights
            if category in top_insights and top_insights[category]:
                st.markdown("### Supporting Insights from Reviews")
                
                # Create expandable sections for each insight
                for j, insight in enumerate(top_insights[category][:5]):  # Show top 5
                    with st.expander(f"Insight {j+1} (Score: {insight.get('overall_score', 0.0):.2f})"):
                        st.markdown(f"**Review Text:**")
                        st.write(insight['review_text'])
                        
                        st.markdown(f"**Rating:** {insight.get('rating', 'N/A')}")
                        st.markdown(f"**Sentiment:** {insight.get('sentiment', 'N/A')}")
                        
                        if 'matched_keywords' in insight and insight['matched_keywords']:
                            st.markdown(f"**Keywords:** {', '.join(insight['matched_keywords'])}")
            else:
                st.info("No specific insights for this category.")

def export_chat_as_txt(messages):
    """
    Export chat conversation as a text file.
    
    Args:
        messages (List): List of message dictionaries
    
    Returns:
        str: Path to the exported file
    """
    logger.info("Exporting chat conversation as text file")
    
    # Create directory for exported chats if it doesn't exist
    export_dir = os.path.join("ui", "exported_chats")
    os.makedirs(export_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"disney_chat_{timestamp}.txt"
    filepath = os.path.join(export_dir, filename)
    
    # Write conversation to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Disney Parks Review Q&A Conversation - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for msg in messages:
            # Write role and content
            f.write(f"## {msg['role'].upper()}:\n")
            f.write(f"{msg['content']}\n\n")
            
            # Write references if available
            if "references" in msg and msg["references"]:
                f.write("### References:\n")
                for i, ref in enumerate(msg["references"]):
                    f.write(f"Reference {i+1}:\n")
                    f.write(f"Metadata: {ref['metadata']}\n")
                    f.write(f"Content: {ref['content']}\n\n")
            
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"Chat exported to {filepath}")
    return filepath

def display_chat_qa(qa_chain):
    """Display chat Q&A interface."""
    st.header("Disney Parks Review Q&A")
    
    # Check if QA system is available
    if qa_chain is None:
        st.error("⚠️ The Chat Q&A system is currently unavailable.")
        st.info("""
        **Why this happened:**
        - The vector database (ChromaDB) may not be initialized
        - The vector database may be empty
        
        **How to fix it:**
        1. Run the vector database initialization script (`initialize_vector_db.py` or notebook)
        2. Restart the Streamlit app
        """)
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add chat controls in sidebar
    st.sidebar.markdown("### Chat Controls")
    
    # Add clear button
    if st.sidebar.button("Clear Chat"):
        logger.info("Clearing chat history")
        st.session_state.messages = []
        st.rerun()
    
    # Add export button
    if st.session_state.messages and st.sidebar.button("Export Chat (.txt)"):
        try:
            filepath = export_chat_as_txt(st.session_state.messages)
            st.sidebar.success(f"Chat exported to {os.path.basename(filepath)}")
            logger.info(f"Chat successfully exported to {filepath}")
        except Exception as e:
            st.sidebar.error(f"Error exporting chat: {str(e)}")
            logger.error(f"Failed to export chat: {str(e)}")
    
    # Add a timeout setting
    st.sidebar.markdown("### Settings")
    
    # Set default timeout to 60 seconds
    if "timeout_value" not in st.session_state:
        st.session_state.timeout_value = 60
    
    timeout = st.sidebar.slider(
        "Response Timeout (seconds)", 
        min_value=60, 
        max_value=300, 
        value=st.session_state.timeout_value,
        step=30
    )
    st.session_state.timeout_value = timeout
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "references" in message and message["references"]:
                with st.expander("Show References"):
                    for i, ref in enumerate(message["references"]):
                        st.markdown(f"**Reference {i+1}:**")
                        st.markdown(f"*{ref['metadata']}*")
                        st.markdown(ref['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about Disney parks..."):
        logger.info(f"User question: {prompt}")
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("Thinking..."):
                try:
                    # Set the timeout for the response
                    # Note: This is a simple implementation and may not work perfectly in all cases
                    logger.info(f"Processing question with {timeout} second timeout")
                    
                    # Generate answer
                    answer, reviews = qa_chain.answer_question(prompt)
                    
                    # Format references
                    references = []
                    for review in reviews:
                        references.append({
                            "metadata": ", ".join([f"{k}: {v}" for k, v in review.metadata.items()]),
                            "content": review.page_content
                        })
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    logger.info(f"Question answered in {processing_time:.2f} seconds")
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display references
                    if references:
                        with st.expander("Show References"):
                            for i, ref in enumerate(references):
                                st.markdown(f"**Reference {i+1}:**")
                                st.markdown(f"*{ref['metadata']}*")
                                st.markdown(ref['content'])
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "references": references,
                        "processing_time": processing_time
                    })
                    
                    logger.info(f"Response added to chat history")
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg)
                    
                    # Add error response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I'm sorry, I encountered an error while trying to answer your question: {str(e)}",
                        "references": []
                    })

if __name__ == "__main__":
    main() 