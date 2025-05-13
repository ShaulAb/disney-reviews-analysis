"""
Q&A chain module for the Disney Reviews Q&A system.
"""
import os
from typing import Dict, List, Tuple, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
from llm_qa.document_store import DisneyReviewsDocumentStore
import config

# Setup logger
logger = setup_logger("qa_chain")

# Define output schema
class QuestionAnalysis(BaseModel):
    """Schema for parsing a user's question."""
    query: str = Field(description="The semantic search query extracted from the question")
    branch: Optional[str] = Field(None, description="The specific Disney branch/park mentioned in the question (if any)")
    reviewer_location: Optional[str] = Field(None, description="The reviewer location mentioned in the question (if any)")
    rating: Optional[int] = Field(None, description="The rating mentioned in the question (if any)")
    time_period: Optional[str] = Field(None, description="The time period mentioned in the question (if any)")
    sentiment: Optional[str] = Field(None, description="The sentiment mentioned in the question (positive, negative, neutral)")
    num_results: int = Field(5, description="The number of results to return, default is 5")

class QAResponse(BaseModel):
    """Schema for the QA response."""
    answer: str = Field(description="The answer to the user's question")
    context_used: List[str] = Field(description="The context used to generate the answer")
    metadata_used: List[Dict[str, Any]] = Field(description="The metadata of the context used")

# Question analysis prompt
QUESTION_ANALYSIS_PROMPT = """
You are an AI assistant helping to analyze questions about Disney park reviews.

Given a question about Disney parks, your task is to extract search parameters to help find relevant reviews.

The question: {question}

Extract the following information:
1. The core query for semantic search
2. Any specific Disney branch/park mentioned (e.g., "Disneyland_California", "Disneyland_Paris", "Disneyland_HongKong", "Walt Disney World_Orlando")
3. Any specific reviewer location mentioned (e.g., "Australia", "United States", "Japan")
4. Any specific rating mentioned (1-5)
5. Any time period mentioned (convert to YYYY-M format if possible, e.g., "2019-3" for March 2019)
6. Any sentiment mentioned (positive, negative, neutral)
7. Number of results to return (default: 5)

{format_instructions}
"""

# Answer generation prompt
ANSWER_GENERATION_PROMPT = """
You are an AI assistant providing helpful information about Disney parks based on customer reviews.

Question: {question}

Below are excerpts from relevant customer reviews that may help answer this question:

{context}

Based ONLY on the provided review excerpts, provide a comprehensive answer to the question.
Be specific and reference details from the reviews where possible.
If the reviews don't contain enough information to fully answer the question, acknowledge the limitations.
Always maintain a helpful, balanced, and informative tone.

Your answer should be well-structured and easy to understand.
"""

class DisneyReviewsQAChain:
    """QA chain for answering questions about Disney reviews."""
    
    def __init__(self, document_store: DisneyReviewsDocumentStore):
        """
        Initialize the QA chain.
        
        Args:
            document_store (DisneyReviewsDocumentStore): Document store containing reviews
        """
        self.document_store = document_store
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
            temperature=0.2,
        )
        
        # Initialize question analyzer
        self._initialize_question_analyzer()
        
        # Initialize answer generator
        self._initialize_answer_generator()
    
    def _initialize_question_analyzer(self) -> None:
        """Initialize the question analyzer chain."""
        parser = PydanticOutputParser(pydantic_object=QuestionAnalysis)
        
        prompt = PromptTemplate(
            template=QUESTION_ANALYSIS_PROMPT,
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        self.question_analyzer = LLMChain(
            llm=self.llm,
            prompt=prompt,
            output_parser=parser,
            verbose=True
        )
    
    def _initialize_answer_generator(self) -> None:
        """Initialize the answer generator chain."""
        prompt = PromptTemplate(
            template=ANSWER_GENERATION_PROMPT,
            input_variables=["question", "context"]
        )
        
        self.answer_generator = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True
        )
    
    def analyze_question(self, question: str) -> QuestionAnalysis:
        """
        Analyze a question to extract search parameters.
        
        Args:
            question (str): User's question
        
        Returns:
            QuestionAnalysis: Extracted search parameters
        """
        logger.info(f"Analyzing question: {question}")
        
        try:
            result = self.question_analyzer.run(question=question)
            if isinstance(result, QuestionAnalysis):
                return result
            elif isinstance(result, str):
                # Sometimes the LLM might return a JSON string instead of a parsed object
                import json
                try:
                    data = json.loads(result)
                    return QuestionAnalysis(**data)
                except:
                    logger.error(f"Failed to parse result as JSON: {result}")
                    
            logger.info(f"Question analysis result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing question: {str(e)}")
            # Return default analysis
            return QuestionAnalysis(query=question)
    
    def retrieve_relevant_reviews(self, analysis: QuestionAnalysis) -> List[Document]:
        """
        Retrieve relevant reviews based on question analysis.
        
        Args:
            analysis (QuestionAnalysis): Extracted search parameters
        
        Returns:
            List[Document]: Relevant reviews
        """
        logger.info(f"Retrieving relevant reviews for query: {analysis.query}")
        
        return self.document_store.get_relevant_reviews(
            query=analysis.query,
            k=analysis.num_results,
            branch=analysis.branch,
            reviewer_location=analysis.reviewer_location,
            rating=analysis.rating,
            year_month=analysis.time_period,
            sentiment=analysis.sentiment
        )
    
    def generate_answer(self, question: str, reviews: List[Document]) -> str:
        """
        Generate an answer based on relevant reviews.
        
        Args:
            question (str): User's question
            reviews (List[Document]): Relevant reviews
        
        Returns:
            str: Generated answer
        """
        logger.info(f"Generating answer for question: {question}")
        
        if not reviews:
            return "I couldn't find any relevant reviews to answer your question."
        
        # Format context
        context = ""
        for i, review in enumerate(reviews):
            metadata_str = ", ".join([f"{k}: {v}" for k, v in review.metadata.items()])
            context += f"REVIEW {i+1}:\n"
            context += f"[Metadata: {metadata_str}]\n"
            context += f"Review Text: {review.page_content}\n\n"
        
        # Generate answer
        result = self.answer_generator.run(
            question=question,
            context=context
        )
        
        logger.info(f"Generated answer: {result[:100]}...")
        
        return result
    
    def answer_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Answer a question about Disney reviews.
        
        Args:
            question (str): User's question
        
        Returns:
            Tuple[str, List[Document]]: Generated answer and relevant reviews
        """
        logger.info(f"Answering question: {question}")
        
        # Analyze question
        analysis = self.analyze_question(question)
        
        # Retrieve relevant reviews
        reviews = self.retrieve_relevant_reviews(analysis)
        
        # Generate answer
        answer = self.generate_answer(question, reviews)
        
        return answer, reviews

def create_qa_chain(document_store: DisneyReviewsDocumentStore) -> DisneyReviewsQAChain:
    """
    Create a QA chain.
    
    Args:
        document_store (DisneyReviewsDocumentStore): Document store containing reviews
    
    Returns:
        DisneyReviewsQAChain: QA chain
    """
    return DisneyReviewsQAChain(document_store)

if __name__ == "__main__":
    # Test the QA chain
    from data_ingestion.data_loader import load_and_preprocess
    from analysis.sentiment import analyze_reviews_sentiment
    from llm_qa.document_store import create_and_populate_document_store
    
    # Load and preprocess the dataset
    df, _ = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Create and populate document store
    doc_store = create_and_populate_document_store(sentiment_df)
    
    # Create QA chain
    qa_chain = create_qa_chain(doc_store)
    
    # Test questions
    questions = [
        "What do visitors from Australia say about Disneyland in Hong Kong?",
        "Is spring a good time to visit Disneyland?",
        "Is Disneyland California usually crowded in June?",
        "Is the staff in Paris friendly?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        
        answer, reviews = qa_chain.answer_question(question)
        
        print(f"Answer: {answer}")
        print(f"Based on {len(reviews)} reviews")
        
        for i, review in enumerate(reviews[:2]):  # Show first 2 for brevity
            print(f"\nReview {i+1}:")
            print(f"Metadata: {review.metadata}")
            print(f"Content: {review.page_content[:100]}...") 