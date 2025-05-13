"""
Insight generator module for Disney Reviews.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger
import config

# Setup logger
logger = setup_logger("insight_generator")

class InsightCategory:
    STAFF = "Staff and Service"
    RIDES = "Rides and Attractions"
    FOOD = "Food and Dining"
    CLEANLINESS = "Cleanliness and Maintenance"
    WAIT_TIMES = "Wait Times and Crowds"
    VALUE = "Value for Money"
    EXPERIENCE = "Overall Experience"
    LOGISTICS = "Logistics and Operations"
    FACILITIES = "Facilities and Amenities"
    
    ALL_CATEGORIES = [
        STAFF, RIDES, FOOD, CLEANLINESS, WAIT_TIMES, 
        VALUE, EXPERIENCE, LOGISTICS, FACILITIES
    ]

class InsightMetric:
    FREQUENCY = "Mention Frequency"
    SENTIMENT = "Sentiment Impact"
    RECENCY = "Recency Weight"
    BUSINESS_IMPACT = "Business Impact"
    IMPLEMENTATION = "Implementation Complexity"
    ROI = "Return on Investment"
    
    ALL_METRICS = [
        FREQUENCY, SENTIMENT, RECENCY, BUSINESS_IMPACT, 
        IMPLEMENTATION, ROI
    ]

def categorize_insights_basic(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Categorize insights based on keywords and bigrams.
    
    Args:
        df (pd.DataFrame): DataFrame with keywords and bigrams
    
    Returns:
        Dict: Dictionary with category and list of insights
    """
    # Define keywords for each category
    category_keywords = {
        InsightCategory.STAFF: [
            'staff', 'service', 'employee', 'employees', 'cast', 'member', 'members',
            'helpful', 'friendly', 'rude', 'attentive', 'polite', 'professional'
        ],
        InsightCategory.RIDES: [
            'ride', 'rides', 'attraction', 'attractions', 'roller', 'coaster', 'thrill',
            'broken', 'closed', 'height', 'safety', 'scary', 'exciting', 'adventure',
            'boring', 'fun', 'thrilling', 'themed', 'theme'
        ],
        InsightCategory.FOOD: [
            'food', 'dining', 'restaurant', 'restaurants', 'meal', 'meals', 'lunch',
            'dinner', 'breakfast', 'snack', 'snacks', 'eat', 'eating', 'ate', 'drink',
            'drinks', 'menu', 'expensive', 'tasty', 'delicious', 'hungry'
        ],
        InsightCategory.CLEANLINESS: [
            'clean', 'cleanliness', 'dirty', 'trash', 'garbage', 'bathroom', 'bathrooms',
            'toilet', 'toilets', 'mess', 'messy', 'filthy', 'neat', 'tidy', 'maintained',
            'maintenance', 'renovation'
        ],
        InsightCategory.WAIT_TIMES: [
            'wait', 'waiting', 'line', 'lines', 'queue', 'queues', 'long', 'short',
            'hour', 'hours', 'crowd', 'crowds', 'crowded', 'busy', 'fast', 'pass',
            'fastpass', 'reservation'
        ],
        InsightCategory.VALUE: [
            'price', 'prices', 'pricing', 'expensive', 'cheap', 'cost', 'costs',
            'worth', 'value', 'money', 'pay', 'paid', 'paying', 'ticket', 'tickets',
            'admission', 'budget', 'afford', 'affordable'
        ],
        InsightCategory.EXPERIENCE: [
            'experience', 'experiences', 'enjoyable', 'enjoy', 'enjoyed', 'magical',
            'magic', 'wonderful', 'amazing', 'awesome', 'terrible', 'horrible',
            'disappointing', 'disappointed', 'vacation', 'holiday', 'memorable'
        ],
        InsightCategory.LOGISTICS: [
            'parking', 'transport', 'transportation', 'bus', 'shuttle', 'train',
            'boat', 'walk', 'walking', 'stroller', 'strollers', 'wheelchair',
            'accessibility', 'accessible', 'map', 'maps', 'direction', 'directions',
            'lost', 'guide', 'entry', 'exit', 'gate', 'gates'
        ],
        InsightCategory.FACILITIES: [
            'hotel', 'hotels', 'resort', 'resorts', 'accommodation', 'accommodations',
            'room', 'rooms', 'pool', 'pools', 'shop', 'shops', 'shopping', 'store',
            'stores', 'merchandise', 'souvenir', 'souvenirs', 'rest', 'area', 'areas',
            'seat', 'seats', 'bench', 'benches', 'shade', 'shelter'
        ]
    }
    
    # Initialize insights dictionary
    insights = {category: [] for category in InsightCategory.ALL_CATEGORIES}
    
    # Process each review
    for _, row in df.iterrows():
        review_id = row['Review_ID']
        review_text = row['Review_Text']
        rating = row['Rating']
        sentiment = row['sentiment_label'] if 'sentiment_label' in row else None
        polarity = row['sentiment_polarity'] if 'sentiment_polarity' in row else 0.0
        keywords = row['keywords'] if 'keywords' in row else []
        
        # Check each keyword against categories
        categorized = False
        for category, category_kw in category_keywords.items():
            matched_keywords = [kw for kw in keywords if kw.lower() in category_kw]
            
            if matched_keywords:
                insights[category].append({
                    'review_id': review_id,
                    'review_text': review_text,
                    'rating': rating,
                    'sentiment': sentiment,
                    'polarity': polarity,
                    'matched_keywords': matched_keywords
                })
                categorized = True
                # Don't break here - a review can belong to multiple categories
        
        # If not categorized, put in Overall Experience
        if not categorized:
            insights[InsightCategory.EXPERIENCE].append({
                'review_id': review_id,
                'review_text': review_text,
                'rating': rating,
                'sentiment': sentiment,
                'polarity': polarity,
                'matched_keywords': []
            })
    
    return insights

def calculate_insight_metrics(insights: Dict[str, List[Dict]], df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Calculate metrics for each insight.
    
    Args:
        insights (Dict): Dictionary with category and list of insights
        df (pd.DataFrame): Original DataFrame with all reviews
    
    Returns:
        Dict: Dictionary with category and list of insights with metrics
    """
    # Get total number of reviews
    total_reviews = len(df)
    
    # Calculate average sentiment polarity
    avg_polarity = df['sentiment_polarity'].mean() if 'sentiment_polarity' in df.columns else 0.0
    
    # Process each category
    for category, category_insights in insights.items():
        # Skip if no insights in this category
        if not category_insights:
            continue
        
        # Calculate frequency metric
        category_frequency = len(category_insights) / total_reviews
        
        # Process each insight
        for insight in category_insights:
            # Frequency metric
            insight['metrics'] = {
                InsightMetric.FREQUENCY: category_frequency
            }
            
            # Sentiment metric (how far from average)
            if 'polarity' in insight:
                sentiment_impact = abs(insight['polarity'] - avg_polarity)
                insight['metrics'][InsightMetric.SENTIMENT] = sentiment_impact
            
            # TODO: Add more metrics as needed
    
    return insights

def rank_insights(insights: Dict[str, List[Dict]], metrics: List[str]) -> Dict[str, List[Dict]]:
    """
    Rank insights based on specified metrics.
    
    Args:
        insights (Dict): Dictionary with category and list of insights with metrics
        metrics (List[str]): List of metrics to use for ranking
    
    Returns:
        Dict: Dictionary with category and ranked list of insights
    """
    ranked_insights = {}
    
    for category, category_insights in insights.items():
        # Skip if no insights in this category
        if not category_insights:
            ranked_insights[category] = []
            continue
        
        # Calculate overall score based on specified metrics
        for insight in category_insights:
            if 'metrics' not in insight:
                insight['overall_score'] = 0.0
                continue
                
            # Calculate overall score as average of available metrics
            available_metrics = [m for m in metrics if m in insight['metrics']]
            if not available_metrics:
                insight['overall_score'] = 0.0
                continue
                
            overall_score = sum(insight['metrics'][m] for m in available_metrics) / len(available_metrics)
            insight['overall_score'] = overall_score
        
        # Sort insights by overall score (descending)
        ranked_insights[category] = sorted(
            category_insights, 
            key=lambda x: x.get('overall_score', 0.0), 
            reverse=True
        )
    
    return ranked_insights

def get_top_insights(insights: Dict[str, List[Dict]], top_n: int = 5) -> Dict[str, List[Dict]]:
    """
    Get top N insights for each category.
    
    Args:
        insights (Dict): Dictionary with category and ranked list of insights
        top_n (int): Number of top insights to return per category
    
    Returns:
        Dict: Dictionary with category and top N insights
    """
    top_insights = {}
    
    for category, category_insights in insights.items():
        top_insights[category] = category_insights[:top_n]
    
    return top_insights

def generate_actionable_recommendations(insights: Dict[str, List[Dict]], top_n: int = 3) -> Dict[str, List[str]]:
    """
    Generate actionable recommendations based on insights.
    
    Args:
        insights (Dict): Dictionary with category and ranked list of insights
        top_n (int): Number of top recommendations to generate per category
    
    Returns:
        Dict: Dictionary with category and list of recommendations
    """
    recommendations = {}
    
    for category, category_insights in insights.items():
        # Skip if no insights in this category
        if not category_insights:
            recommendations[category] = []
            continue
        
        # Get insights based on sentiment
        negative_insights = [i for i in category_insights if i.get('sentiment') == 'negative'][:top_n]
        
        # Generate recommendations based on category and negative insights
        category_recommendations = []
        
        if category == InsightCategory.STAFF:
            if negative_insights:
                category_recommendations.append("Implement additional staff training programs focused on customer service and interaction")
                category_recommendations.append("Increase staff presence in areas with high visitor concentration")
                category_recommendations.append("Review staff scheduling to ensure adequate coverage during peak times")
            
        elif category == InsightCategory.RIDES:
            if negative_insights:
                category_recommendations.append("Increase maintenance frequency for frequently mentioned problematic rides")
                category_recommendations.append("Update ride information to set correct expectations about thrill levels and safety")
                category_recommendations.append("Consider adding more family-friendly attractions that appeal to all age groups")
            
        elif category == InsightCategory.FOOD:
            if negative_insights:
                category_recommendations.append("Review pricing strategy for food and beverages")
                category_recommendations.append("Increase variety of food options, particularly healthy choices")
                category_recommendations.append("Improve quality control processes for food preparation and service")
            
        elif category == InsightCategory.CLEANLINESS:
            if negative_insights:
                category_recommendations.append("Increase frequency of cleaning in high-traffic areas")
                category_recommendations.append("Add more trash receptacles throughout the park")
                category_recommendations.append("Implement a rapid response system for cleanliness issues reported by visitors")
            
        elif category == InsightCategory.WAIT_TIMES:
            if negative_insights:
                category_recommendations.append("Optimize queue management systems to reduce perceived wait times")
                category_recommendations.append("Expand virtual queuing options to more attractions")
                category_recommendations.append("Better communicate expected wait times and peaks to help visitors plan their day")
            
        elif category == InsightCategory.VALUE:
            if negative_insights:
                category_recommendations.append("Review pricing strategy for tickets and packages")
                category_recommendations.append("Create more value-based packages for families")
                category_recommendations.append("Implement seasonal discounts during traditionally slower periods")
            
        elif category == InsightCategory.EXPERIENCE:
            if negative_insights:
                category_recommendations.append("Focus on creating more 'magical moments' throughout the visitor journey")
                category_recommendations.append("Ensure all park areas maintain the Disney standard of immersive experiences")
                category_recommendations.append("Train staff to identify and address visitor disappointment proactively")
            
        elif category == InsightCategory.LOGISTICS:
            if negative_insights:
                category_recommendations.append("Improve signage and wayfinding throughout the park")
                category_recommendations.append("Enhance the park's mobile app with better navigation and real-time updates")
                category_recommendations.append("Review transportation systems between park areas and hotels")
            
        elif category == InsightCategory.FACILITIES:
            if negative_insights:
                category_recommendations.append("Add more rest areas, particularly shaded seating in high-traffic areas")
                category_recommendations.append("Increase the number and accessibility of restroom facilities")
                category_recommendations.append("Review and enhance facilities for guests with special needs")
        
        recommendations[category] = category_recommendations
    
    return recommendations

def generate_insights(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, List[Dict]]]:
    """
    Generate insights from review data.
    
    Args:
        df (pd.DataFrame): DataFrame with processed reviews
    
    Returns:
        Tuple: (recommendations, top_insights)
    """
    logger.info("Generating insights from review data")
    
    # Ensure all required columns are available
    required_columns = ['Review_ID', 'Review_Text', 'Rating', 'keywords', 'sentiment_label', 'sentiment_polarity']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Categorize insights
    insights = categorize_insights_basic(df)
    
    # Calculate metrics
    insights_with_metrics = calculate_insight_metrics(insights, df)
    
    # Rank insights based on metrics
    metrics_to_use = [InsightMetric.FREQUENCY, InsightMetric.SENTIMENT]
    ranked_insights = rank_insights(insights_with_metrics, metrics_to_use)
    
    # Get top insights for each category
    top_insights = get_top_insights(ranked_insights, top_n=config.TOP_N_INSIGHTS)
    
    # Generate actionable recommendations
    recommendations = generate_actionable_recommendations(ranked_insights)
    
    logger.info("Insight generation complete")
    
    return recommendations, top_insights

if __name__ == "__main__":
    # Test the insight generator
    from data_ingestion.data_loader import load_and_preprocess
    from analysis.sentiment import analyze_reviews_sentiment
    from analysis.keywords import extract_all_review_keywords
    
    # Load and preprocess the dataset
    df, _ = load_and_preprocess()
    
    # Analyze sentiment
    sentiment_df = analyze_reviews_sentiment(df)
    
    # Extract keywords
    keywords_df = extract_all_review_keywords(sentiment_df)
    
    # Generate insights
    recommendations, top_insights = generate_insights(keywords_df)
    
    print("\nTop Insights by Category:")
    for category, insights in top_insights.items():
        print(f"\n{category} ({len(insights)} insights):")
        for i, insight in enumerate(insights[:3]):  # Show top 3 for brevity
            print(f"  {i+1}. Score: {insight.get('overall_score', 0.0):.2f}")
            print(f"     Text: {insight['review_text'][:100]}...")
            print(f"     Keywords: {insight.get('matched_keywords', [])}")
    
    print("\nActionable Recommendations by Category:")
    for category, category_recommendations in recommendations.items():
        print(f"\n{category}:")
        for i, recommendation in enumerate(category_recommendations):
            print(f"  {i+1}. {recommendation}") 