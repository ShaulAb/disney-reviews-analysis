"""
Insights package for Disney Reviews Analysis project.
"""
from insights.insight_generator import (
    InsightCategory,
    InsightMetric,
    categorize_insights_basic,
    calculate_insight_metrics,
    rank_insights,
    get_top_insights,
    generate_actionable_recommendations,
    generate_insights
)

__all__ = [
    "InsightCategory",
    "InsightMetric",
    "categorize_insights_basic",
    "calculate_insight_metrics",
    "rank_insights",
    "get_top_insights",
    "generate_actionable_recommendations",
    "generate_insights"
] 