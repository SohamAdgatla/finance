"""
Personal Finance Mistake Detector - Core Package

A hybrid system combining rule-based checks and ML classification
to detect financial health issues and provide corrective suggestions.
"""

from .rule_engine import RuleEngine, RedFlag
from .ml_engine import MLEngine
from .suggestions import SuggestionEngine
from .health_score import calculate_financial_health_score
from .synthetic_data import generate_synthetic_data

__all__ = [
    "RuleEngine",
    "RedFlag",
    "MLEngine",
    "SuggestionEngine",
    "calculate_financial_health_score",
    "generate_synthetic_data",
]
