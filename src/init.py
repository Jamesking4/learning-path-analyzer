"""
Learning Path Analyzer - система анализа образовательных траекторий
"""

__version__ = "1.0.0"
__author__ = "Learning Analytics Team"

from .data_parser import LogParser
from .analyzer import LearningAnalyzer
from .visualizer import ResultVisualizer
from .recommender import RecommendationEngine
