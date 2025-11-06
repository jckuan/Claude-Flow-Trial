"""
Feature Engineering Module for Rossmann Sales Prediction

This module provides comprehensive feature engineering capabilities including:
- Temporal features (day, month, year, quarter, week)
- Lag features (previous sales patterns)
- Rolling statistics (moving averages)
- Categorical encoding
- Interaction features
- Missing value handling
- Preprocessing and scaling
"""

from .temporal_features import TemporalFeatureEngineer
from .categorical_features import CategoricalFeatureEngineer
from .lag_features import LagFeatureEngineer
from .preprocessing import DataPreprocessor
from .pipeline import FeatureEngineeringPipeline

__all__ = [
    'TemporalFeatureEngineer',
    'CategoricalFeatureEngineer',
    'LagFeatureEngineer',
    'DataPreprocessor',
    'FeatureEngineeringPipeline'
]

__version__ = '1.0.0'
