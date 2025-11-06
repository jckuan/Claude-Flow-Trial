"""
Machine Learning Models for Rossmann Sales Prediction
"""

from .baseline import BaselineModel, MeanBaseline, MedianBaseline
from .linear_models import LinearRegressionModel, RidgeModel, LassoModel, ElasticNetModel
from .ensemble_models import StackingEnsemble, WeightedEnsemble
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator

# Import tree models with optional dependencies
try:
    from .tree_models import RandomForestModel, XGBoostModel, LightGBMModel
    tree_models_available = True
except ImportError as e:
    print(f"Warning: Some tree models unavailable - {e}")
    # Still export what's available
    try:
        from .tree_models import RandomForestModel
        tree_models_available = 'partial'
    except:
        tree_models_available = False

__all__ = [
    'BaselineModel',
    'MeanBaseline',
    'MedianBaseline',
    'LinearRegressionModel',
    'RidgeModel',
    'LassoModel',
    'ElasticNetModel',
    'RandomForestModel',
    'StackingEnsemble',
    'WeightedEnsemble',
    'ModelTrainer',
    'ModelEvaluator'
]

if tree_models_available == True:
    __all__.extend(['XGBoostModel', 'LightGBMModel'])

__version__ = '1.0.0'
