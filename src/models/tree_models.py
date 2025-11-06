"""
Tree-based models for sales prediction.
Includes Random Forest, XGBoost, and LightGBM.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestRegressor
import warnings

# Try to import XGBoost and LightGBM (may not be installed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")


class RandomForestModel:
    """
    Random Forest Regressor for sales prediction.
    Ensemble of decision trees with bagging.
    """

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 max_features: str = 'sqrt', n_jobs: int = -1):
        """
        Initialize Random Forest model.

        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int or None
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at leaf node
        max_features : str
            Number of features to consider for best split
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=42,
            verbose=0
        )
        self.name = f"RandomForest_n{n_estimators}_d{max_depth}"
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y, **kwargs):
        """Fit the Random Forest model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the Random Forest model."""
        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importances from the trained model."""
        return self.model.feature_importances_

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class XGBoostModel:
    """
    XGBoost Regressor for sales prediction.
    Gradient boosting with advanced regularization.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0, n_jobs: int = -1):
        """
        Initialize XGBoost model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth
        learning_rate : float
            Step size shrinkage (eta)
        subsample : float
            Fraction of samples used per tree
        colsample_bytree : float
            Fraction of features used per tree
        reg_alpha : float
            L1 regularization term
        reg_lambda : float
            L2 regularization term
        n_jobs : int
            Number of parallel threads
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=42,
            verbosity=0
        )
        self.name = f"XGBoost_n{n_estimators}_d{max_depth}_lr{learning_rate}"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Fit the XGBoost model.

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        eval_set : list of tuples
            Validation sets for early stopping
        early_stopping_rounds : int
            Stop if no improvement for N rounds
        verbose : bool
            Print training progress
        """
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        if verbose:
            fit_params['verbose'] = True

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """Predict using the XGBoost model."""
        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importances from the trained model."""
        return self.model.feature_importances_

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class LightGBMModel:
    """
    LightGBM Regressor for sales prediction.
    Fast gradient boosting optimized for large datasets.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = -1,
                 learning_rate: float = 0.1, num_leaves: int = 31,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 reg_alpha: float = 0.0, reg_lambda: float = 0.0,
                 n_jobs: int = -1):
        """
        Initialize LightGBM model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting rounds
        max_depth : int
            Maximum tree depth (-1 = no limit)
        learning_rate : float
            Boosting learning rate
        num_leaves : int
            Maximum number of leaves per tree
        subsample : float
            Fraction of samples used per iteration
        colsample_bytree : float
            Fraction of features used per tree
        reg_alpha : float
            L1 regularization
        reg_lambda : float
            L2 regularization
        n_jobs : int
            Number of parallel threads
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")

        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=42,
            verbosity=-1
        )
        self.name = f"LightGBM_n{n_estimators}_d{max_depth}_lr{learning_rate}"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, verbose=False):
        """
        Fit the LightGBM model.

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        eval_set : list of tuples
            Validation sets for early stopping
        early_stopping_rounds : int
            Stop if no improvement for N rounds
        verbose : bool
            Print training progress
        """
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        if early_stopping_rounds is not None:
            fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds)]

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        """Predict using the LightGBM model."""
        return self.model.predict(X)

    def get_feature_importance(self):
        """Get feature importances from the trained model."""
        return self.model.feature_importances_

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


def get_tree_models() -> Dict[str, Any]:
    """
    Get dictionary of tree-based models with default configurations.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to instantiated models
    """
    models = {
        'random_forest_100': RandomForestModel(n_estimators=100, max_depth=20),
        'random_forest_200': RandomForestModel(n_estimators=200, max_depth=20),
    }

    if XGBOOST_AVAILABLE:
        models.update({
            'xgboost_100': XGBoostModel(n_estimators=100, max_depth=6, learning_rate=0.1),
            'xgboost_200': XGBoostModel(n_estimators=200, max_depth=6, learning_rate=0.05),
            'xgboost_deep': XGBoostModel(n_estimators=150, max_depth=10, learning_rate=0.05),
        })

    if LIGHTGBM_AVAILABLE:
        models.update({
            'lightgbm_100': LightGBMModel(n_estimators=100, max_depth=8, learning_rate=0.1),
            'lightgbm_200': LightGBMModel(n_estimators=200, max_depth=8, learning_rate=0.05),
            'lightgbm_large': LightGBMModel(n_estimators=300, max_depth=10, learning_rate=0.03),
        })

    return models


def get_hyperparameter_grid_rf() -> Dict[str, list]:
    """Get hyperparameter grid for Random Forest."""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }


def get_hyperparameter_grid_xgb() -> Dict[str, list]:
    """Get hyperparameter grid for XGBoost."""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }


def get_hyperparameter_grid_lgbm() -> Dict[str, list]:
    """Get hyperparameter grid for LightGBM."""
    return {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10, -1],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }
