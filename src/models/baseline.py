"""
Baseline models for Rossmann sales prediction.
Provides simple benchmarks for comparison with more complex models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


class BaselineModel(BaseEstimator, RegressorMixin):
    """Base class for baseline models."""

    def __init__(self):
        self.prediction_value_ = None
        self.is_fitted_ = False

    def fit(self, X, y, **kwargs):
        """Fit the baseline model."""
        raise NotImplementedError("Subclasses must implement fit method")

    def predict(self, X):
        """Predict using the baseline model."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")
        return np.full(len(X), self.prediction_value_)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        return self


class MeanBaseline(BaselineModel):
    """
    Simple baseline that predicts the mean of training targets.
    Useful for establishing minimum performance threshold.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y, **kwargs):
        """
        Fit by computing mean of target values.

        Parameters
        ----------
        X : array-like
            Feature matrix (not used, but required for sklearn compatibility)
        y : array-like
            Target values
        """
        self.prediction_value_ = np.mean(y)
        self.is_fitted_ = True
        return self


class MedianBaseline(BaselineModel):
    """
    Baseline that predicts the median of training targets.
    More robust to outliers than mean baseline.
    """

    def __init__(self):
        super().__init__()

    def fit(self, X, y, **kwargs):
        """
        Fit by computing median of target values.

        Parameters
        ----------
        X : array-like
            Feature matrix (not used, but required for sklearn compatibility)
        y : array-like
            Target values
        """
        self.prediction_value_ = np.median(y)
        self.is_fitted_ = True
        return self


class SimpleLinearBaseline(LinearRegression):
    """
    Simple linear regression baseline without regularization.
    Serves as a more sophisticated baseline than mean/median.
    """

    def __init__(self):
        super().__init__()
        self.name = "SimpleLinearBaseline"

    def fit(self, X, y, **kwargs):
        """Fit linear regression model."""
        return super().fit(X, y)


class StoreAverageBaseline(BaseEstimator, RegressorMixin):
    """
    Baseline that predicts average sales per store.
    Takes into account store-specific patterns.
    """

    def __init__(self, store_col: str = 'Store'):
        self.store_col = store_col
        self.store_means_ = None
        self.global_mean_ = None
        self.is_fitted_ = False

    def fit(self, X, y, **kwargs):
        """
        Fit by computing mean sales per store.

        Parameters
        ----------
        X : DataFrame
            Feature matrix containing store identifier
        y : array-like
            Target sales values
        """
        if isinstance(X, pd.DataFrame):
            data = pd.DataFrame({self.store_col: X[self.store_col], 'Sales': y})
            self.store_means_ = data.groupby(self.store_col)['Sales'].mean().to_dict()
            self.global_mean_ = np.mean(y)
        else:
            raise ValueError("X must be a pandas DataFrame with store identifier")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using store-specific means.
        Falls back to global mean for unseen stores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            predictions = X[self.store_col].map(self.store_means_)
            predictions = predictions.fillna(self.global_mean_)
            return predictions.values
        else:
            raise ValueError("X must be a pandas DataFrame with store identifier")

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'store_col': self.store_col}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class DayOfWeekBaseline(BaseEstimator, RegressorMixin):
    """
    Baseline that predicts average sales per day of week.
    Captures weekly seasonality patterns.
    """

    def __init__(self, dow_col: str = 'DayOfWeek'):
        self.dow_col = dow_col
        self.dow_means_ = None
        self.global_mean_ = None
        self.is_fitted_ = False

    def fit(self, X, y, **kwargs):
        """
        Fit by computing mean sales per day of week.

        Parameters
        ----------
        X : DataFrame
            Feature matrix containing day of week
        y : array-like
            Target sales values
        """
        if isinstance(X, pd.DataFrame):
            data = pd.DataFrame({self.dow_col: X[self.dow_col], 'Sales': y})
            self.dow_means_ = data.groupby(self.dow_col)['Sales'].mean().to_dict()
            self.global_mean_ = np.mean(y)
        else:
            raise ValueError("X must be a pandas DataFrame with day of week")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict using day-of-week specific means."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            predictions = X[self.dow_col].map(self.dow_means_)
            predictions = predictions.fillna(self.global_mean_)
            return predictions.values
        else:
            raise ValueError("X must be a pandas DataFrame with day of week")

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {'dow_col': self.dow_col}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def get_baseline_models() -> Dict[str, BaseEstimator]:
    """
    Get dictionary of all baseline models.

    Returns
    -------
    Dict[str, BaseEstimator]
        Dictionary mapping model names to instantiated baseline models
    """
    return {
        'mean': MeanBaseline(),
        'median': MedianBaseline(),
        'linear': SimpleLinearBaseline()
    }
