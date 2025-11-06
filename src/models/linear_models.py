"""
Linear models with regularization for sales prediction.
Includes Ridge, Lasso, and ElasticNet regression.
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class LinearRegressionModel:
    """
    Simple Linear Regression wrapper for consistent interface.
    """

    def __init__(self, fit_intercept: bool = True):
        """
        Initialize Linear Regression model.

        Parameters
        ----------
        fit_intercept : bool
            Whether to calculate the intercept
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression(fit_intercept=fit_intercept))
        ])
        self.name = "LinearRegression"

    def fit(self, X, y, **kwargs):
        """Fit the linear regression model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the linear regression model."""
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class RidgeModel:
    """
    Ridge Regression with L2 regularization.
    Good for handling multicollinearity in features.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 max_iter: int = 1000):
        """
        Initialize Ridge Regression model.

        Parameters
        ----------
        alpha : float
            Regularization strength (higher = more regularization)
        fit_intercept : bool
            Whether to calculate the intercept
        max_iter : int
            Maximum number of iterations
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                random_state=42
            ))
        ])
        self.name = f"Ridge_alpha{alpha}"
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        """Fit the Ridge regression model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the Ridge regression model."""
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class LassoModel:
    """
    Lasso Regression with L1 regularization.
    Performs feature selection by driving some coefficients to zero.
    """

    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True,
                 max_iter: int = 5000, tol: float = 1e-4):
        """
        Initialize Lasso Regression model.

        Parameters
        ----------
        alpha : float
            Regularization strength (higher = more regularization)
        fit_intercept : bool
            Whether to calculate the intercept
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for optimization
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(
                alpha=alpha,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            ))
        ])
        self.name = f"Lasso_alpha{alpha}"
        self.alpha = alpha

    def fit(self, X, y, **kwargs):
        """Fit the Lasso regression model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the Lasso regression model."""
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


class ElasticNetModel:
    """
    ElasticNet Regression combining L1 and L2 regularization.
    Balances feature selection (L1) and coefficient shrinkage (L2).
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5,
                 fit_intercept: bool = True, max_iter: int = 5000,
                 tol: float = 1e-4):
        """
        Initialize ElasticNet Regression model.

        Parameters
        ----------
        alpha : float
            Overall regularization strength
        l1_ratio : float
            Mix of L1 vs L2 (0 = pure L2, 1 = pure L1)
        fit_intercept : bool
            Whether to calculate the intercept
        max_iter : int
            Maximum number of iterations
        tol : float
            Tolerance for optimization
        """
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=fit_intercept,
                max_iter=max_iter,
                tol=tol,
                random_state=42
            ))
        ])
        self.name = f"ElasticNet_alpha{alpha}_l1{l1_ratio}"
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def fit(self, X, y, **kwargs):
        """Fit the ElasticNet regression model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the ElasticNet regression model."""
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get model parameters."""
        return self.model.get_params(deep=deep)

    def set_params(self, **params):
        """Set model parameters."""
        self.model.set_params(**params)
        return self


def get_linear_models() -> Dict[str, Any]:
    """
    Get dictionary of linear models with various regularization strengths.

    Returns
    -------
    Dict[str, Any]
        Dictionary mapping model names to instantiated linear models
    """
    models = {
        'linear_regression': LinearRegressionModel(),
        'ridge_0.1': RidgeModel(alpha=0.1),
        'ridge_1.0': RidgeModel(alpha=1.0),
        'ridge_10.0': RidgeModel(alpha=10.0),
        'ridge_100.0': RidgeModel(alpha=100.0),
        'lasso_0.1': LassoModel(alpha=0.1),
        'lasso_1.0': LassoModel(alpha=1.0),
        'lasso_10.0': LassoModel(alpha=10.0),
        'elasticnet_0.5': ElasticNetModel(alpha=1.0, l1_ratio=0.5),
        'elasticnet_0.7': ElasticNetModel(alpha=1.0, l1_ratio=0.7),
        'elasticnet_0.9': ElasticNetModel(alpha=1.0, l1_ratio=0.9)
    }
    return models


def get_hyperparameter_grid_ridge() -> Dict[str, list]:
    """
    Get hyperparameter grid for Ridge regression.

    Returns
    -------
    Dict[str, list]
        Hyperparameter grid for grid search
    """
    return {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    }


def get_hyperparameter_grid_lasso() -> Dict[str, list]:
    """
    Get hyperparameter grid for Lasso regression.

    Returns
    -------
    Dict[str, list]
        Hyperparameter grid for grid search
    """
    return {
        'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }


def get_hyperparameter_grid_elasticnet() -> Dict[str, list]:
    """
    Get hyperparameter grid for ElasticNet regression.

    Returns
    -------
    Dict[str, list]
        Hyperparameter grid for grid search
    """
    return {
        'regressor__alpha': [0.01, 0.1, 1.0, 10.0],
        'regressor__l1_ratio': [0.3, 0.5, 0.7, 0.9]
    }
