"""
Ensemble models combining predictions from multiple base models.
Includes stacking and weighted averaging strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
import warnings


class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """
    Weighted ensemble that combines predictions from multiple models.
    Weights can be uniform, performance-based, or custom.
    """

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 weight_strategy: str = 'uniform'):
        """
        Initialize weighted ensemble.

        Parameters
        ----------
        models : List[Any]
            List of trained model instances
        weights : List[float] or None
            Custom weights for each model (must sum to 1)
        weight_strategy : str
            Strategy for weighting: 'uniform', 'performance', or 'custom'
        """
        self.models = models
        self.weights = weights
        self.weight_strategy = weight_strategy
        self.computed_weights_ = None
        self.model_names_ = [getattr(m, 'name', f'model_{i}')
                             for i, m in enumerate(models)]

    def _compute_weights(self, X, y):
        """Compute weights based on individual model performance."""
        if self.weight_strategy == 'uniform':
            self.computed_weights_ = np.ones(len(self.models)) / len(self.models)

        elif self.weight_strategy == 'custom':
            if self.weights is None:
                raise ValueError("Custom weights must be provided when strategy='custom'")
            if len(self.weights) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            if not np.isclose(sum(self.weights), 1.0):
                warnings.warn("Weights do not sum to 1, normalizing...")
                self.computed_weights_ = np.array(self.weights) / sum(self.weights)
            else:
                self.computed_weights_ = np.array(self.weights)

        elif self.weight_strategy == 'performance':
            # Compute weights based on cross-validation RMSE
            from sklearn.metrics import mean_squared_error
            scores = []
            for model in self.models:
                try:
                    y_pred = cross_val_predict(model, X, y, cv=3)
                    rmse = np.sqrt(mean_squared_error(y, y_pred))
                    scores.append(rmse)
                except Exception as e:
                    warnings.warn(f"Error computing score for model: {e}")
                    scores.append(float('inf'))

            # Convert RMSE to weights (inverse weighting)
            scores = np.array(scores)
            inverse_scores = 1.0 / (scores + 1e-10)
            self.computed_weights_ = inverse_scores / inverse_scores.sum()

        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")

    def fit(self, X, y, **kwargs):
        """
        Fit all models in the ensemble.

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        """
        # Compute weights if needed
        if self.computed_weights_ is None:
            self._compute_weights(X, y)

        # Fit all models
        for model in self.models:
            model.fit(X, y)

        return self

    def predict(self, X):
        """
        Make weighted predictions.

        Parameters
        ----------
        X : array-like
            Features to predict

        Returns
        -------
        array-like
            Weighted ensemble predictions
        """
        if self.computed_weights_ is None:
            raise RuntimeError("Model must be fitted before prediction")

        # Collect predictions from all models
        predictions = np.column_stack([model.predict(X) for model in self.models])

        # Compute weighted average
        weighted_pred = np.average(predictions, axis=1, weights=self.computed_weights_)

        return weighted_pred

    def get_weights(self) -> Dict[str, float]:
        """
        Get the computed weights for each model.

        Returns
        -------
        Dict[str, float]
            Dictionary mapping model names to their weights
        """
        if self.computed_weights_ is None:
            return {}
        return dict(zip(self.model_names_, self.computed_weights_))


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble using base models and a meta-learner.
    Base models generate predictions used as features for meta-learner.
    """

    def __init__(self, base_models: List[Any],
                 meta_model: Optional[Any] = None,
                 cv: int = 5,
                 use_original_features: bool = True):
        """
        Initialize stacking ensemble.

        Parameters
        ----------
        base_models : List[Any]
            List of base model instances
        meta_model : Any
            Meta-learner model (default: Ridge regression)
        cv : int
            Number of cross-validation folds for generating meta-features
        use_original_features : bool
            Whether to include original features in meta-learner
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.cv = cv
        self.use_original_features = use_original_features
        self.model_names_ = [getattr(m, 'name', f'model_{i}')
                             for i, m in enumerate(base_models)]

    def fit(self, X, y, **kwargs):
        """
        Fit stacking ensemble.

        Two-stage process:
        1. Generate out-of-fold predictions from base models
        2. Train meta-model on base model predictions

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        """
        # Stage 1: Generate meta-features using cross-validation
        meta_features = []

        for i, model in enumerate(self.base_models):
            print(f"Generating meta-features from {self.model_names_[i]}...")

            # Get out-of-fold predictions
            try:
                oof_predictions = cross_val_predict(
                    model, X, y, cv=self.cv, n_jobs=-1
                )
                meta_features.append(oof_predictions)
            except Exception as e:
                warnings.warn(f"Error generating meta-features for {self.model_names_[i]}: {e}")
                # Fall back to simple predictions
                model.fit(X, y)
                meta_features.append(model.predict(X))

        meta_features = np.column_stack(meta_features)

        # Include original features if specified
        if self.use_original_features:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            meta_features = np.hstack([meta_features, X_array])

        # Stage 2: Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)

        # Fit all base models on full training data
        print("Fitting base models on full data...")
        for i, model in enumerate(self.base_models):
            print(f"  Fitting {self.model_names_[i]}...")
            model.fit(X, y)

        return self

    def predict(self, X):
        """
        Make stacking predictions.

        Parameters
        ----------
        X : array-like
            Features to predict

        Returns
        -------
        array-like
            Stacking ensemble predictions
        """
        # Generate predictions from base models
        base_predictions = []
        for model in self.base_models:
            base_predictions.append(model.predict(X))

        base_predictions = np.column_stack(base_predictions)

        # Include original features if specified
        if self.use_original_features:
            if isinstance(X, pd.DataFrame):
                X_array = X.values
            else:
                X_array = X
            meta_features = np.hstack([base_predictions, X_array])
        else:
            meta_features = base_predictions

        # Meta-model prediction
        return self.meta_model.predict(meta_features)

    def get_base_model_names(self) -> List[str]:
        """Get names of base models."""
        return self.model_names_


class BlendingEnsemble(BaseEstimator, RegressorMixin):
    """
    Blending ensemble using holdout validation set.
    Simpler than stacking, uses single validation split.
    """

    def __init__(self, base_models: List[Any],
                 meta_model: Optional[Any] = None,
                 blend_ratio: float = 0.3):
        """
        Initialize blending ensemble.

        Parameters
        ----------
        base_models : List[Any]
            List of base model instances
        meta_model : Any
            Meta-learner model (default: Ridge regression)
        blend_ratio : float
            Fraction of training data to use for blending (0 < ratio < 1)
        """
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else Ridge(alpha=1.0)
        self.blend_ratio = blend_ratio
        self.model_names_ = [getattr(m, 'name', f'model_{i}')
                             for i, m in enumerate(base_models)]

    def fit(self, X, y, **kwargs):
        """
        Fit blending ensemble.

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training targets
        """
        from sklearn.model_selection import train_test_split

        # Split data for blending
        X_train, X_blend, y_train, y_blend = train_test_split(
            X, y, test_size=self.blend_ratio, random_state=42
        )

        # Train base models on training set
        blend_predictions = []
        for i, model in enumerate(self.base_models):
            print(f"Training {self.model_names_[i]} for blending...")
            model.fit(X_train, y_train)
            blend_predictions.append(model.predict(X_blend))

        blend_predictions = np.column_stack(blend_predictions)

        # Train meta-model on blend set
        print("Training meta-model on blend predictions...")
        self.meta_model.fit(blend_predictions, y_blend)

        return self

    def predict(self, X):
        """Make blending predictions."""
        # Get predictions from base models
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])

        # Meta-model prediction
        return self.meta_model.predict(base_predictions)
