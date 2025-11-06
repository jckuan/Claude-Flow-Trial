"""
Model training pipeline with cross-validation and hyperparameter tuning.
Provides consistent interface for training all model types.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import (
    TimeSeriesSplit, KFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import json
from pathlib import Path
from datetime import datetime


class ModelTrainer:
    """
    Comprehensive model training with cross-validation and hyperparameter tuning.
    Handles time-series aware splitting and experiment tracking.
    """

    def __init__(self, cv_strategy: str = 'timeseries', n_splits: int = 5,
                 random_state: int = 42, verbose: bool = True):
        """
        Initialize ModelTrainer.

        Parameters
        ----------
        cv_strategy : str
            Cross-validation strategy: 'timeseries', 'kfold', or 'custom'
        n_splits : int
            Number of cross-validation splits
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print training progress
        """
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.cv_splitter = self._get_cv_splitter()
        self.training_history_ = []
        self.best_models_ = {}

    def _get_cv_splitter(self):
        """Get cross-validation splitter based on strategy."""
        if self.cv_strategy == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        elif self.cv_strategy == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True,
                        random_state=self.random_state)
        else:
            raise ValueError(f"Unknown cv_strategy: {self.cv_strategy}")

    def train_single_model(self, model: Any, X_train, y_train,
                          X_val=None, y_val=None,
                          model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train a single model and evaluate performance.

        Parameters
        ----------
        model : Any
            Model instance to train
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation targets
        model_name : str, optional
            Name for the model

        Returns
        -------
        Dict[str, Any]
            Training results including metrics and trained model
        """
        if model_name is None:
            model_name = getattr(model, 'name', type(model).__name__)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")

        start_time = datetime.now()

        # Fit model
        try:
            # Check if model supports early stopping
            if hasattr(model, 'fit') and X_val is not None and y_val is not None:
                if 'XGBoost' in model_name or 'LightGBM' in model_name:
                    model.fit(X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=50,
                            verbose=False)
                else:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)

            training_time = (datetime.now() - start_time).total_seconds()

            # Make predictions
            y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)

            results = {
                'model_name': model_name,
                'model': model,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'training_time': training_time,
                'timestamp': datetime.now().isoformat()
            }

            # Validation metrics
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                val_mae = mean_absolute_error(y_val, y_val_pred)
                val_r2 = r2_score(y_val, y_val_pred)

                results.update({
                    'val_rmse': val_rmse,
                    'val_mae': val_mae,
                    'val_r2': val_r2
                })

                if self.verbose:
                    print(f"Train RMSE: {train_rmse:.2f} | MAE: {train_mae:.2f} | R²: {train_r2:.4f}")
                    print(f"Val   RMSE: {val_rmse:.2f} | MAE: {val_mae:.2f} | R²: {val_r2:.4f}")
            else:
                if self.verbose:
                    print(f"Train RMSE: {train_rmse:.2f} | MAE: {train_mae:.2f} | R²: {train_r2:.4f}")

            if self.verbose:
                print(f"Training time: {training_time:.2f}s")

            # Store in history
            self.training_history_.append(results)

            return results

        except Exception as e:
            if self.verbose:
                print(f"❌ Error training {model_name}: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def train_multiple_models(self, models: Dict[str, Any],
                            X_train, y_train,
                            X_val=None, y_val=None) -> pd.DataFrame:
        """
        Train multiple models and compare performance.

        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary mapping model names to model instances
        X_train : array-like
            Training features
        y_train : array-like
            Training targets
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation targets

        Returns
        -------
        pd.DataFrame
            Summary of all model results
        """
        results_list = []

        for name, model in models.items():
            result = self.train_single_model(
                model, X_train, y_train, X_val, y_val, model_name=name
            )

            if 'error' not in result:
                results_list.append({
                    'Model': result['model_name'],
                    'Train_RMSE': result['train_rmse'],
                    'Train_MAE': result['train_mae'],
                    'Train_R2': result['train_r2'],
                    'Val_RMSE': result.get('val_rmse', np.nan),
                    'Val_MAE': result.get('val_mae', np.nan),
                    'Val_R2': result.get('val_r2', np.nan),
                    'Training_Time': result['training_time']
                })

                # Store best model
                self.best_models_[name] = model

        results_df = pd.DataFrame(results_list)

        if len(results_df) > 0 and 'Val_RMSE' in results_df.columns:
            results_df = results_df.sort_values('Val_RMSE')

        if self.verbose:
            print(f"\n{'='*80}")
            print("MODEL COMPARISON SUMMARY")
            print(f"{'='*80}")
            print(results_df.to_string(index=False))
            print(f"{'='*80}\n")

        return results_df

    def cross_validate_model(self, model: Any, X, y,
                           scoring: str = 'neg_root_mean_squared_error') -> Dict[str, Any]:
        """
        Perform cross-validation on a model.

        Parameters
        ----------
        model : Any
            Model instance to cross-validate
        X : array-like
            Features
        y : array-like
            Targets
        scoring : str
            Scoring metric for cross-validation

        Returns
        -------
        Dict[str, Any]
            Cross-validation scores and statistics
        """
        model_name = getattr(model, 'name', type(model).__name__)

        if self.verbose:
            print(f"\nCross-validating {model_name}...")

        try:
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_splitter,
                scoring=scoring,
                n_jobs=-1
            )

            # Convert negative scores to positive for RMSE
            if 'neg' in scoring:
                scores = -scores

            results = {
                'model_name': model_name,
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_scores': scores.tolist(),
                'n_splits': len(scores)
            }

            if self.verbose:
                print(f"  {scoring}: {scores.mean():.2f} (+/- {scores.std():.2f})")

            return results

        except Exception as e:
            if self.verbose:
                print(f"❌ Error in cross-validation: {str(e)}")
            return {'model_name': model_name, 'error': str(e)}

    def tune_hyperparameters(self, model: Any, X, y,
                           param_grid: Dict[str, List],
                           search_type: str = 'grid',
                           n_iter: int = 50,
                           scoring: str = 'neg_root_mean_squared_error') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid or random search.

        Parameters
        ----------
        model : Any
            Model instance to tune
        X : array-like
            Features
        y : array-like
            Targets
        param_grid : Dict[str, List]
            Parameter grid for search
        search_type : str
            'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        n_iter : int
            Number of iterations for random search
        scoring : str
            Scoring metric

        Returns
        -------
        Dict[str, Any]
            Best parameters and model
        """
        model_name = getattr(model, 'name', type(model).__name__)

        if self.verbose:
            print(f"\nTuning hyperparameters for {model_name}...")
            print(f"Search type: {search_type}")
            print(f"Parameter grid: {param_grid}")

        try:
            if search_type == 'grid':
                search = GridSearchCV(
                    model, param_grid,
                    cv=self.cv_splitter,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=1 if self.verbose else 0
                )
            elif search_type == 'random':
                search = RandomizedSearchCV(
                    model, param_grid,
                    n_iter=n_iter,
                    cv=self.cv_splitter,
                    scoring=scoring,
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1 if self.verbose else 0
                )
            else:
                raise ValueError(f"Unknown search_type: {search_type}")

            search.fit(X, y)

            results = {
                'model_name': model_name,
                'best_params': search.best_params_,
                'best_score': -search.best_score_ if 'neg' in scoring else search.best_score_,
                'best_model': search.best_estimator_,
                'cv_results': search.cv_results_
            }

            if self.verbose:
                print(f"\n✅ Best parameters: {search.best_params_}")
                print(f"✅ Best score: {results['best_score']:.2f}")

            return results

        except Exception as e:
            if self.verbose:
                print(f"❌ Error in hyperparameter tuning: {str(e)}")
            return {'model_name': model_name, 'error': str(e)}

    def save_model(self, model: Any, filepath: str, metadata: Optional[Dict] = None):
        """
        Save trained model to disk.

        Parameters
        ----------
        model : Any
            Trained model to save
        filepath : str
            Path to save model
        metadata : Dict, optional
            Additional metadata to save with model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

        # Save metadata
        if metadata is not None:
            metadata_path = filepath.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        if self.verbose:
            print(f"✅ Model saved to: {filepath}")

    def load_model(self, filepath: str) -> Tuple[Any, Optional[Dict]]:
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to load model from

        Returns
        -------
        Tuple[Any, Optional[Dict]]
            Loaded model and metadata (if available)
        """
        filepath = Path(filepath)

        # Load model
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        # Load metadata
        metadata = None
        metadata_path = filepath.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        if self.verbose:
            print(f"✅ Model loaded from: {filepath}")

        return model, metadata

    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame.

        Returns
        -------
        pd.DataFrame
            Training history for all models
        """
        if not self.training_history_:
            return pd.DataFrame()

        history_df = pd.DataFrame(self.training_history_)
        return history_df

    def get_best_model(self, metric: str = 'val_rmse') -> Tuple[str, Any]:
        """
        Get best model based on specified metric.

        Parameters
        ----------
        metric : str
            Metric to use for selection

        Returns
        -------
        Tuple[str, Any]
            Name and instance of best model
        """
        history_df = self.get_training_history()

        if len(history_df) == 0 or metric not in history_df.columns:
            return None, None

        # Lower is better for RMSE/MAE, higher is better for R2
        if 'r2' in metric.lower():
            best_idx = history_df[metric].idxmax()
        else:
            best_idx = history_df[metric].idxmin()

        best_name = history_df.loc[best_idx, 'model_name']
        best_model = history_df.loc[best_idx, 'model']

        return best_name, best_model
