"""
Configuration management for MLE-STAR Rossmann Sales Prediction.
Centralized configuration to ensure reproducibility.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "rossmann-store-sales"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)


@dataclass
class DataConfig:
    """Data-related configuration."""
    train_file: Path = DATA_DIR / "train.csv"
    test_file: Path = DATA_DIR / "test.csv"
    store_file: Path = DATA_DIR / "store.csv"
    target_column: str = "Sales"
    date_column: str = "Date"
    id_column: str = "Store"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # Temporal features
    extract_date_features: bool = True
    create_lag_features: bool = True
    lag_periods: List[int] = None
    rolling_windows: List[int] = None

    # Store features
    encode_categorical: bool = True
    competition_bins: int = 5

    # Interaction features
    create_interactions: bool = True

    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 7, 14, 30]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 30]


@dataclass
class ModelConfig:
    """Model training configuration."""
    random_seed: int = 42
    test_size: float = 0.2
    n_folds: int = 5

    # Model-specific parameters
    models_to_train: List[str] = None

    # XGBoost
    xgb_params: Dict[str, Any] = None

    # LightGBM
    lgbm_params: Dict[str, Any] = None

    # Random Forest
    rf_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ["baseline", "random_forest", "xgboost", "lightgbm"]

        if self.xgb_params is None:
            self.xgb_params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.1,
                "max_depth": 7,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_seed,
            }

        if self.lgbm_params is None:
            self.lgbm_params = {
                "objective": "regression",
                "learning_rate": 0.1,
                "max_depth": 7,
                "n_estimators": 200,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.random_seed,
                "verbose": -1,
            }

        if self.rf_params is None:
            self.rf_params = {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_split": 5,
                "random_state": self.random_seed,
                "n_jobs": -1,
            }


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    primary_metric: str = "rmspe"
    additional_metrics: List[str] = None

    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = ["rmse", "mae", "r2"]


class Config:
    """Master configuration class."""

    def __init__(self):
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()
        self.evaluation = EvaluationConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        lines = ["Configuration:"]
        for key, value in self.to_dict().items():
            lines.append(f"\n{key.upper()}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# Global configuration instance
config = Config()


if __name__ == "__main__":
    # Print configuration
    print(config)
