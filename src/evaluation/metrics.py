"""
Custom evaluation metrics for Rossmann sales prediction.
Primary metric: Root Mean Square Percentage Error (RMSPE)
"""

import numpy as np
import pandas as pd
from typing import Union, Dict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)


def rmspe(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Calculate Root Mean Square Percentage Error.

    RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        RMSPE score
    """
    # Remove zero values (closed stores)
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        logger.warning("No non-zero true values found for RMSPE calculation")
        return np.nan

    # Calculate percentage errors
    percentage_errors = ((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon)) ** 2

    # Return RMSPE
    return np.sqrt(np.mean(percentage_errors))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE score
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE score
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-9) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero

    Returns:
        MAPE score
    """
    # Remove zero values
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        return np.nan

    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / (y_true_filtered + epsilon)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - (ss_res / ss_tot)


def evaluate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metrics: list = None
) -> Dict[str, float]:
    """
    Evaluate predictions using multiple metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        metrics: List of metric names to calculate (default: all)

    Returns:
        Dictionary of metric names and scores
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Default metrics
    if metrics is None:
        metrics = ["rmspe", "rmse", "mae", "mape", "r2"]

    # Calculate metrics
    results = {}

    metric_functions = {
        "rmspe": rmspe,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2_score,
    }

    for metric_name in metrics:
        if metric_name in metric_functions:
            try:
                score = metric_functions[metric_name](y_true, y_pred)
                results[metric_name] = score
                logger.info(f"{metric_name.upper()}: {score:.6f}")
            except Exception as e:
                logger.error(f"Error calculating {metric_name}: {str(e)}")
                results[metric_name] = np.nan
        else:
            logger.warning(f"Unknown metric: {metric_name}")

    return results


def rmspe_xgb(y_pred: np.ndarray, y_true) -> tuple:
    """
    RMSPE objective function for XGBoost.

    Args:
        y_pred: Predicted values
        y_true: DMatrix with true values

    Returns:
        Tuple of (metric_name, score)
    """
    y_true_values = y_true.get_label()
    score = rmspe(y_true_values, y_pred)
    return "rmspe", score


def rmspe_lgb(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    RMSPE objective function for LightGBM.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Tuple of (metric_name, score, is_higher_better)
    """
    score = rmspe(y_true, y_pred)
    return "rmspe", score, False


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
):
    """
    Print a formatted evaluation report.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for display
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} Evaluation Report")
    print(f"{'=' * 60}")

    results = evaluate_predictions(y_true, y_pred)

    print(f"\nPerformance Metrics:")
    print(f"  RMSPE:  {results.get('rmspe', np.nan):.6f} (primary metric)")
    print(f"  RMSE:   {results.get('rmse', np.nan):.2f}")
    print(f"  MAE:    {results.get('mae', np.nan):.2f}")
    print(f"  MAPE:   {results.get('mape', np.nan):.6f}")
    print(f"  R²:     {results.get('r2', np.nan):.6f}")

    print(f"\nPrediction Statistics:")
    print(f"  Mean True:      {np.mean(y_true):.2f}")
    print(f"  Mean Predicted: {np.mean(y_pred):.2f}")
    print(f"  Std True:       {np.std(y_true):.2f}")
    print(f"  Std Predicted:  {np.std(y_pred):.2f}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    # Test metrics with sample data
    np.random.seed(42)

    # Simulate some predictions
    y_true = np.random.randint(1000, 10000, 1000)
    y_pred = y_true + np.random.normal(0, 500, 1000)

    print("Testing evaluation metrics...")
    print_evaluation_report(y_true, y_pred, "Test Model")
