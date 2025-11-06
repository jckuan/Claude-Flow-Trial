"""
Comprehensive model evaluation with multiple metrics and visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from pathlib import Path
import warnings


class ModelEvaluator:
    """
    Comprehensive model evaluation with metrics and visualizations.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize ModelEvaluator.

        Parameters
        ----------
        verbose : bool
            Whether to print evaluation results
        """
        self.verbose = verbose
        self.evaluation_results_ = {}

    def calculate_metrics(self, y_true, y_pred, model_name: str = 'Model') -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Parameters
        ----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Remove any negative predictions for sales data
        y_pred = np.maximum(y_pred, 0)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE (handle zero values)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
        else:
            mape = np.nan

        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        residuals = y_true - y_pred
        residual_std = np.std(residuals)

        metrics = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'mape': mape,
            'residual_std': residual_std,
            'n_samples': len(y_true)
        }

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation Metrics: {model_name}")
            print(f"{'='*60}")
            print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
            print(f"MAE  (Mean Absolute Error):     {mae:.2f}")
            print(f"MAPE (Mean Absolute % Error):   {mape:.2f}%")
            print(f"R²   (R-squared):               {r2:.4f}")
            print(f"Residual Std Dev:               {residual_std:.2f}")
            print(f"{'='*60}\n")

        self.evaluation_results_[model_name] = metrics
        return metrics

    def evaluate_multiple_models(self, models: Dict[str, Any],
                                X_test, y_test) -> pd.DataFrame:
        """
        Evaluate multiple models and create comparison.

        Parameters
        ----------
        models : Dict[str, Any]
            Dictionary of trained models
        X_test : array-like
            Test features
        y_test : array-like
            Test targets

        Returns
        -------
        pd.DataFrame
            Comparison of all model metrics
        """
        results = []

        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                metrics = self.calculate_metrics(y_test, y_pred, model_name=name)
                results.append(metrics)
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error evaluating {name}: {str(e)}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            results_df = results_df.sort_values('rmse')

        return results_df

    def plot_predictions(self, y_true, y_pred, model_name: str = 'Model',
                        save_path: Optional[str] = None):
        """
        Plot actual vs predicted values.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Sales', fontsize=12)
        axes[0].set_ylabel('Predicted Sales', fontsize=12)
        axes[0].set_title(f'Actual vs Predicted: {model_name}', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Residuals plot
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Predicted Sales', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title(f'Residuals Plot: {model_name}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")

        plt.show()

    def plot_residuals_distribution(self, y_true, y_pred, model_name: str = 'Model',
                                   save_path: Optional[str] = None):
        """
        Plot distribution of residuals.

        Parameters
        ----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the plot
        """
        residuals = y_true - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Histogram
        axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Residuals', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Residuals Distribution: {model_name}',
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot: {model_name}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")

        plt.show()

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            metric: str = 'rmse',
                            save_path: Optional[str] = None):
        """
        Plot comparison of multiple models.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            DataFrame with model comparison results
        metric : str
            Metric to compare ('rmse', 'mae', 'r2', 'mape')
        save_path : str, optional
            Path to save the plot
        """
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in comparison DataFrame")

        plt.figure(figsize=(12, 6))

        # Sort by metric
        if metric == 'r2':
            df_sorted = comparison_df.sort_values(metric, ascending=False)
        else:
            df_sorted = comparison_df.sort_values(metric)

        # Create bar plot
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(df_sorted))]
        plt.barh(df_sorted['model_name'], df_sorted[metric], color=colors, edgecolor='black')

        plt.xlabel(metric.upper(), fontsize=12)
        plt.ylabel('Model', fontsize=12)
        plt.title(f'Model Comparison: {metric.upper()}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            plt.text(row[metric], i, f"  {row[metric]:.2f}",
                    va='center', fontsize=10)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")

        plt.show()

    def plot_feature_importance(self, model: Any, feature_names: List[str],
                               model_name: str = 'Model', top_n: int = 20,
                               save_path: Optional[str] = None):
        """
        Plot feature importance for tree-based models.

        Parameters
        ----------
        model : Any
            Trained model with feature_importances_ attribute
        feature_names : List[str]
            Names of features
        model_name : str
            Name of the model
        top_n : int
            Number of top features to display
        save_path : str, optional
            Path to save the plot
        """
        # Get feature importances
        if hasattr(model, 'get_feature_importance'):
            importances = model.get_feature_importance()
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            importances = model.model.feature_importances_
        else:
            if self.verbose:
                print(f"⚠️  Model {model_name} does not have feature importances")
            return

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'],
                color='steelblue', edgecolor='black')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances: {model_name}',
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")

        plt.show()

        return importance_df

    def plot_learning_curves(self, train_sizes, train_scores, val_scores,
                           model_name: str = 'Model',
                           save_path: Optional[str] = None):
        """
        Plot learning curves showing model performance vs training size.

        Parameters
        ----------
        train_sizes : array-like
            Training set sizes
        train_scores : array-like
            Training scores for each size
        val_scores : array-like
            Validation scores for each size
        model_name : str
            Name of the model
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(10, 6))

        plt.plot(train_sizes, train_scores, 'o-', color='r',
                label='Training Score', linewidth=2, markersize=8)
        plt.plot(train_sizes, val_scores, 'o-', color='g',
                label='Validation Score', linewidth=2, markersize=8)

        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Learning Curves: {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"✅ Plot saved to: {save_path}")

        plt.show()

    def create_evaluation_report(self, comparison_df: pd.DataFrame,
                                 output_dir: str = 'docs'):
        """
        Create comprehensive evaluation report.

        Parameters
        ----------
        comparison_df : pd.DataFrame
            Model comparison results
        output_dir : str
            Directory to save report
        """
        output_path = Path(output_dir) / 'model_comparison.md'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"Total models evaluated: {len(comparison_df)}\n\n")

            # Best models by metric
            f.write("## Best Models by Metric\n\n")

            best_rmse = comparison_df.loc[comparison_df['rmse'].idxmin()]
            f.write(f"**Best RMSE**: {best_rmse['model_name']} "
                   f"({best_rmse['rmse']:.2f})\n\n")

            best_mae = comparison_df.loc[comparison_df['mae'].idxmin()]
            f.write(f"**Best MAE**: {best_mae['model_name']} "
                   f"({best_mae['mae']:.2f})\n\n")

            best_r2 = comparison_df.loc[comparison_df['r2'].idxmax()]
            f.write(f"**Best R²**: {best_r2['model_name']} "
                   f"({best_r2['r2']:.4f})\n\n")

            # Full comparison table
            f.write("## Detailed Model Comparison\n\n")
            f.write(comparison_df.to_markdown(index=False))
            f.write("\n\n")

            # Key insights
            f.write("## Key Insights\n\n")
            f.write(f"- RMSE range: {comparison_df['rmse'].min():.2f} - "
                   f"{comparison_df['rmse'].max():.2f}\n")
            f.write(f"- MAE range: {comparison_df['mae'].min():.2f} - "
                   f"{comparison_df['mae'].max():.2f}\n")
            f.write(f"- R² range: {comparison_df['r2'].min():.4f} - "
                   f"{comparison_df['r2'].max():.4f}\n")

        if self.verbose:
            print(f"✅ Evaluation report saved to: {output_path}")

        return output_path
