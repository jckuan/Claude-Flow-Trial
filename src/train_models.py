#!/usr/bin/env python3
"""
Main training script for Rossmann sales prediction models.
Orchestrates data loading, model training, evaluation, and saving.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from models.baseline import get_baseline_models
from models.linear_models import get_linear_models
from models.tree_models import get_tree_models
from models.ensemble_models import WeightedEnsemble, StackingEnsemble
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator


def load_processed_data():
    """
    Load processed training and validation data.
    This should load data that has been through feature engineering.
    """
    # TODO: Update this path once feature engineering is complete
    print("Loading processed data...")

    # For now, use placeholder - will be updated by FeatureEngineer
    train_path = Path(__file__).parent.parent / 'data' / 'train_processed.csv'
    val_path = Path(__file__).parent.parent / 'data' / 'val_processed.csv'

    if train_path.exists() and val_path.exists():
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        # Separate features and target
        target_col = 'Sales'
        feature_cols = [col for col in train_df.columns if col != target_col]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]

        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")

        return X_train, X_val, y_train, y_val, feature_cols
    else:
        raise FileNotFoundError(
            "Processed data files not found. "
            "Please run feature engineering first."
        )


def train_baseline_models(trainer, X_train, y_train, X_val, y_val):
    """Train baseline models."""
    print("\n" + "="*80)
    print("PHASE 1: BASELINE MODELS")
    print("="*80)

    baseline_models = get_baseline_models()
    results = trainer.train_multiple_models(
        baseline_models, X_train, y_train, X_val, y_val
    )

    return results, baseline_models


def train_linear_models(trainer, X_train, y_train, X_val, y_val):
    """Train linear models with regularization."""
    print("\n" + "="*80)
    print("PHASE 2: LINEAR MODELS")
    print("="*80)

    linear_models = get_linear_models()
    results = trainer.train_multiple_models(
        linear_models, X_train, y_train, X_val, y_val
    )

    return results, linear_models


def train_tree_models(trainer, X_train, y_train, X_val, y_val):
    """Train tree-based models."""
    print("\n" + "="*80)
    print("PHASE 3: TREE-BASED MODELS")
    print("="*80)

    tree_models = get_tree_models()
    results = trainer.train_multiple_models(
        tree_models, X_train, y_train, X_val, y_val
    )

    return results, tree_models


def train_ensemble_models(trainer, base_models, X_train, y_train, X_val, y_val):
    """Train ensemble models."""
    print("\n" + "="*80)
    print("PHASE 4: ENSEMBLE MODELS")
    print("="*80)

    # Select best models for ensembling
    best_models = list(base_models.values())[:5]  # Top 5 models

    # Weighted ensemble
    print("\nTraining Weighted Ensemble (uniform)...")
    weighted_uniform = WeightedEnsemble(
        models=best_models,
        weight_strategy='uniform'
    )
    trainer.train_single_model(
        weighted_uniform, X_train, y_train, X_val, y_val,
        model_name='WeightedEnsemble_Uniform'
    )

    # Stacking ensemble
    print("\nTraining Stacking Ensemble...")
    stacking = StackingEnsemble(
        base_models=best_models,
        cv=3
    )
    trainer.train_single_model(
        stacking, X_train, y_train, X_val, y_val,
        model_name='StackingEnsemble'
    )

    return {
        'weighted_uniform': weighted_uniform,
        'stacking': stacking
    }


def evaluate_and_save_results(evaluator, trainer, all_models,
                              X_val, y_val, feature_names):
    """Evaluate all models and save results."""
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)

    # Get training history
    history_df = trainer.get_training_history()

    # Find best model
    best_name, best_model = trainer.get_best_model(metric='val_rmse')

    if best_model is not None:
        print(f"\n🏆 Best Model: {best_name}")

        # Detailed evaluation of best model
        y_pred = best_model.predict(X_val)
        evaluator.calculate_metrics(y_val, y_pred, model_name=best_name)

        # Create visualizations for best model
        print("\nCreating visualizations...")
        viz_dir = Path(__file__).parent.parent / 'docs' / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Prediction plot
        evaluator.plot_predictions(
            np.array(y_val), y_pred,
            model_name=best_name,
            save_path=viz_dir / f'{best_name}_predictions.png'
        )

        # Residuals plot
        evaluator.plot_residuals_distribution(
            np.array(y_val), y_pred,
            model_name=best_name,
            save_path=viz_dir / f'{best_name}_residuals.png'
        )

        # Feature importance (if applicable)
        if hasattr(best_model, 'get_feature_importance') or \
           hasattr(best_model, 'feature_importances_'):
            evaluator.plot_feature_importance(
                best_model, feature_names,
                model_name=best_name, top_n=20,
                save_path=viz_dir / f'{best_name}_feature_importance.png'
            )

    # Model comparison plot
    if len(history_df) > 0:
        evaluator.plot_model_comparison(
            history_df, metric='val_rmse',
            save_path=viz_dir / 'model_comparison_rmse.png'
        )

    # Create evaluation report
    evaluator.create_evaluation_report(history_df)

    # Save best model
    if best_model is not None:
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / f'{best_name}_best.pkl'
        metadata = {
            'model_name': best_name,
            'val_rmse': history_df[history_df['model_name'] == best_name]['val_rmse'].values[0],
            'val_mae': history_df[history_df['model_name'] == best_name]['val_mae'].values[0],
            'val_r2': history_df[history_df['model_name'] == best_name]['val_r2'].values[0],
            'feature_names': feature_names
        }

        trainer.save_model(best_model, model_path, metadata=metadata)

    return best_name, best_model, history_df


def main():
    """Main training pipeline."""
    print("="*80)
    print("ROSSMANN SALES PREDICTION - MODEL TRAINING")
    print("="*80)

    # Initialize trainer and evaluator
    trainer = ModelTrainer(cv_strategy='timeseries', n_splits=5, verbose=True)
    evaluator = ModelEvaluator(verbose=True)

    try:
        # Load data
        X_train, X_val, y_train, y_val, feature_names = load_processed_data()

        # Train models in phases
        all_results = []
        all_models = {}

        # Phase 1: Baselines
        baseline_results, baseline_models = train_baseline_models(
            trainer, X_train, y_train, X_val, y_val
        )
        all_results.append(baseline_results)
        all_models.update(baseline_models)

        # Phase 2: Linear models
        linear_results, linear_models = train_linear_models(
            trainer, X_train, y_train, X_val, y_val
        )
        all_results.append(linear_results)
        all_models.update(linear_models)

        # Phase 3: Tree models
        tree_results, tree_models = train_tree_models(
            trainer, X_train, y_train, X_val, y_val
        )
        all_results.append(tree_results)
        all_models.update(tree_models)

        # Phase 4: Ensembles
        ensemble_models = train_ensemble_models(
            trainer, all_models, X_train, y_train, X_val, y_val
        )
        all_models.update(ensemble_models)

        # Final evaluation and saving
        best_name, best_model, final_results = evaluate_and_save_results(
            evaluator, trainer, all_models,
            X_val, y_val, feature_names
        )

        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print(f"✅ Best Model: {best_name}")
        print(f"✅ Results saved to: docs/")
        print(f"✅ Model saved to: models/")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure feature engineering has been completed first.")
        print("The processed data files should be in data/train_processed.csv")
        return 1

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
