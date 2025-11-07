#!/usr/bin/env python3
"""
Run the full automated pipeline: feature engineering -> train -> evaluate -> generate submission

This script programmatically runs the feature pipeline in `src.features.pipeline`,
trains a default model (RandomForest) on the time-based train/validation split,
computes evaluation metrics including MAPE and RMSPE, saves the model, and writes
the final submission CSV.

Usage:
    python scripts/run_full_pipeline.py

Outputs:
    - data/processed/train_processed.csv, val_processed.csv, test_processed.csv
    - models/full_pipeline_model.pkl
    - results/submission_final.csv
    - printed evaluation metrics including MAPE
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.pipeline import create_features
from src.evaluation.metrics import evaluate_predictions, mape, rmspe

from sklearn.ensemble import RandomForestRegressor

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will use RandomForest as fallback")


def ensure_dirs():
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(parents=True, exist_ok=True)


def train_and_evaluate():
    ensure_dirs()

    # 1. Run feature engineering and create processed datasets
    print("\n=== Running feature engineering to create processed datasets ===")
    create_features(save_path='data/processed')

    # 2. Load processed datasets
    print("\n=== Loading processed data ===")
    train_df = pd.read_csv('data/processed/train_processed.csv')
    val_df = pd.read_csv('data/processed/val_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv')

    # Ensure Sales is present
    if 'Sales' not in train_df.columns:
        raise RuntimeError('Processed training data must contain Sales target')

    X_train = train_df.drop('Sales', axis=1)
    y_train = train_df['Sales']
    X_val = val_df.drop('Sales', axis=1)
    y_val = val_df['Sales']

    print(f"Training rows: {len(X_train)}, Validation rows: {len(X_val)}, Test rows: {len(test_df)}")

    # 3. Train model - XGBoost_DeepTrees config or RandomForest fallback
    if XGBOOST_AVAILABLE:
        print("\n=== Training XGBoost_DeepTrees model ===")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0,
            reg_lambda=1,
            random_state=42,
            tree_method='hist',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        model_name = "XGBoost_DeepTrees"
    else:
        print("\n=== Training RandomForest model (XGBoost not available) ===")
        model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        model_name = "RandomForest"

    # 4. Evaluate on validation set using multiple metrics including MAPE
    print("\n=== Evaluating on validation set ===")
    val_pred = model.predict(X_val)
    results = evaluate_predictions(y_val.values, val_pred, metrics=['rmspe', 'rmse', 'mae', 'mape', 'r2'])

    print("\nValidation Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k.upper():6}: {v:.6f}")
        else:
            print(f"  {k.upper():6}: {v}")

    # 5. Save model
    model_path = f'models/full_pipeline_model_{model_name.lower()}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model ({model_name}) saved to {model_path}")

    # 6. Generate predictions for test set and save submission
    print("\n=== Generating submission ===")
    # Align test features with training features if necessary
    test_X = test_df.drop('Sales', axis=1, errors='ignore')
    test_pred = model.predict(test_X)

    # Load original test file for IDs
    original_test = pd.read_csv('data/rossmann-store-sales/test.csv')

    # Create submission DataFrame
    submission = pd.DataFrame({
        'Id': original_test['Id'].values[: len(test_pred)],
        'Sales': np.clip(test_pred, 0, None)
    })

    submission_path = 'results/submission_final.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\n✓ Submission saved to {submission_path}")

    # 7. Print a short summary
    print("\nSubmission statistics:")
    print(f"  Rows: {len(submission)}")
    print(f"  Mean prediction: {submission['Sales'].mean():.2f}")
    print(f"  Median prediction: {submission['Sales'].median():.2f}")

    # Save metrics to results/metrics.json for downstream reporting
    try:
        import json
        metrics_path = 'results/metrics.json'
        metrics_output = {
            'model': model_name,
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(test_df),
            **{k: float(v) if (isinstance(v, (int, float, np.floating, np.integer))) else v for k, v in results.items()}
        }
        with open(metrics_path, 'w') as mf:
            json.dump(metrics_output, mf, indent=2)
        print(f"\n✓ Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"⚠️  Unable to save metrics.json: {e}")

    return results, model, model_name, y_val, val_pred


if __name__ == '__main__':
    results, model, model_name, y_val, val_pred = train_and_evaluate()
    print('\nFull pipeline complete.')
