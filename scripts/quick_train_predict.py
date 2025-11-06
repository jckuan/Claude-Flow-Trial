#!/usr/bin/env python3
"""
Quick model training and prediction script for Rossmann sales.
Trains Random Forest and generates submission file.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("ROSSMANN SALES - QUICK TRAIN & PREDICT")
    print("="*80)
    
    # Load processed data
    print("\nLoading data...")
    train_df = pd.read_csv('data/processed/train_processed.csv')
    val_df = pd.read_csv('data/processed/val_processed.csv')
    test_df = pd.read_csv('data/processed/test_processed.csv')
    
    # Separate features and target
    X_train = train_df.drop('Sales', axis=1)
    y_train = train_df['Sales']
    X_val = val_df.drop('Sales', axis=1)
    y_val = val_df['Sales']
    X_test = test_df.drop('Sales', axis=1, errors='ignore')
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train Random Forest
    print("\nTraining Random Forest (200 trees)...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating on validation set...")
    y_val_pred = model.predict(X_val)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = 'models/random_forest_best.pkl'
    Path('models').mkdir(exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✓ Model saved to {model_path}")
    
    # Generate predictions for test set
    print("\nGenerating test predictions...")
    test_predictions = model.predict(X_test)
    
    # Load original test file for IDs
    original_test = pd.read_csv('data/rossmann-store-sales/test.csv')
    
    print(f"Test predictions: {len(test_predictions)}")
    print(f"Original test IDs: {len(original_test)}")
    
    # If lengths don't match, we need to match by row indices
    # The processed test data should correspond to the original test rows
    if len(test_predictions) != len(original_test):
        print(f"⚠️  Warning: Length mismatch. Using first {min(len(test_predictions), len(original_test))} predictions")
        min_len = min(len(test_predictions), len(original_test))
        test_predictions = test_predictions[:min_len]
        original_test = original_test.iloc[:min_len]
    
    # Create submission
    submission = pd.DataFrame({
        'Id': original_test['Id'].values,
        'Sales': np.clip(test_predictions, 0, None)  # Ensure non-negative
    })
    
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission saved to {submission_path}")
    print(f"\nSubmission Statistics:")
    print(f"  Number of predictions: {len(submission)}")
    print(f"  Mean predicted sales: ${submission['Sales'].mean():.2f}")
    print(f"  Median predicted sales: ${submission['Sales'].median():.2f}")
    print(f"  Min: ${submission['Sales'].min():.2f}")
    print(f"  Max: ${submission['Sales'].max():.2f}")
    
    # Show sample predictions
    print("\nSample predictions (first 10):")
    print(submission.head(10))
    
    print("\n" + "="*80)
    print("✅ COMPLETE! Model trained and predictions generated.")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
