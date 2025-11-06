#!/usr/bin/env python3
"""
XGBoost Hyperparameter Tuning with RMSPE Metric
"""

import pandas as pd
import numpy as np
import time
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error (RMSPE)
    
    RMSPE = sqrt(mean((y_true - y_pred)^2 / y_true^2))
    
    Note: Excludes zeros to avoid division by zero
    """
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

def xgb_rmspe_obj(y_pred, dtrain):
    """Custom XGBoost objective function for RMSPE"""
    y_true = dtrain.get_label()
    mask = y_true != 0
    
    grad = np.zeros_like(y_pred)
    hess = np.zeros_like(y_pred)
    
    grad[mask] = -2 * (y_true[mask] - y_pred[mask]) / (y_true[mask] ** 2)
    hess[mask] = 2 / (y_true[mask] ** 2)
    
    return grad, hess

def evaluate_model(model, X_val, y_val, model_name="Model"):
    """Evaluate model with multiple metrics including RMSPE"""
    y_pred = model.predict(X_val)
    
    # Ensure non-negative predictions
    y_pred = np.maximum(y_pred, 0)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    rmspe_val = rmspe(y_val, y_pred)
    
    return {
        'model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'RMSPE': rmspe_val
    }

print("="*80)
print("XGBOOST HYPERPARAMETER TUNING WITH RMSPE")
print("="*80)

# Load data
print("\n1. Loading processed data...")
train_df = pd.read_csv('data/processed/train_processed.csv')
val_df = pd.read_csv('data/processed/val_processed.csv')

X_train = train_df.drop('Sales', axis=1, errors='ignore')
y_train = train_df['Sales']
X_val = val_df.drop('Sales', axis=1, errors='ignore')
y_val = val_df['Sales']

print(f"   Training: {X_train.shape}, Validation: {X_val.shape}")

# Define hyperparameter configurations to test
configs = [
    {
        'name': 'XGBoost_Baseline',
        'params': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'XGBoost_DeepTrees',
        'params': {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'XGBoost_MoreTrees',
        'params': {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    },
    {
        'name': 'XGBoost_Regularized',
        'params': {
            'n_estimators': 150,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5
        }
    },
    {
        'name': 'XGBoost_Aggressive',
        'params': {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1
        }
    }
]

# Train and evaluate each configuration
results = []

print("\n2. Training XGBoost variants...")
print("="*80)

for i, config in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Training {config['name']}...")
    print(f"   Parameters: {config['params']}")
    
    start_time = time.time()
    
    model = xgb.XGBRegressor(
        **config['params'],
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    training_time = time.time() - start_time
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val, config['name'])
    metrics['training_time'] = training_time
    results.append(metrics)
    
    print(f"   ✓ Training time: {training_time:.1f}s")
    print(f"   Metrics:")
    print(f"     RMSPE: {metrics['RMSPE']:.6f} ⭐")
    print(f"     RMSE:  {metrics['RMSE']:.2f}")
    print(f"     MAE:   {metrics['MAE']:.2f}")
    print(f"     R²:    {metrics['R²']:.4f}")
    
    # Save model
    model_path = f"models/{config['name'].lower()}.pkl"
    joblib.dump(model, model_path)
    print(f"   ✓ Saved to {model_path}")

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSPE')

print("\n" + "="*80)
print("3. RESULTS SUMMARY (Sorted by RMSPE)")
print("="*80)
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('models/xgboost_tuning_results.csv', index=False)
print(f"\n✓ Results saved to models/xgboost_tuning_results.csv")

# Identify best model
best_model = results_df.iloc[0]
print("\n" + "="*80)
print("🏆 BEST MODEL")
print("="*80)
print(f"Model: {best_model['model']}")
print(f"RMSPE: {best_model['RMSPE']:.6f} ⭐")
print(f"RMSE:  {best_model['RMSE']:.2f}")
print(f"MAE:   {best_model['MAE']:.2f}")
print(f"R²:    {best_model['R²']:.4f}")
print(f"Training Time: {best_model['training_time']:.1f}s")

print("\n" + "="*80)
print("✅ HYPERPARAMETER TUNING COMPLETE")
print("="*80)
