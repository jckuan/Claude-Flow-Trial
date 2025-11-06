#!/usr/bin/env python3
"""
Test XGBoost and LightGBM after libomp installation
"""

import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("="*80)
print("TESTING XGBOOST & LIGHTGBM WITH LIBOMP")
print("="*80)

# Test imports first
print("\n1. Testing imports...")
try:
    import xgboost as xgb
    print("✓ XGBoost imported successfully")
    print(f"  Version: {xgb.__version__}")
except Exception as e:
    print(f"✗ XGBoost import failed: {e}")
    xgb = None

try:
    import lightgbm as lgb
    print("✓ LightGBM imported successfully")
    print(f"  Version: {lgb.__version__}")
except Exception as e:
    print(f"✗ LightGBM import failed: {e}")
    lgb = None

if not xgb and not lgb:
    print("\n❌ Both libraries failed to import. Please check installation.")
    exit(1)

# Load data
print("\n2. Loading processed data...")
train_df = pd.read_csv('data/processed/train_processed.csv')
val_df = pd.read_csv('data/processed/val_processed.csv')

print(f"   Training set: {train_df.shape}")
print(f"   Validation set: {val_df.shape}")

# Prepare features
X_train = train_df.drop('Sales', axis=1, errors='ignore')
y_train = train_df['Sales'] if 'Sales' in train_df.columns else None

X_val = val_df.drop('Sales', axis=1, errors='ignore')
y_val = val_df['Sales'] if 'Sales' in val_df.columns else None

if y_train is None or y_val is None:
    print("\n❌ Sales column not found in data")
    exit(1)

print(f"   Features: {X_train.shape[1]}")

# Test XGBoost
if xgb:
    print("\n" + "="*80)
    print("3. TESTING XGBOOST")
    print("="*80)
    
    try:
        print("\nTraining XGBoost model...")
        start_time = time.time()
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = xgb_model.predict(X_val)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"\n✓ XGBoost Training Complete!")
        print(f"  Training time: {training_time:.1f} seconds")
        print(f"\n  Validation Metrics:")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE:  {mae:.2f}")
        print(f"    R²:   {r2:.4f}")
        
        # Save model
        import joblib
        model_path = 'models/xgboost_test.pkl'
        joblib.dump(xgb_model, model_path)
        print(f"\n  ✓ Model saved to {model_path}")
        
    except Exception as e:
        print(f"\n✗ XGBoost training failed: {e}")
        import traceback
        traceback.print_exc()

# Test LightGBM
if lgb:
    print("\n" + "="*80)
    print("4. TESTING LIGHTGBM")
    print("="*80)
    
    try:
        print("\nTraining LightGBM model...")
        start_time = time.time()
        
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse'
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = lgb_model.predict(X_val)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"\n✓ LightGBM Training Complete!")
        print(f"  Training time: {training_time:.1f} seconds")
        print(f"\n  Validation Metrics:")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE:  {mae:.2f}")
        print(f"    R²:   {r2:.4f}")
        
        # Save model
        import joblib
        model_path = 'models/lightgbm_test.pkl'
        joblib.dump(lgb_model, model_path)
        print(f"\n  ✓ Model saved to {model_path}")
        
    except Exception as e:
        print(f"\n✗ LightGBM training failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("✅ TESTING COMPLETE")
print("="*80)
