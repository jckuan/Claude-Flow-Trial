#!/usr/bin/env python3
"""
Create Ensemble Model combining XGBoost, LightGBM, and Random Forest
Optimized for RMSPE metric
"""

import pandas as pd
import numpy as np
import time
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmspe(y_true, y_pred):
    """Calculate Root Mean Square Percentage Error (RMSPE)"""
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Evaluate predictions with all metrics"""
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    return {
        'model': model_name,
        'RMSPE': rmspe(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }

print("="*80)
print("ENSEMBLE MODEL CREATION WITH RMSPE OPTIMIZATION")
print("="*80)

# Load validation data
print("\n1. Loading validation data...")
val_df = pd.read_csv('data/processed/val_processed.csv')
X_val = val_df.drop('Sales', axis=1, errors='ignore')
y_val = val_df['Sales']
print(f"   Validation set: {X_val.shape}")

# Load all trained models
print("\n2. Loading trained models...")
models = {}

model_files = {
    'XGBoost_DeepTrees': 'models/xgboost_deeptrees.pkl',
    'XGBoost_Aggressive': 'models/xgboost_aggressive.pkl',
    'XGBoost_Regularized': 'models/xgboost_regularized.pkl',
    'Random_Forest': 'models/random_forest_best.pkl',
    'LightGBM': 'models/lightgbm_test.pkl'
}

for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
        print(f"   ✓ Loaded {name}")
    except FileNotFoundError:
        print(f"   ✗ {name} not found at {path}")

if len(models) == 0:
    print("\n❌ No models loaded. Please train models first.")
    exit(1)

# Generate predictions from each model
print("\n3. Generating predictions from each model...")
predictions = {}
individual_results = []

for name, model in models.items():
    pred = model.predict(X_val)
    predictions[name] = pred
    
    metrics = evaluate_predictions(y_val, pred, name)
    individual_results.append(metrics)
    
    print(f"   {name}:")
    print(f"      RMSPE: {metrics['RMSPE']:.6f}")
    print(f"      RMSE:  {metrics['RMSE']:.2f}")

# Create ensemble strategies
print("\n4. Creating ensemble strategies...")
print("="*80)

ensemble_strategies = [
    {
        'name': 'Ensemble_Uniform',
        'weights': {name: 1.0/len(models) for name in models.keys()},
        'description': 'Equal weights for all models'
    },
    {
        'name': 'Ensemble_TopThree',
        'weights': {
            'XGBoost_DeepTrees': 0.5,
            'XGBoost_Aggressive': 0.3,
            'XGBoost_Regularized': 0.2
        },
        'description': 'Top 3 XGBoost models'
    },
    {
        'name': 'Ensemble_BestHeavy',
        'weights': {
            'XGBoost_DeepTrees': 0.6,
            'XGBoost_Aggressive': 0.2,
            'Random_Forest': 0.1,
            'LightGBM': 0.1
        },
        'description': '60% best model, diversified rest'
    },
    {
        'name': 'Ensemble_Diversified',
        'weights': {
            'XGBoost_DeepTrees': 0.4,
            'XGBoost_Aggressive': 0.25,
            'XGBoost_Regularized': 0.15,
            'Random_Forest': 0.1,
            'LightGBM': 0.1
        },
        'description': 'Balanced across all models'
    },
    {
        'name': 'Ensemble_Conservative',
        'weights': {
            'XGBoost_DeepTrees': 0.35,
            'XGBoost_Aggressive': 0.20,
            'XGBoost_Regularized': 0.30,
            'Random_Forest': 0.15
        },
        'description': 'Higher weight on regularized models'
    }
]

# Evaluate each ensemble strategy
ensemble_results = []

for strategy in ensemble_strategies:
    print(f"\n{strategy['name']}:")
    print(f"   Description: {strategy['description']}")
    print(f"   Weights: {strategy['weights']}")
    
    # Create weighted ensemble prediction
    ensemble_pred = np.zeros(len(y_val))
    for name, weight in strategy['weights'].items():
        if name in predictions:
            ensemble_pred += weight * predictions[name]
    
    # Evaluate
    metrics = evaluate_predictions(y_val, ensemble_pred, strategy['name'])
    ensemble_results.append(metrics)
    
    print(f"   Results:")
    print(f"      RMSPE: {metrics['RMSPE']:.6f} ⭐")
    print(f"      RMSE:  {metrics['RMSE']:.2f}")
    print(f"      MAE:   {metrics['MAE']:.2f}")
    print(f"      R²:    {metrics['R²']:.4f}")
    
    # Save ensemble predictions for later use
    strategy['predictions'] = ensemble_pred

# Combine all results
print("\n" + "="*80)
print("5. COMPLETE RESULTS COMPARISON")
print("="*80)

all_results = individual_results + ensemble_results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('RMSPE')

print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv('models/ensemble_comparison_results.csv', index=False)
print(f"\n✓ Results saved to models/ensemble_comparison_results.csv")

# Identify best overall model
best_result = results_df.iloc[0]
print("\n" + "="*80)
print("🏆 BEST OVERALL MODEL")
print("="*80)
print(f"Model: {best_result['model']}")
print(f"RMSPE: {best_result['RMSPE']:.6f} ⭐")
print(f"RMSE:  {best_result['RMSE']:.2f}")
print(f"MAE:   {best_result['MAE']:.2f}")
print(f"R²:    {best_result['R²']:.4f}")

# Save best ensemble model if it's an ensemble
if best_result['model'].startswith('Ensemble_'):
    best_strategy = [s for s in ensemble_strategies if s['name'] == best_result['model']][0]
    
    # Create ensemble model class
    class WeightedEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
            
        def predict(self, X):
            pred = np.zeros(len(X))
            for name, weight in self.weights.items():
                if name in self.models:
                    pred += weight * self.models[name].predict(X)
            return pred
    
    ensemble_model = WeightedEnsemble(models, best_strategy['weights'])
    ensemble_path = f"models/{best_result['model'].lower()}.pkl"
    joblib.dump(ensemble_model, ensemble_path)
    print(f"\n✓ Best ensemble saved to {ensemble_path}")

print("\n" + "="*80)
print("✅ ENSEMBLE MODELING COMPLETE")
print("="*80)
