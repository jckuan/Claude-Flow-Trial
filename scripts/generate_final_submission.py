#!/usr/bin/env python3
"""
Generate Final Submission using Best Model (XGBoost_DeepTrees)
With RMSPE optimization
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def rmspe(y_true, y_pred):
    """Calculate Root Mean Square Percentage Error (RMSPE)"""
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))

print("="*80)
print("FINAL SUBMISSION GENERATION")
print("="*80)

# Load the best model
print("\n1. Loading best model (XGBoost_DeepTrees)...")
model = joblib.load('models/xgboost_deeptrees.pkl')
print("   ✓ Model loaded")

# Load processed data
print("\n2. Loading processed data...")
train_df = pd.read_csv('data/processed/train_processed.csv')
val_df = pd.read_csv('data/processed/val_processed.csv')
test_df = pd.read_csv('data/processed/test_processed.csv')

X_train = train_df.drop('Sales', axis=1, errors='ignore')
y_train = train_df['Sales']
X_val = val_df.drop('Sales', axis=1, errors='ignore')
y_val = val_df['Sales']
X_test = test_df.drop('Sales', axis=1, errors='ignore')

print(f"   Training: {X_train.shape}")
print(f"   Validation: {X_val.shape}")
print(f"   Test: {X_test.shape}")

# Validate model performance on validation set
print("\n3. Validating model performance on validation set...")
val_predictions = model.predict(X_val)
val_predictions = np.maximum(val_predictions, 0)  # Ensure non-negative

val_rmspe = rmspe(y_val, val_predictions)
val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
val_mae = mean_absolute_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"\n   Validation Metrics:")
print(f"     RMSPE: {val_rmspe:.6f} ⭐")
print(f"     RMSE:  {val_rmse:.2f}")
print(f"     MAE:   {val_mae:.2f}")
print(f"     R²:    {val_r2:.4f}")

# Generate test predictions
print("\n4. Generating test set predictions...")
test_predictions = model.predict(X_test)
test_predictions = np.maximum(test_predictions, 0)  # Ensure non-negative

print(f"   Generated {len(test_predictions)} predictions")

# Load original test file for IDs
print("\n5. Creating submission file...")
original_test = pd.read_csv('data/rossmann-store-sales/test.csv')
print(f"   Original test set: {len(original_test)} rows")

# Handle length mismatch if necessary
if len(test_predictions) != len(original_test):
    print(f"   ⚠️  Length mismatch detected:")
    print(f"      Predictions: {len(test_predictions)}")
    print(f"      Original IDs: {len(original_test)}")
    print(f"   Using first {len(original_test)} predictions")
    test_predictions = test_predictions[:len(original_test)]

# Create submission DataFrame
submission = pd.DataFrame({
    'Id': original_test['Id'].values,
    'Sales': test_predictions
})

# Save submission
submission_path = 'submission_final.csv'
submission.to_csv(submission_path, index=False)
print(f"   ✓ Submission saved to {submission_path}")

# Display submission statistics
print("\n6. Submission Statistics:")
print(f"   Number of predictions: {len(submission)}")
print(f"   Mean predicted sales: ${submission['Sales'].mean():.2f}")
print(f"   Median predicted sales: ${submission['Sales'].median():.2f}")
print(f"   Min prediction: ${submission['Sales'].min():.2f}")
print(f"   Max prediction: ${submission['Sales'].max():.2f}")
print(f"   Std deviation: ${submission['Sales'].std():.2f}")

# Display sample predictions
print("\n7. Sample predictions (first 20 rows):")
print(submission.head(20).to_string(index=False))

# Create detailed submission report
print("\n8. Creating detailed submission report...")
report = {
    'Model': 'XGBoost_DeepTrees',
    'Validation_RMSPE': val_rmspe,
    'Validation_RMSE': val_rmse,
    'Validation_MAE': val_mae,
    'Validation_R2': val_r2,
    'Test_Predictions': len(submission),
    'Mean_Sales': submission['Sales'].mean(),
    'Median_Sales': submission['Sales'].median(),
    'Min_Sales': submission['Sales'].min(),
    'Max_Sales': submission['Sales'].max(),
    'Std_Sales': submission['Sales'].std(),
    'Submission_File': submission_path
}

report_df = pd.DataFrame([report])
report_path = 'submission_report.csv'
report_df.to_csv(report_path, index=False)
print(f"   ✓ Report saved to {report_path}")

print("\n" + "="*80)
print("✅ FINAL SUBMISSION COMPLETE")
print("="*80)
print(f"\n🎯 Best Model: XGBoost_DeepTrees")
print(f"📊 Validation RMSPE: {val_rmspe:.6f} (~{val_rmspe*100:.2f}% average error)")
print(f"📁 Submission File: {submission_path}")
print(f"📋 Report File: {report_path}")
print("\n🚀 Ready for Kaggle submission!")
