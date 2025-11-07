#!/usr/bin/env python3
"""
Generate visualizations for the sales analysis report.

Creates:
- Predictions vs Actual scatter plot
- Residual distribution plot
- Feature importance chart (top 20 features)
- Time series comparison plot
- Error distribution by store type

Saves all figures to docs/figures/
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def ensure_figures_dir():
    Path('docs/figures').mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    """Load the trained model and validation data"""
    # Try to load XGBoost model first, then RandomForest
    model_path = None
    for candidate in ['models/full_pipeline_model_xgboost_deeptrees.pkl', 
                      'models/full_pipeline_model_randomforest.pkl',
                      'models/xgboost_deeptrees.pkl',
                      'models/random_forest_best.pkl']:
        if Path(candidate).exists():
            model_path = candidate
            break
    
    if model_path is None:
        raise FileNotFoundError("No trained model found. Run pipeline first.")
    
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load validation data
    val_df = pd.read_csv('data/processed/val_processed.csv')
    
    # Also try to load original data with Store column
    try:
        # Read from rossmann data if available
        train_orig = pd.read_csv('data/rossmann-store-sales/train.csv')
        store_info = pd.read_csv('data/rossmann-store-sales/store.csv')
    except:
        train_orig = None
        store_info = None
    
    return model, val_df, train_orig, store_info


def plot_predictions_vs_actual(y_true, y_pred, save_path='docs/figures/predictions_vs_actual.png'):
    """Scatter plot of predictions vs actual values"""
    plt.figure(figsize=(10, 8))
    
    # Sample if too many points
    if len(y_true) > 10000:
        idx = np.random.choice(len(y_true), 10000, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    plt.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)
    
    # Perfect prediction line
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Sales ($)', fontsize=12)
    plt.ylabel('Predicted Sales ($)', fontsize=12)
    plt.title('Predictions vs Actual Sales (Validation Set)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_residuals(y_true, y_pred, save_path='docs/figures/residual_distribution.png'):
    """Plot residual distribution"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Residuals ($)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Residual Distribution', fontsize=13, fontweight='bold')
    axes[0].legend()
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_feature_importance(model, feature_names, top_n=20, save_path='docs/figures/feature_importance.png'):
    """Plot top N most important features"""
    # Get feature importances
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print("Model doesn't have feature_importances_. Skipping.")
            return
    except:
        print("Unable to extract feature importances. Skipping.")
        return
    
    # Create dataframe
    feat_imp_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feat_imp_df)), feat_imp_df['importance'])
    plt.yticks(range(len(feat_imp_df)), feat_imp_df['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_error_by_magnitude(y_true, y_pred, save_path='docs/figures/error_by_magnitude.png'):
    """Plot error distribution by sales magnitude"""
    errors = np.abs(y_true - y_pred)
    
    # Create bins
    bins = [0, 2000, 4000, 6000, 8000, 10000, 15000, np.inf]
    labels = ['0-2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', '10k-15k', '15k+']
    
    df = pd.DataFrame({
        'actual': y_true,
        'error': errors,
        'sales_bin': pd.cut(y_true, bins=bins, labels=labels)
    })
    
    plt.figure(figsize=(12, 6))
    df.boxplot(column='error', by='sales_bin', ax=plt.gca())
    plt.xlabel('Actual Sales Range ($)', fontsize=12)
    plt.ylabel('Absolute Error ($)', fontsize=12)
    plt.title('Prediction Error by Sales Magnitude', fontsize=14, fontweight='bold')
    plt.suptitle('')  # Remove default title
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def plot_time_series_sample(val_df, y_pred, save_path='docs/figures/time_series_sample.png'):
    """Plot time series comparison for a sample of stores"""
    # Check if Date column exists
    if 'Date' not in val_df.columns:
        print("Date column not found. Skipping time series plot.")
        return
    
    df = val_df.copy()
    df['Predicted'] = y_pred
    
    # Convert Date to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Sample 3 stores if Store column exists
    if 'Store' in df.columns:
        sample_stores = df['Store'].unique()[:3]
    else:
        print("Store column not found. Using overall time series.")
        # Aggregate by date
        df_agg = df.groupby('Date').agg({'Sales': 'mean', 'Predicted': 'mean'}).reset_index()
        
        plt.figure(figsize=(14, 6))
        plt.plot(df_agg['Date'], df_agg['Sales'], label='Actual', linewidth=2)
        plt.plot(df_agg['Date'], df_agg['Predicted'], label='Predicted', linewidth=2, alpha=0.7)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Average Daily Sales ($)', fontsize=12)
        plt.title('Time Series: Actual vs Predicted Sales (All Stores Average)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved {save_path}")
        return
    
    # Plot for sample stores
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for i, store in enumerate(sample_stores):
        store_data = df[df['Store'] == store].sort_values('Date')
        axes[i].plot(store_data['Date'], store_data['Sales'], label='Actual', linewidth=2)
        axes[i].plot(store_data['Date'], store_data['Predicted'], label='Predicted', linewidth=2, alpha=0.7)
        axes[i].set_title(f'Store {store}', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Sales ($)', fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date', fontsize=12)
    plt.suptitle('Time Series Comparison: Sample Stores', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {save_path}")


def main():
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS FOR SALES ANALYSIS REPORT")
    print("="*80)
    
    ensure_figures_dir()
    
    # Load model and data
    try:
        model, val_df, train_orig, store_info = load_model_and_data()
    except Exception as e:
        print(f"Error loading model/data: {e}")
        print("Please run the pipeline first: python scripts/run_full_pipeline.py")
        return 1
    
    # Prepare data
    X_val = val_df.drop('Sales', axis=1)
    y_val = val_df['Sales'].values
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_val)
    
    # Get feature names (exclude target and identifiers)
    feature_names = [col for col in X_val.columns if col not in ['Date', 'Store', 'Customers']]
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    plot_predictions_vs_actual(y_val, y_pred)
    plot_residuals(y_val, y_pred)
    plot_feature_importance(model, feature_names)
    plot_error_by_magnitude(y_val, y_pred)
    plot_time_series_sample(val_df, y_pred)
    
    print("\n" + "="*80)
    print("✅ All visualizations generated successfully!")
    print("Figures saved to: docs/figures/")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
