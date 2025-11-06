# Rossmann Sales Prediction Models

## Overview

This module contains a comprehensive suite of machine learning models for predicting Rossmann store sales. It includes baseline models, regularized linear models, tree-based models, and ensemble methods.

## Module Structure

```
models/
├── __init__.py                 # Package initialization and exports
├── baseline.py                 # Baseline models (mean, median, linear)
├── linear_models.py           # Regularized linear models (Ridge, Lasso, ElasticNet)
├── tree_models.py             # Tree-based models (RF, XGBoost, LightGBM)
├── ensemble_models.py         # Ensemble strategies (Weighted, Stacking)
├── trainer.py                 # Training pipeline with CV and tuning
├── evaluator.py               # Evaluation metrics and visualizations
└── README.md                  # This file
```

## Quick Start

### Basic Usage

```python
from models import ModelTrainer, ModelEvaluator
from models.tree_models import XGBoostModel

# Initialize
trainer = ModelTrainer(cv_strategy='timeseries', n_splits=5)
evaluator = ModelEvaluator()

# Train a model
model = XGBoostModel(n_estimators=100, learning_rate=0.1)
result = trainer.train_single_model(model, X_train, y_train, X_val, y_val)

# Evaluate
metrics = evaluator.calculate_metrics(y_val, model.predict(X_val))
```

### Running Full Training Pipeline

```bash
cd src
python train_models.py
```

This will:
1. Load preprocessed data
2. Train all model categories
3. Generate comparison reports
4. Save best model to `models/` directory

## Model Categories

### 1. Baseline Models (`baseline.py`)

Simple models for establishing performance baselines:

- `MeanBaseline`: Predicts average sales
- `MedianBaseline`: Predicts median sales
- `SimpleLinearBaseline`: Basic linear regression
- `StoreAverageBaseline`: Store-specific averages
- `DayOfWeekBaseline`: Day-specific averages

**Usage:**
```python
from models.baseline import get_baseline_models

models = get_baseline_models()
mean_model = models['mean']
mean_model.fit(X_train, y_train)
predictions = mean_model.predict(X_test)
```

### 2. Linear Models (`linear_models.py`)

Regularized linear regression models:

- `LinearRegressionModel`: Standard OLS regression
- `RidgeModel`: L2 regularization (multiple α values)
- `LassoModel`: L1 regularization with feature selection
- `ElasticNetModel`: Combined L1/L2 regularization

**Usage:**
```python
from models.linear_models import RidgeModel

model = RidgeModel(alpha=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. Tree-Based Models (`tree_models.py`)

Ensemble tree methods for capturing non-linear patterns:

- `RandomForestModel`: Bagged decision trees
- `XGBoostModel`: Gradient boosting with regularization
- `LightGBMModel`: Fast gradient boosting

**Usage:**
```python
from models.tree_models import XGBoostModel

model = XGBoostModel(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05
)
model.fit(X_train, y_train,
         eval_set=[(X_val, y_val)],
         early_stopping_rounds=50)
predictions = model.predict(X_test)
```

### 4. Ensemble Models (`ensemble_models.py`)

Meta-learning strategies combining multiple models:

- `WeightedEnsemble`: Weighted average of predictions
- `StackingEnsemble`: Two-level stacking with meta-learner
- `BlendingEnsemble`: Simple holdout-based blending

**Usage:**
```python
from models.ensemble_models import StackingEnsemble
from models.tree_models import RandomForestModel, XGBoostModel
from models.linear_models import RidgeModel

# Create base models
base_models = [
    RandomForestModel(n_estimators=100),
    XGBoostModel(n_estimators=100),
    RidgeModel(alpha=1.0)
]

# Create stacking ensemble
stacker = StackingEnsemble(
    base_models=base_models,
    cv=5,
    use_original_features=True
)

stacker.fit(X_train, y_train)
predictions = stacker.predict(X_test)
```

## ModelTrainer (`trainer.py`)

Comprehensive training pipeline with cross-validation and hyperparameter tuning.

### Features

- **Time-series aware CV**: Respects temporal order
- **Multiple model training**: Batch training with comparison
- **Hyperparameter tuning**: Grid and random search
- **Model persistence**: Save/load models with metadata
- **Experiment tracking**: Training history and metrics

### Key Methods

```python
trainer = ModelTrainer(cv_strategy='timeseries', n_splits=5)

# Train single model
result = trainer.train_single_model(
    model, X_train, y_train, X_val, y_val
)

# Train multiple models
models = {'rf': RandomForestModel(), 'xgb': XGBoostModel()}
results_df = trainer.train_multiple_models(
    models, X_train, y_train, X_val, y_val
)

# Cross-validation
cv_results = trainer.cross_validate_model(model, X, y)

# Hyperparameter tuning
tuning_results = trainer.tune_hyperparameters(
    model, X, y, param_grid, search_type='grid'
)

# Save/load model
trainer.save_model(model, 'models/best_model.pkl', metadata)
model, metadata = trainer.load_model('models/best_model.pkl')

# Get best model
best_name, best_model = trainer.get_best_model(metric='val_rmse')
```

## ModelEvaluator (`evaluator.py`)

Comprehensive evaluation with metrics and visualizations.

### Features

- **Multiple metrics**: RMSE, MAE, MAPE, R²
- **Visualizations**: Predictions, residuals, comparisons
- **Feature importance**: For tree-based models
- **Evaluation reports**: Markdown documentation

### Key Methods

```python
evaluator = ModelEvaluator()

# Calculate metrics
metrics = evaluator.calculate_metrics(y_true, y_pred, model_name='XGBoost')

# Evaluate multiple models
comparison_df = evaluator.evaluate_multiple_models(
    models, X_test, y_test
)

# Create visualizations
evaluator.plot_predictions(y_true, y_pred,
                          save_path='docs/predictions.png')

evaluator.plot_residuals_distribution(y_true, y_pred,
                                     save_path='docs/residuals.png')

evaluator.plot_model_comparison(comparison_df, metric='rmse',
                               save_path='docs/comparison.png')

evaluator.plot_feature_importance(model, feature_names,
                                 save_path='docs/importance.png')

# Generate report
evaluator.create_evaluation_report(comparison_df, output_dir='docs')
```

## Performance Expectations

| Model Type | Expected RMSE | Training Time | Memory |
|------------|---------------|---------------|--------|
| Baseline | 2000-3000 | < 1 min | Low |
| Linear | 1500-2000 | 1-5 min | Low |
| Random Forest | 1200-1500 | 10-30 min | Medium |
| XGBoost | 1000-1200 | 5-20 min | Medium |
| LightGBM | 1000-1200 | 3-15 min | Low |
| Stacking | 900-1100 | 20-60 min | High |

## Hyperparameter Tuning Grids

### Random Forest
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

### XGBoost
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}
```

### LightGBM
```python
{
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 50, 100]
}
```

## Best Practices

1. **Always start with baselines** to establish minimum performance
2. **Use time-series CV** to prevent data leakage
3. **Monitor train/val gap** to detect overfitting
4. **Feature importance analysis** to understand models
5. **Ensemble diverse models** for best results
6. **Save metadata** with models for reproducibility

## Common Workflows

### Train and Compare All Models

```python
from models import *
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator

# Initialize
trainer = ModelTrainer()
evaluator = ModelEvaluator()

# Get all models
all_models = {}
all_models.update(get_baseline_models())
all_models.update(get_linear_models())
all_models.update(get_tree_models())

# Train and compare
results_df = trainer.train_multiple_models(
    all_models, X_train, y_train, X_val, y_val
)

# Evaluate and visualize
evaluator.plot_model_comparison(results_df, metric='val_rmse')
evaluator.create_evaluation_report(results_df)
```

### Hyperparameter Tuning

```python
from models.tree_models import XGBoostModel, get_hyperparameter_grid_xgb
from models.trainer import ModelTrainer

trainer = ModelTrainer()
model = XGBoostModel()
param_grid = get_hyperparameter_grid_xgb()

results = trainer.tune_hyperparameters(
    model, X_train, y_train,
    param_grid=param_grid,
    search_type='random',
    n_iter=50
)

best_model = results['best_model']
```

### Create Custom Ensemble

```python
from models.ensemble_models import StackingEnsemble
from models.tree_models import *
from models.linear_models import RidgeModel

# Select best base models
base_models = [
    XGBoostModel(n_estimators=200, max_depth=6),
    LightGBMModel(n_estimators=200, max_depth=8),
    RandomForestModel(n_estimators=200),
    RidgeModel(alpha=1.0)
]

# Create stacking ensemble
ensemble = StackingEnsemble(
    base_models=base_models,
    meta_model=RidgeModel(alpha=0.1),
    cv=5
)

ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
```

## Dependencies

See `requirements.txt` for full list:
- `scikit-learn>=1.0.0` - Core ML algorithms
- `xgboost>=1.5.0` - Gradient boosting
- `lightgbm>=3.3.0` - Fast gradient boosting
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `matplotlib>=3.4.0` - Visualization
- `seaborn>=0.11.0` - Statistical visualization

## Troubleshooting

### XGBoost/LightGBM Not Available
```bash
pip install xgboost lightgbm
```

### Memory Issues with Large Datasets
- Use LightGBM instead of XGBoost
- Reduce number of trees (n_estimators)
- Use smaller max_depth
- Sample data for initial experiments

### Slow Training
- Reduce n_splits for CV
- Use RandomizedSearchCV instead of GridSearchCV
- Enable n_jobs=-1 for parallel processing
- Use LightGBM for faster training

## Next Steps

After model training:
1. Evaluate on test set
2. Perform error analysis
3. Generate predictions for submission
4. Deploy best model to production
5. Monitor model performance

## References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
