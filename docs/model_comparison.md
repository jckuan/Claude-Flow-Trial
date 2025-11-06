# Model Architecture and Training Strategy

## Overview

This document describes the comprehensive model selection and training strategy for Rossmann sales prediction.

## Model Categories

### 1. Baseline Models

**Purpose**: Establish minimum performance thresholds

- **Mean Baseline**: Predicts average sales across all data
- **Median Baseline**: Predicts median sales (robust to outliers)
- **Simple Linear Regression**: Basic linear model without regularization
- **Store Average Baseline**: Store-specific average sales
- **Day of Week Baseline**: Day-specific average sales

**Expected Performance**: RMSE ~2000-3000

### 2. Linear Models with Regularization

**Purpose**: Handle multicollinearity and prevent overfitting

- **Ridge Regression**: L2 regularization (α = 0.1, 1.0, 10.0, 100.0)
- **Lasso Regression**: L1 regularization with feature selection (α = 0.1, 1.0, 10.0)
- **ElasticNet**: Combined L1/L2 regularization (l1_ratio = 0.5, 0.7, 0.9)

**Advantages**:
- Fast training time
- Interpretable coefficients
- Good for linear relationships
- Built-in feature selection (Lasso)

**Expected Performance**: RMSE ~1500-2000

### 3. Tree-Based Models

**Purpose**: Capture non-linear patterns and interactions

#### Random Forest
- Ensemble of decision trees with bagging
- Configurations: 100, 200 trees with depth 20
- Robust to outliers
- Parallel training
- **Expected Performance**: RMSE ~1200-1500

#### XGBoost (if installed)
- Gradient boosting with advanced regularization
- Configurations: Various depth, learning rate, subsample ratios
- L1/L2 regularization support
- Early stopping capability
- **Expected Performance**: RMSE ~1000-1200

#### LightGBM (if installed)
- Fast gradient boosting optimized for large datasets
- Leaf-wise tree growth
- Categorical feature support
- Memory efficient
- **Expected Performance**: RMSE ~1000-1200

**Advantages**:
- Handle non-linear relationships
- Feature interactions automatically
- Robust to outliers
- Feature importance analysis

### 4. Ensemble Models

**Purpose**: Combine multiple models for improved predictions

#### Weighted Ensemble
- Combines predictions from top models
- Strategies: uniform, performance-based weighting
- Simple but effective

#### Stacking Ensemble
- Two-level architecture:
  - **Level 1**: Multiple diverse base models
  - **Level 2**: Meta-learner (Ridge regression)
- Uses out-of-fold predictions
- Can include original features
- **Expected Performance**: RMSE ~900-1100 (best overall)

**Advantages**:
- Reduces model variance
- Combines different model strengths
- Often achieves best performance

## Training Strategy

### Cross-Validation

**Time Series Split** (5 folds):
```
Fold 1: Train [===    ] Test [=]
Fold 2: Train [====   ] Test [=]
Fold 3: Train [=====  ] Test [=]
Fold 4: Train [====== ] Test [=]
Fold 5: Train [=======] Test [=]
```

Respects temporal order - no data leakage from future to past.

### Hyperparameter Tuning

**Grid Search** for systematic exploration:
- Ridge: alpha values [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
- Random Forest: n_estimators, max_depth, min_samples_split
- XGBoost: learning_rate, max_depth, subsample, colsample_bytree

**Random Search** for large parameter spaces:
- More efficient for high-dimensional spaces
- 50-100 iterations typical

### Evaluation Metrics

**Primary Metrics**:
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Robust to outliers
- **MAPE** (Mean Absolute Percentage Error): Relative error measure
- **R²** (R-squared): Goodness of fit

**Model Selection**: Based on validation RMSE (lowest is best)

## Feature Requirements

Models expect these feature groups:

1. **Temporal Features**:
   - Year, Month, Day, DayOfWeek, WeekOfYear
   - IsWeekend, IsMonthStart, IsMonthEnd
   - DaysFromStart, DaysToEnd

2. **Store Features**:
   - Store ID (encoded)
   - StoreType (one-hot encoded)
   - Assortment (one-hot encoded)
   - CompetitionDistance

3. **Promotion Features**:
   - Promo (binary)
   - Promo2 (binary)
   - PromoInterval (encoded)

4. **Holiday Features**:
   - StateHoliday (encoded)
   - SchoolHoliday (binary)

5. **Lag Features**:
   - Sales_Lag_7, Sales_Lag_14, Sales_Lag_30
   - Sales_RollingMean_7, Sales_RollingMean_30

6. **Competition Features**:
   - CompetitionOpenDays
   - HasCompetition

## Training Pipeline

1. **Data Loading**: Load preprocessed features
2. **Baseline Training**: Quick performance baseline
3. **Linear Models**: Train regularized linear models
4. **Tree Models**: Train Random Forest, XGBoost, LightGBM
5. **Ensemble Training**: Combine best models
6. **Evaluation**: Compare all models
7. **Model Saving**: Save best model with metadata

## Expected Results

| Model Type | Expected RMSE | Training Time | Memory Usage |
|------------|---------------|---------------|--------------|
| Baseline | 2000-3000 | < 1 min | Low |
| Linear | 1500-2000 | 1-5 min | Low |
| Random Forest | 1200-1500 | 10-30 min | Medium |
| XGBoost | 1000-1200 | 5-20 min | Medium |
| LightGBM | 1000-1200 | 3-15 min | Low |
| Stacking | 900-1100 | 20-60 min | High |

## Best Practices

1. **Always start with baselines** - establish minimum performance
2. **Use time-series aware CV** - prevent data leakage
3. **Monitor for overfitting** - compare train vs validation metrics
4. **Feature importance analysis** - understand model decisions
5. **Ensemble diverse models** - combine different architectures
6. **Save metadata** - track hyperparameters and performance
7. **Version models** - track experiments and changes

## Model Selection Criteria

Choose model based on:
- **Performance**: Validation RMSE
- **Speed**: Training and inference time
- **Interpretability**: Need to explain predictions?
- **Resources**: Memory and compute constraints
- **Deployment**: Production requirements

## Next Steps After Training

1. **Test Set Evaluation**: Evaluate on held-out test set
2. **Error Analysis**: Identify systematic errors
3. **Feature Engineering**: Create additional features if needed
4. **Hyperparameter Tuning**: Fine-tune best models
5. **Production Deployment**: Package model for serving
6. **Monitoring**: Track performance in production

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Ensemble Methods Guide](https://scikit-learn.org/stable/modules/ensemble.html)
