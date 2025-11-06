# MLE-STAR Methodology Documentation

## Overview

This project implements the MLE-STAR (Machine Learning Engineering - Search, Train, Adapt, Refine) methodology for the Rossmann Store Sales forecasting problem.

## Problem Statement

Forecast daily sales for 1,115 Rossmann drug stores up to 6 weeks in advance. Sales are influenced by:
- Promotions
- Competition
- School and state holidays
- Seasonality
- Store characteristics

## Dataset Information

### Training Data (1,017,209 records)
- **Date Range**: January 1, 2013 to July 31, 2015 (941 days)
- **Features**: Store, DayOfWeek, Date, Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday
- **Target**: Sales (daily revenue per store)

### Store Data (1,115 stores)
- **Features**: StoreType, Assortment, CompetitionDistance, CompetitionOpenSince, Promo2, PromoInterval
- **Missing Values**: Competition data (31.75%), Promo2 data (48.79%)

### Test Data (41,088 records)
- **Date Range**: August 1, 2015 to September 17, 2015 (48 days)
- **Task**: Predict sales for each store-day combination

## MLE-STAR Phases

### Phase 1: Search (Exploratory Data Analysis)

**Objectives**:
- Understand data structure and quality
- Identify patterns and relationships
- Detect anomalies and outliers
- Determine feature engineering opportunities

**Key Findings**:
1. **Sales Patterns**:
   - Average daily sales: $5,773.82
   - Strong correlation with customer count (r=0.824)
   - 16.99% records have zero sales (stores closed)
   - Promo increases sales by 38.77%

2. **Temporal Patterns**:
   - Clear weekly seasonality (lower sales on Sundays)
   - Monthly seasonality (December highest, January lowest)
   - Year-over-year trends

3. **Store Characteristics**:
   - 4 store types (a, b, c, d)
   - 3 assortment levels (a, b, c)
   - Type 'b' stores have highest average sales ($10,059)
   - Assortment 'c' has highest average sales ($6,059)

4. **Missing Data**:
   - No missing values in training data
   - Store data: Competition information (31.75%), Promo2 (48.79%)

### Phase 2: Train (Model Development)

**Feature Engineering Strategy**:

1. **Temporal Features**:
   - Day, Month, Year, DayOfWeek, WeekOfYear
   - IsWeekend, IsMonthStart, IsMonthEnd
   - DaysSinceLastPromo, DaysUntilNextPromo
   - CompetitionOpenMonths

2. **Lag Features**:
   - Sales lag (1, 7, 14, 30 days)
   - Rolling averages (7, 14, 30 days)
   - Store-level historical averages

3. **Store Features**:
   - One-hot encoding for StoreType, Assortment
   - Competition distance bins
   - Promo participation indicators

4. **Interaction Features**:
   - Promo x DayOfWeek
   - StoreType x Assortment
   - Holiday x Promo

**Model Candidates**:

1. **Baseline Models**:
   - Historical average
   - Moving average
   - Seasonal naive

2. **Traditional ML**:
   - Linear Regression
   - Ridge/Lasso Regression
   - Random Forest
   - Gradient Boosting (XGBoost, LightGBM)

3. **Advanced Models**:
   - Time Series Models (SARIMA, Prophet)
   - Deep Learning (LSTM, Transformer)
   - Ensemble methods

**Evaluation Metric**:
- Root Mean Square Percentage Error (RMSPE)
- Formula: RMSPE = sqrt(mean(((y_true - y_pred) / y_true)^2))
- Only non-zero sales days are considered

### Phase 3: Adapt (Model Tuning)

**Cross-Validation Strategy**:
- Time-series split (respecting temporal order)
- 5-fold validation with expanding window
- Validation set: Last 6 weeks of training data

**Hyperparameter Tuning**:
- Grid search for traditional models
- Bayesian optimization for complex models
- Random search for initial exploration

**Key Parameters by Model**:

1. **Random Forest**:
   - n_estimators: [100, 200, 500]
   - max_depth: [10, 20, 30, None]
   - min_samples_split: [2, 5, 10]

2. **XGBoost/LightGBM**:
   - learning_rate: [0.01, 0.05, 0.1]
   - max_depth: [3, 5, 7]
   - n_estimators: [100, 200, 500]
   - subsample: [0.8, 0.9, 1.0]

3. **Neural Networks**:
   - layers: [64, 128, 256]
   - dropout: [0.2, 0.3, 0.4]
   - learning_rate: [0.001, 0.0001]

### Phase 4: Refine (Model Selection & Ensemble)

**Model Selection Criteria**:
1. RMSPE on validation set
2. Training time and inference speed
3. Model interpretability
4. Generalization capability

**Ensemble Strategy**:
- Weighted average of top 3 models
- Weights determined by validation performance
- Stacking with meta-learner

**Feature Importance Analysis**:
- SHAP values for model interpretation
- Permutation importance
- Partial dependence plots

**Final Model Pipeline**:
```
1. Data preprocessing
2. Feature engineering
3. Model prediction
4. Post-processing (handle edge cases)
5. Evaluation
```

## Implementation Best Practices

### Code Organization
```
src/
├── data/
│   ├── data_loader.py      # Data loading utilities
│   └── preprocessing.py     # Data cleaning and validation
├── features/
│   ├── engineering.py       # Feature creation
│   └── selection.py         # Feature selection
├── models/
│   ├── baseline.py          # Simple baseline models
│   ├── traditional_ml.py    # Scikit-learn models
│   ├── gradient_boosting.py # XGBoost, LightGBM
│   └── ensemble.py          # Ensemble methods
├── evaluation/
│   ├── metrics.py           # Custom metrics (RMSPE)
│   └── validation.py        # Cross-validation
└── utils/
    ├── config.py            # Configuration management
    └── logger.py            # Logging utilities
```

### Testing Strategy
- Unit tests for all functions
- Integration tests for pipelines
- Test coverage target: >80%
- Property-based testing for data validation

### Reproducibility
- Fixed random seeds
- Version-controlled dependencies
- Configuration files for all hyperparameters
- Experiment tracking (MLflow/Weights & Biases)

## Expected Outcomes

### Performance Targets
- Baseline RMSPE: ~0.20 (simple average)
- Target RMSPE: <0.12 (competitive performance)
- Stretch goal: <0.10 (top-tier performance)

### Deliverables
1. Clean, modular codebase
2. Comprehensive documentation
3. Trained models with performance metrics
4. Feature importance analysis
5. Prediction pipeline for new data
6. Final report with insights

## References

- Kaggle Competition: [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- MLE-STAR Framework: Systematic approach to ML engineering
- Time Series Forecasting: Standard practices and techniques

## Version History

- v1.0.0 (2025-11-06): Initial methodology documentation
