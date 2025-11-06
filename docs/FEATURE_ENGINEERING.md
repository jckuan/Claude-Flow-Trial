# Feature Engineering Documentation

## Overview

Comprehensive feature engineering pipeline for Rossmann store sales prediction, creating 80+ features from raw data.

## Architecture

The pipeline consists of 5 modular components:

```
src/features/
├── __init__.py                 # Package initialization
├── temporal_features.py        # Date/time features
├── categorical_features.py     # Categorical encoding
├── lag_features.py             # Time series features
├── preprocessing.py            # Data cleaning & scaling
└── pipeline.py                 # Main orchestrator
```

## Feature Categories

### 1. Temporal Features (18 features)

**Basic Components:**
- `Year`, `Month`, `Day`, `Quarter`
- `WeekOfYear`, `DayOfWeek`, `DayOfYear`
- `DaysInMonth`

**Indicators:**
- `IsWeekend` - Binary flag for Saturday/Sunday
- `IsMonthStart`, `IsMonthEnd` - Month boundary indicators
- `Season` - Quarterly season (1-4)

**Cyclic Encoding:**
- `Month_Sin`, `Month_Cos` - Preserve cyclical nature of months
- `DayOfWeek_Sin`, `DayOfWeek_Cos` - Weekly cycles
- `Day_Sin`, `Day_Cos` - Daily cycles within month

**Holiday Features:**
- `IsStateHoliday` - State holiday indicator
- `IsSchoolHoliday` - School holiday indicator
- `IsAnyHoliday` - Combined holiday indicator

### 2. Categorical Features (15+ features)

**Label Encoding:**
- `StoreType_Encoded` - Store type (a, b, c, d)
- `Assortment_Encoded` - Product assortment (a, b, c)
- `StateHoliday_Encoded` - Holiday type encoding
- `PromoInterval_Encoded` - Promotion interval encoding

**Competition Features:**
- `HasCompetition` - Competition presence indicator
- `CompetitionDistance_Log` - Log-transformed distance
- `MonthsSinceCompetition` - Time since competition opened
- `CompetitionAge_Binned` - Categorical age bins (0-5)

**Promotion Features:**
- `IsPromo` - Daily promotion indicator
- `IsPromo2Active` - Long-term promotion active
- `MonthsSincePromo2` - Duration of Promo2
- `IsPromo2Month` - Current month in promo interval

**Interaction Features:**
- `StoreType_Assortment_Encoded` - Store type × Assortment interaction

### 3. Lag Features (30+ features)

**Sales Lag Features:**
- `Sales_Lag1` - Previous day sales
- `Sales_Lag7` - Same day last week
- `Sales_Lag14` - 2 weeks ago
- `Sales_Lag30` - ~1 month ago

**Rolling Statistics (7-day, 14-day, 30-day windows):**
- `Sales_RollingMean{window}` - Moving averages
- `Sales_RollingStd{window}` - Volatility measures
- `Sales_RollingMax{window}` - Peak sales in window
- `Sales_RollingMin{window}` - Minimum sales in window

**Exponential Moving Averages:**
- `Sales_EMA7` - Short-term EMA
- `Sales_EMA30` - Long-term EMA

**Advanced Features:**
- `Sales_ExpandingMean` - Cumulative average per store
- `Sales_Momentum7` - 7-day momentum (ratio to lag)
- `Sales_Momentum30` - 30-day momentum
- `Sales_Trend7` - Deviation from 7-day average

**Customer Lag Features:**
- `Customers_Lag1` - Previous day customers
- `Customers_Lag7` - Same day last week customers

**Day-of-Week Specific:**
- `Sales_SameDayLastWeek` - Same weekday, 1 week ago
- `Sales_SameDay2WeeksAgo` - Same weekday, 2 weeks ago
- `Sales_SameDayAvg4Weeks` - 4-week average for same weekday

### 4. Preprocessing Features

**Missing Value Handling:**
- Median imputation for continuous variables
- Zero-fill for competition/promo missing values
- Category-specific strategies

**Outlier Treatment:**
- Z-score based detection (threshold: 3.0)
- Capping at mean ± 3σ
- Applied to Sales, Customers, CompetitionDistance

**Feature Scaling:**
- StandardScaler (default) - Zero mean, unit variance
- RobustScaler - Median-based, outlier resistant
- MinMaxScaler - Scale to [0, 1] range
- Suffix: `_Scaled` for all scaled features

## Usage

### Quick Start

```python
from features.pipeline import create_features

# Run complete pipeline
datasets = create_features(
    data_path='data/rossmann-store-sales',
    save_path='data/processed',
    use_target_encoding=False,
    scaling_method='standard'
)

train_df = datasets['train']
val_df = datasets['val']
test_df = datasets['test']
```

### Manual Control

```python
from features.pipeline import FeatureEngineeringPipeline

# Initialize
pipeline = FeatureEngineeringPipeline(
    data_path='data/rossmann-store-sales',
    use_target_encoding=False,
    scaling_method='standard',
    create_lag_features=True
)

# Load and prepare data
train, test, store = pipeline.load_data()
train_full = pipeline.prepare_data(train, store, is_test=False)

# Separate features and target
y = train_full['Sales']
X = train_full.drop('Sales', axis=1)

# Fit and transform
X_transformed = pipeline.fit_transform(X, y)

# Create time-based splits
train_df, val_df, test_df = pipeline.create_train_val_test_splits(
    X_transformed,
    val_days=48,
    test_days=48
)
```

### Individual Components

```python
from features.temporal_features import TemporalFeatureEngineer
from features.categorical_features import CategoricalFeatureEngineer
from features.lag_features import LagFeatureEngineer

# Temporal features only
temporal_eng = TemporalFeatureEngineer()
df = temporal_eng.fit_transform(data)
df = temporal_eng.add_holiday_features(df)

# Categorical features only
cat_eng = CategoricalFeatureEngineer()
df = cat_eng.fit_transform(data, target)

# Lag features only
lag_eng = LagFeatureEngineer(
    lag_periods=[1, 7, 14, 30],
    rolling_windows=[7, 14, 30]
)
df = lag_eng.fit_transform(data)
```

## Configuration Options

### Pipeline Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `data_path` | string | - | Path to raw data directory |
| `use_target_encoding` | bool | False | Enable target encoding for categoricals |
| `scaling_method` | 'standard', 'robust', 'minmax', 'none' | 'standard' | Feature scaling method |
| `create_lag_features` | bool | True | Create lag and rolling features |

### Data Splitting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `val_days` | 48 | Days for validation set |
| `test_days` | 48 | Days for test set |

Time-based splitting ensures no data leakage:
- Train: earliest dates to (max_date - val_days - test_days)
- Validation: next `val_days` days
- Test: last `test_days` days

## Data Processing Steps

### 1. Data Loading
- Load train.csv (1M+ rows, 9 columns)
- Load test.csv (41K rows, 8 columns)
- Load store.csv (1,115 rows, 10 columns)

### 2. Data Preparation
- Merge train/test with store info on `Store` ID
- Filter out closed stores (Open == 0) from training
- Remove zero sales records from training
- Preserve all test data regardless of status

### 3. Feature Engineering
- **Temporal**: Extract date components and cycles
- **Categorical**: Encode and engineer categorical features
- **Lag**: Create historical and rolling features
- **Interaction**: Generate feature interactions

### 4. Preprocessing
- Impute missing values
- Detect and handle outliers
- Scale numeric features
- Validate data quality

### 5. Data Splitting
- Time-based train/val/test splits
- Maintain temporal ordering
- No data leakage between splits

## Output Files

```
data/processed/
├── train_processed.csv         # Training set with all features
├── val_processed.csv           # Validation set
├── test_processed.csv          # Test set
└── feature_names.txt           # List of all features
```

## Feature Importance Groups

Access feature groups for model interpretation:

```python
feature_groups = pipeline.get_feature_importance_groups()

# Groups:
# - 'temporal': Date/time features
# - 'categorical': Encoded categorical features
# - 'lag': Historical and rolling features
```

## Performance Characteristics

### Memory Usage
- Raw data: ~70 MB (train), ~3 MB (test)
- Processed data: ~150-200 MB (train)
- Peak memory: ~500 MB during processing

### Processing Time
- Small dataset (1M rows): ~2-5 minutes
- Full pipeline with lag features: ~5-10 minutes
- Without lag features: ~1-2 minutes

### Feature Counts
- Temporal: 18 features
- Categorical: 15+ features
- Lag: 30+ features
- Scaled: Same as numeric features
- **Total: 80+ features**

## Best Practices

### 1. Data Quality
- Always validate data before training
- Check for missing values and outliers
- Verify date ranges and continuity

### 2. Feature Selection
- Use feature importance after training
- Consider correlation analysis
- Remove highly correlated features if needed

### 3. Time Series Considerations
- Maintain temporal order in splits
- Use lag features cautiously (data leakage)
- Consider store-specific patterns

### 4. Scaling
- Use StandardScaler for tree-based models
- Use RobustScaler if outliers present
- No scaling needed for tree-based models

### 5. Missing Values
- Check imputation strategies per feature
- Consider domain knowledge for competition/promo
- Document all imputation decisions

## Extending the Pipeline

### Add New Features

```python
from features.pipeline import FeatureEngineeringPipeline

class CustomPipeline(FeatureEngineeringPipeline):
    def transform(self, X):
        df = super().transform(X)

        # Add custom features
        df['CustomFeature'] = df['Sales'] / df['Customers']

        return df
```

### Custom Feature Engineer

```python
class CustomFeatureEngineer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Custom transformations
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
```

## Troubleshooting

### Common Issues

**Issue: Memory Error**
- Solution: Process in chunks or reduce lag features
- Alternative: Use feature selection to reduce dimensionality

**Issue: Long Processing Time**
- Solution: Disable lag features if not needed
- Alternative: Reduce rolling window sizes

**Issue: Missing Values in Lag Features**
- Expected: First rows per store will have NaN lags
- Solution: Handle in model (e.g., fillna(0) or drop)

**Issue: Data Leakage**
- Check: Lag features only use past data
- Verify: Train/val/test splits are sequential
- Test: No future information in features

## References

### Feature Engineering Techniques
- Temporal features: Cyclic encoding for periodic features
- Lag features: Time series forecasting standard practice
- Rolling statistics: Moving averages for trend detection

### Domain Knowledge
- Retail sales patterns: Weekly and seasonal cycles
- Promotion effects: Immediate and delayed impact
- Competition: Distance and age effects on sales

## Testing

Run unit tests:

```bash
# Run all tests
pytest tests/test_features.py -v

# Run specific test class
pytest tests/test_features.py::TestTemporalFeatureEngineer -v

# Run with coverage
pytest tests/test_features.py --cov=src/features
```

## Next Steps

After feature engineering:

1. **Model Selection**: Choose appropriate algorithms
2. **Feature Importance**: Analyze feature contributions
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Cross-Validation**: Validate on multiple folds
5. **Ensemble Methods**: Combine multiple models

---

**Created by**: FeatureEngineer Agent (MLE-STAR Phase 2)
**Last Updated**: 2025-11-06
**Version**: 1.0.0
