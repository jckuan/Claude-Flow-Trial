# Feature Engineering Pipeline - Quick Start

## Overview

Comprehensive feature engineering pipeline for Rossmann sales prediction with 80+ features.

## Files Created

### Core Modules (src/features/)
- `__init__.py` - Package initialization
- `temporal_features.py` - 18 temporal features (dates, cycles, holidays)
- `categorical_features.py` - 15+ categorical features (encoding, competition, promo)
- `lag_features.py` - 30+ lag/rolling features (historical patterns)
- `preprocessing.py` - Data cleaning, scaling, validation
- `pipeline.py` - Complete orchestrator

### Testing & Examples
- `tests/test_features.py` - Comprehensive unit tests
- `src/run_feature_pipeline.py` - Main execution script
- `examples/feature_engineering_example.py` - Simple example

### Documentation
- `docs/FEATURE_ENGINEERING.md` - Complete documentation
- `memory/feature_engineering_summary.json` - Metadata

## Quick Start

### Option 1: Run Complete Pipeline
```bash
cd /Users/jckuan/Dev/MLE-STAR-trial
python src/run_feature_pipeline.py
```

### Option 2: Use in Code
```python
from features.pipeline import create_features

datasets = create_features(
    data_path='data/rossmann-store-sales',
    save_path='data/processed'
)

train = datasets['train']
val = datasets['val']
test = datasets['test']
```

### Option 3: Manual Control
```python
from features.pipeline import FeatureEngineeringPipeline

pipeline = FeatureEngineeringPipeline()
datasets = pipeline.run_full_pipeline(save_path='data/processed')
```

## Features Created

### Temporal (18 features)
- Basic: Year, Month, Day, Quarter, WeekOfYear
- Indicators: IsWeekend, IsMonthStart, IsMonthEnd
- Cyclic: Month_Sin/Cos, DayOfWeek_Sin/Cos, Day_Sin/Cos
- Holiday: IsStateHoliday, IsSchoolHoliday, IsAnyHoliday

### Categorical (15+ features)
- Encoding: StoreType, Assortment, StateHoliday
- Competition: Distance, Age, MonthsSince
- Promo: IsPromo, IsPromo2Active, MonthsSincePromo2
- Interactions: StoreType × Assortment

### Lag & Rolling (30+ features)
- Lags: 1, 7, 14, 30 days for Sales and Customers
- Rolling: Mean, Std, Max, Min (7, 14, 30-day windows)
- Advanced: EMA, Expanding Mean, Momentum, Trend

### Preprocessing
- Missing value imputation
- Outlier detection and capping
- Feature scaling (Standard/Robust/MinMax)
- Time-based splits (train/val/test)

## Output

Processed data saved to `data/processed/`:
- `train_processed.csv` - Training set
- `val_processed.csv` - Validation set (48 days)
- `test_processed.csv` - Test set (48 days)
- `feature_names.txt` - All feature names

## Testing

```bash
# Run all tests
pytest tests/test_features.py -v

# Run specific tests
pytest tests/test_features.py::TestTemporalFeatureEngineer -v

# With coverage
pytest tests/test_features.py --cov=src/features
```

## Performance

- Processing time: 5-10 minutes (full dataset)
- Memory usage: ~500 MB peak
- Features created: 80+ total

## Next Steps for Model Training

1. Load processed data from `data/processed/`
2. Use `Sales` column as target variable
3. Select features based on importance
4. Train models (XGBoost, LightGBM, etc.)
5. Evaluate on validation set

## Key Insights from EDA

- Mean sales: $5,773.82
- Sales-Customer correlation: 0.82
- Promo lift: 38.77%
- Zero sales: 16.99% of records
- Missing competition data: 31.75%

## Documentation

Full documentation: `docs/FEATURE_ENGINEERING.md`

## Coordination

Feature engineering metadata stored in memory for swarm coordination:
- Memory key: `swarm/phase2/feature-pipeline`
- Summary: `memory/feature_engineering_summary.json`

---

**Phase**: Phase 2 - Feature Engineering
**Agent**: FeatureEngineer
**Status**: ✅ Completed
**Timestamp**: 2025-11-06
