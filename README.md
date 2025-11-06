# Rossmann Store Sales Prediction - MLE-STAR Framework

A systematic machine learning project applying the MLE-STAR (Search, Train, Adapt, Refine) methodology to forecast daily sales for Rossmann drug stores.

## Project Overview

**Goal**: Predict 6 weeks of daily sales for 1,115 Rossmann stores using historical sales data, store characteristics, and promotional information.

**Dataset**: [Kaggle Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)
- Training: 1,017,209 records (Jan 2013 - Jul 2015)
- Test: 41,088 records (Aug 2015 - Sep 2015)
- Stores: 1,115 unique locations with metadata

**Evaluation Metric**: Root Mean Square Percentage Error (RMSPE)

## Project Structure

```
MLE-STAR-trial/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
│
├── data/                      # Data directory (gitignored)
│   └── rossmann-store-sales/
│       ├── train.csv
│       ├── test.csv
│       ├── store.csv
│       └── data_description.md
│
├── src/                       # Source code
│   ├── data/
│   │   ├── data_loader.py    # Data loading utilities
│   │   └── preprocessing.py   # Data cleaning and validation
│   ├── features/
│   │   ├── engineering.py     # Feature creation
│   │   └── selection.py       # Feature selection
│   ├── models/
│   │   ├── baseline.py        # Simple baseline models
│   │   ├── traditional_ml.py  # Scikit-learn models
│   │   ├── gradient_boosting.py # XGBoost, LightGBM
│   │   └── ensemble.py        # Ensemble methods
│   ├── evaluation/
│   │   ├── metrics.py         # Custom metrics (RMSPE)
│   │   └── validation.py      # Cross-validation
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
│
├── tests/                     # Unit and integration tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_models.py
│
├── scripts/                   # Execution scripts
│   ├── run_eda.py            # Exploratory data analysis
│   ├── train_model.py        # Model training
│   ├── evaluate_model.py     # Model evaluation
│   └── predict.py            # Generate predictions
│
├── docs/                      # Documentation
│   ├── METHODOLOGY.md        # MLE-STAR methodology
│   ├── RESULTS.md            # Final results and insights
│   └── API.md                # Code API documentation
│
├── models/                    # Saved models (gitignored)
├── logs/                      # Log files (gitignored)
└── notebooks/                 # Jupyter notebooks
    └── rossmann_eda.ipynb    # Exploratory analysis
```

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd MLE-STAR-trial

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

Download the Rossmann Store Sales dataset from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) and place files in `data/rossmann-store-sales/`:
- train.csv
- test.csv
- store.csv

### 3. Run Exploratory Data Analysis

```bash
# Option 1: Run the notebook
jupyter notebook rossmann_eda.ipynb

# Option 2: Run the script (coming soon)
python scripts/run_eda.py
```

### 4. Train Models

```bash
# Train all models with default configuration
python scripts/train_model.py

# Train specific model
python scripts/train_model.py --model xgboost

# Train with custom parameters
python scripts/train_model.py --model lightgbm --n-folds 5
```

### 5. Evaluate and Generate Predictions

```bash
# Evaluate trained models
python scripts/evaluate_model.py

# Generate predictions for test set
python scripts/predict.py --model best_model.pkl --output submission.csv
```

## MLE-STAR Methodology

This project follows the MLE-STAR framework:

### 1. Search (Exploratory Data Analysis)
- Understand data structure and quality
- Identify patterns and relationships
- Detect anomalies and outliers
- Determine feature engineering opportunities

**Key Findings**:
- Strong correlation between sales and customers (r=0.824)
- 38.77% sales lift with promotions
- Clear weekly and monthly seasonality
- Store type and assortment significantly impact sales

### 2. Train (Model Development)
- Feature engineering (temporal, lag, store features)
- Train multiple model families (baseline, tree-based, boosting)
- Implement proper cross-validation

**Models**:
- Baseline: Historical averages
- Traditional ML: Random Forest
- Gradient Boosting: XGBoost, LightGBM
- Ensemble: Weighted average of top models

### 3. Adapt (Hyperparameter Tuning)
- Time-series cross-validation
- Grid search and Bayesian optimization
- Model-specific parameter tuning

### 4. Refine (Model Selection & Deployment)
- Select best model based on RMSPE
- Feature importance analysis
- Create production-ready pipeline

## Key Features

- **Modular Architecture**: Clean separation of concerns
- **Comprehensive Testing**: Unit and integration tests
- **Reproducible**: Fixed random seeds and version control
- **Well-Documented**: Inline documentation and user guides
- **Production-Ready**: Logging, error handling, validation

## Performance Targets

- **Baseline RMSPE**: ~0.20 (simple average)
- **Target RMSPE**: <0.12 (competitive)
- **Stretch Goal**: <0.10 (top-tier)

## Development Status

- [x] Project setup and structure
- [x] Exploratory data analysis
- [ ] Feature engineering implementation
- [ ] Baseline model development
- [ ] Advanced model training
- [ ] Hyperparameter tuning
- [ ] Model evaluation and selection
- [ ] Final documentation and results

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_loader.py -v
```

## Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Contributing

This is a learning project following MLE best practices. Contributions and suggestions welcome!

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Kaggle for the Rossmann Store Sales dataset
- MLE-STAR framework for systematic ML development
- Open source ML community for excellent tools and libraries

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This project is for educational purposes and demonstrates best practices in machine learning engineering, including proper code organization, testing, documentation, and reproducibility.
