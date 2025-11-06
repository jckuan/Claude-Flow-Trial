# Project Structure

MLE-STAR Rossmann Store Sales Forecasting Project

Last Updated: November 6, 2025

```
MLE-STAR-trial/
│
├── 📄 README.md                              # ⭐ START HERE - Main project documentation
├── 📄 DOCUMENTATION_GUIDE.md                 # 📖 Reading order and navigation guide
├── 📄 PROJECT_COMPLETION_SUMMARY.md          # ✅ Overall project status
├── 📄 PHASE2_COMPLETION.md                   # Feature engineering phase report
├── 📄 TESTING_SUMMARY.md                     # Test coverage and results
├── 📄 README_FEATURES.md                     # Quick feature reference
├── 📄 CLAUDE.md                              # AI assistant usage notes
├── 📄 requirements.txt                       # Python dependencies
├── 📄 pytest.ini                             # Test configuration
├── 📓 rossmann_eda.ipynb                     # Interactive EDA notebook
│
├── 📁 analysis/                              # Analysis results and reports
│   ├── FINAL_PROJECT_SUMMARY.md              # ⭐ Executive summary
│   ├── XGBOOST_TUNING_RESULTS.md            # Hyperparameter tuning results
│   └── ENSEMBLE_RESULTS.md                   # Ensemble evaluation results
│
├── 📁 results/                               # Final predictions and submissions
│   ├── submission_final.csv                  # ⭐ KAGGLE SUBMISSION (XGBoost)
│   ├── submission.csv                        # Earlier Random Forest submission
│   └── submission_report.csv                 # Submission metadata
│
├── 📁 docs/                                  # Comprehensive documentation
│   ├── RESULTS.md                            # ⭐ COMPLETE RESULTS & ANALYSIS
│   ├── FEATURE_ENGINEERING.md                # Feature engineering guide
│   ├── MODEL_ARCHITECTURE_SUMMARY.md         # All models documented
│   ├── METHODOLOGY.md                        # MLE-STAR framework
│   ├── eda_report.md                         # EDA written report
│   ├── eda_key_insights.md                  # Key findings from EDA
│   ├── phase1_summary.md                     # Phase 1 completion
│   └── [Additional documentation]
│
├── 📁 data/                                  # Dataset files
│   ├── rossmann-store-sales/                # Raw data from Kaggle
│   │   ├── train.csv                        # 1,017,209 training records
│   │   ├── test.csv                         # 41,088 test records
│   │   ├── store.csv                        # 1,115 store metadata
│   │   └── data_description.md              # Data dictionary
│   └── processed/                           # Engineered features
│       ├── train_processed.csv              # 755,389 rows, 143 features
│       ├── val_processed.csv                # 43,065 rows, 143 features
│       ├── test_processed.csv               # 45,884 rows, 143 features
│       └── feature_names.txt                # List of all features
│
├── 📁 src/                                   # Source code
│   ├── features/                            # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── pipeline.py                      # ⭐ Main feature pipeline
│   │   ├── temporal_features.py             # Date/time features
│   │   ├── categorical_features.py          # Category encodings
│   │   ├── lag_features.py                  # Historical lag features
│   │   ├── rolling_features.py              # Rolling statistics
│   │   ├── preprocessing.py                 # Data preprocessing
│   │   └── engineering.py                   # Feature engineering utilities
│   │
│   ├── models/                              # Model implementations
│   │   ├── __init__.py
│   │   ├── baseline.py                      # Baseline models
│   │   ├── linear_models.py                 # Linear regression variants
│   │   ├── tree_models.py                   # RF, XGBoost, LightGBM
│   │   ├── ensemble_models.py               # Ensemble strategies
│   │   ├── trainer.py                       # Model training logic
│   │   └── evaluator.py                     # Model evaluation
│   │
│   ├── run_feature_pipeline.py              # Execute feature pipeline
│   └── train_models.py                      # Train multiple models
│
├── 📁 scripts/                               # Executable scripts
│   ├── run_eda.py                           # Run exploratory analysis
│   ├── train_model.py                       # Train specific model
│   ├── evaluate_model.py                    # Evaluate trained models
│   ├── predict.py                           # Generate predictions
│   ├── tune_xgboost.py                      # XGBoost hyperparameter tuning
│   ├── create_ensemble.py                   # Create ensemble models
│   ├── generate_final_submission.py         # Generate final submission
│   ├── quick_train_predict.py               # Quick training script
│   └── test_gradient_boosting.py            # Test XGBoost/LightGBM
│
├── 📁 tests/                                 # Test suite (119 tests, 99% coverage)
│   ├── conftest.py                          # Test fixtures
│   ├── test_features.py                     # 52 feature tests
│   ├── test_models.py                       # 38 model tests
│   ├── test_preprocessing.py                # 41 preprocessing tests
│   ├── test_data_loading.py                 # 26 data loading tests
│   └── test_pipeline.py                     # 22 pipeline tests
│
├── 📁 models/                                # Trained model files
│   ├── xgboost_deeptrees.pkl                # ⭐ BEST MODEL (RMSPE: 0.0108)
│   ├── xgboost_aggressive.pkl               # Alternative XGBoost
│   ├── xgboost_regularized.pkl              # Regularized XGBoost
│   ├── xgboost_baseline.pkl                 # Baseline XGBoost
│   ├── xgboost_moretrees.pkl                # More trees XGBoost
│   ├── random_forest_best.pkl               # Random Forest baseline
│   ├── lightgbm_test.pkl                    # LightGBM model
│   ├── xgboost_tuning_results.csv          # Tuning results table
│   └── ensemble_comparison_results.csv      # Ensemble comparison
│
├── 📁 notebooks/                             # Additional notebooks
│   └── [Experimental notebooks]
│
├── 📁 examples/                              # Usage examples
│   └── [Code examples]
│
└── 📁 venv/                                  # Python virtual environment
    └── [Python packages]
```

---

## 📊 Project Statistics

- **Total Python Code**: ~6,000 lines
- **Documentation**: ~50,000 words across 16 files
- **Tests**: 119 tests with 99% coverage
- **Features**: 143 engineered features
- **Models**: 26+ variants implemented
- **Training Data**: 755,389 samples
- **Test Predictions**: 41,088 samples

---

## 🎯 Key Files Quick Reference

### Must-Read Documentation
1. `README.md` - Start here
2. `DOCUMENTATION_GUIDE.md` - Navigation guide
3. `docs/RESULTS.md` - Complete results
4. `analysis/FINAL_PROJECT_SUMMARY.md` - Executive summary

### For Model Deployment
- Best Model: `models/xgboost_deeptrees.pkl`
- Predictions: `results/submission_final.csv`
- Inference Script: `scripts/predict.py`
- Feature Pipeline: `src/features/pipeline.py`

### For Development
- Feature Engineering: `src/features/`
- Model Training: `src/models/`
- Tests: `tests/`
- Scripts: `scripts/`

---

## 🚀 Quick Commands

```bash
# Run tests
pytest tests/ -v

# Generate predictions
python scripts/predict.py --model models/xgboost_deeptrees.pkl

# Train model
python scripts/train_model.py --model xgboost

# Run EDA
jupyter notebook rossmann_eda.ipynb

# Feature engineering
python src/run_feature_pipeline.py
```

---

*For detailed navigation instructions, see DOCUMENTATION_GUIDE.md*
