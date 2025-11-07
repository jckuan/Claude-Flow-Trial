# Rossmann Store Sales Forecasting

**Machine Learning Engineering Project using MLE-STAR Methodology**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Test Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen.svg)]()
[![Project Status](https://img.shields.io/badge/status-complete-success.svg)]()
[![RMSPE](https://img.shields.io/badge/RMSPE-0.0108-blue.svg)]()

> Predict daily sales for Rossmann stores using advanced feature engineering and gradient boosting

## 🎯 Project Overview

This project predicts sales for 1,115 Rossmann drugstores across Germany using historical sales data, store information, and promotional data. Built following the **MLE-STAR methodology** (Search, Train, Adapt, Refine), it achieves **1.08% RMSPE** using XGBoost with 143 engineered features.

### Key Achievements
- ✅ **RMSPE: 0.010757** (~1.08% average error) - 91% better than target
- ✅ **R² Score: 0.9992** (99.92% variance explained)
- ✅ **143 Engineered Features** (temporal, categorical, lag, rolling)
- ✅ **26+ Models Implemented** (baseline, linear, tree-based, ensemble)
- ✅ **99% Test Coverage** (119 tests)
- ✅ **Production-Ready** with deployment guide

---

## Quick Start Guide

**Three Essential Documents**:
1. 📋 **README.md** (this file) - Project overview and setup
2. 🚀 **[QUICK_START.md](QUICK_START.md)** - 4-command workflow to reproduce all results
3. ✅ **[FINAL_COMPLETION_SUMMARY.md](FINAL_COMPLETION_SUMMARY.md)** - Complete requirements coverage, project structure, and status

**Supporting Documentation**:
- 📊 **[docs/RESULTS.md](docs/RESULTS.md)** - Comprehensive technical results and analysis
- 📈 **[analysis/FINAL_PROJECT_SUMMARY.md](analysis/FINAL_PROJECT_SUMMARY.md)** - Executive summary
- 🔧 **[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)** - Feature engineering details
- 🤖 **[docs/MODEL_ARCHITECTURE_SUMMARY.md](docs/MODEL_ARCHITECTURE_SUMMARY.md)** - Model implementations

---
- **[DOCUMENTATION_GUIDE.md](DOCUMENTATION_GUIDE.md)** - Complete navigation guide

### Analysis & Results
- [**docs/REPORT.md**](docs/REPORT.md) ⭐ - Professional sales analysis report (auto-generated)
- [**docs/RESULTS.md**](docs/RESULTS.md) - Complete technical analysis
- [**analysis/FINAL_PROJECT_SUMMARY.md**](analysis/FINAL_PROJECT_SUMMARY.md) - Executive summary
- [**docs/references.md**](docs/references.md) - Literature citations

### Technical Documentation
- [**Project Structure**](PROJECT_STRUCTURE.md) - File organization
- [**Model Details**](analysis/XGBOOST_TUNING_RESULTS.md) - Hyperparameter tuning
- [**Features Guide**](docs/FEATURE_ENGINEERING.md) - All 143 features explained

---

## 🚀 Complete Automated Workflow

Follow the 4-command workflow in **[QUICK_START.md](QUICK_START.md)** to reproduce all results:

```bash
# 1. Run ML pipeline (feature engineering → training → evaluation)
python scripts/run_full_pipeline.py

# 2. Generate visualizations (5 professional charts)
python scripts/generate_visualizations.py

# 3. Generate professional report (with business insights)
python scripts/generate_report.py

# 4. Search literature (optional, requires internet)
python agents/literature_search_agent.py --query "retail sales forecasting"
```

**Outputs**:
- `data/processed/` - Processed datasets with time-based splits (validation=48 days, test=48 days)
- `models/` - Trained XGBoost_DeepTrees model (with RandomForest fallback)
- `results/metrics.json` - Evaluation metrics (RMSPE, RMSE, MAE, MAPE, R²)
- `results/submission_final.csv` - Final predictions (41,088 rows)
- `docs/figures/` - Professional visualizations (5 charts)
- `docs/REPORT.md` - Comprehensive analysis report
- `docs/references.md` - Literature references (if step 4 run)

**For complete details**, see:
- **[QUICK_START.md](QUICK_START.md)** - Complete workflow guide with troubleshooting
- **[FINAL_COMPLETION_SUMMARY.md](FINAL_COMPLETION_SUMMARY.md)** - Requirements coverage and project structure


