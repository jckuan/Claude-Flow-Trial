# Ensemble Model Results

**Date**: November 6, 2025  
**Primary Metric**: RMSPE (Root Mean Square Percentage Error)

---

## Complete Model Comparison (Sorted by RMSPE)

| Rank | Model | RMSPE ⭐ | RMSE | MAE | R² |
|------|-------|---------|------|-----|-----|
| **1** | **XGBoost_DeepTrees** | **0.010757** | 90.33 | 37.49 | 0.9992 |
| 2 | Ensemble_BestHeavy | 0.011213 | 88.46 | 42.54 | 0.9992 |
| 3 | Ensemble_TopThree | 0.011291 | 90.23 | 41.08 | 0.9992 |
| 4 | Ensemble_Diversified | 0.011987 | 91.37 | 46.70 | 0.9991 |
| 5 | Ensemble_Conservative | 0.012450 | 93.53 | 49.46 | 0.9991 |
| 6 | XGBoost_Aggressive | 0.012505 | 94.33 | 47.14 | 0.9991 |
| 7 | Ensemble_Uniform | 0.014030 | 101.24 | 59.07 | 0.9989 |
| 8 | XGBoost_Regularized | 0.017096 | 121.05 | 72.30 | 0.9985 |
| 9 | Random_Forest | 0.022657 | 169.00 | 106.55 | 0.9970 |
| 10 | LightGBM | 0.023526 | 167.10 | 111.49 | 0.9971 |

---

## 🏆 Winner: XGBoost_DeepTrees (Single Model)

### Key Finding
**The single XGBoost_DeepTrees model outperforms all ensemble combinations!**

This suggests:
1. The model is already well-optimized and robust
2. Other models add noise rather than complementary information
3. Deep trees capture all necessary patterns in the data

### Performance
- **RMSPE**: 0.010757 (~1.08% average error)
- **RMSE**: 90.33
- **MAE**: 37.49 (Average $37.49 error per prediction)
- **R²**: 0.9992 (99.92% variance explained)

---

## Ensemble Strategy Analysis

### Best Ensemble: Ensemble_BestHeavy (Rank #2)
**Weights**: 60% XGBoost_DeepTrees, 20% XGBoost_Aggressive, 10% RF, 10% LightGBM
- RMSPE: 0.011213 (only 4% worse than single model)
- **Best RMSE**: 88.46 (slightly better than XGBoost_DeepTrees)
- Trade-off: Marginally higher percentage error but lower absolute error

### Ensemble_TopThree (Rank #3)
**Weights**: 50% DeepTrees, 30% Aggressive, 20% Regularized
- RMSPE: 0.011291
- Pure XGBoost ensemble
- Very close to single model performance

### Why Ensembles Didn't Win
1. **Model Similarity**: All top XGBoost models are highly correlated
2. **Dominant Performance**: XGBoost_DeepTrees is already optimal
3. **Weak Models**: Random Forest and LightGBM dilute performance

---

## Insights

### What Works
✅ **Deep Trees** (max_depth=10) capture complex patterns  
✅ **Lower Learning Rate** (0.05) with more depth  
✅ **Histogram-based** tree method for speed  
✅ **Single optimized model** beats ensemble in this case

### What Didn't Help
❌ Ensembling with weaker models (RF, LightGBM)  
❌ Equal weighting across diverse models  
❌ Over-regularization (conservative strategy)

---

## Model Selection for Final Submission

**Selected Model**: XGBoost_DeepTrees  
**Reason**: Best RMSPE (primary metric) with excellent overall performance

**Alternative**: Ensemble_BestHeavy  
**Reason**: Best RMSE if absolute error is more important than percentage error

---

## Files Generated

- `models/ensemble_comparison_results.csv` - Full comparison table
- Ensemble configurations tested but not saved (single model wins)

---

**Next Step**: Generate final submission using XGBoost_DeepTrees model
