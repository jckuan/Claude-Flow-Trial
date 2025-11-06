"""Tests for model training and prediction."""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class TestModelTraining:
    """Test model training functionality."""

    def test_model_initialization(self):
        """Test that models can be initialized."""
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=10, random_state=42)

        assert lr is not None
        assert rf is not None

    def test_linear_regression_training(self, sample_train_data):
        """Test linear regression model training."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        assert model is not None
        assert hasattr(model, 'coef_')
        assert len(model.coef_) == 3

    def test_random_forest_training(self, sample_train_data):
        """Test random forest model training."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        assert model is not None
        assert hasattr(model, 'feature_importances_')
        assert len(model.feature_importances_) == 3

    def test_feature_importance_extraction(self, sample_train_data):
        """Test extraction of feature importances."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        assert importances.sum() > 0
        assert np.isclose(importances.sum(), 1.0), \
            "Feature importances should sum to 1"

    def test_model_fitting_convergence(self, sample_train_data):
        """Test that models converge during training."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        # Model should have non-zero coefficients
        assert not np.allclose(model.coef_, 0)


class TestModelPrediction:
    """Test model prediction functionality."""

    def test_single_prediction(self, sample_train_data):
        """Test prediction on single sample."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        single_input = X[0].reshape(1, -1)
        prediction = model.predict(single_input)

        assert prediction.shape == (1,)
        assert prediction[0] > 0

    def test_batch_prediction(self, sample_train_data):
        """Test prediction on batch of samples."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X[:10])

        assert predictions.shape == (10,)
        assert (predictions > 0).all() or predictions.min() >= 0

    def test_prediction_bounds(self, sample_train_data):
        """Test that predictions are within reasonable bounds."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        # Predictions should be roughly in the range of training data
        assert predictions.min() > y.min() - 5000
        assert predictions.max() < y.max() + 5000

    def test_all_samples_have_predictions(self, sample_train_data):
        """Test that all samples get predictions."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        assert len(predictions) == len(X)
        assert not np.isnan(predictions).any()


class TestModelEvaluation:
    """Test model evaluation metrics."""

    def test_mean_squared_error_calculation(self, sample_train_data):
        """Test MSE calculation."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)

        assert mse > 0
        assert not np.isnan(mse)

    def test_mean_absolute_error_calculation(self, sample_train_data):
        """Test MAE calculation."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)

        assert mae > 0
        assert not np.isnan(mae)

    def test_r2_score_calculation(self, sample_train_data):
        """Test R2 score calculation."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        r2 = r2_score(y, predictions)

        assert -1 <= r2 <= 1
        assert not np.isnan(r2)

    def test_rmse_calculation(self, sample_train_data):
        """Test RMSE calculation."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, predictions))

        assert rmse > 0
        assert not np.isnan(rmse)

    def test_mape_calculation(self, sample_train_data):
        """Test MAPE calculation."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        # MAPE = mean(|actual - predicted| / actual)
        mape = np.mean(np.abs((y - predictions) / (y + 1)))

        assert mape >= 0
        assert not np.isnan(mape)


class TestCrossValidation:
    """Test cross-validation functionality."""

    def test_train_test_split(self, sample_train_data):
        """Test train-test split."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) > len(X_test)

    def test_temporal_train_test_split(self, sample_train_data):
        """Test temporal train-test split."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()
        df = df.sort_values('Date').reset_index(drop=True)

        split_idx = int(len(df) * 0.8)

        train = df.iloc[:split_idx]
        test = df.iloc[split_idx:]

        X_train = train[['Customers', 'Promo', 'DayOfWeek']].values
        X_test = test[['Customers', 'Promo', 'DayOfWeek']].values

        assert len(train) + len(test) == len(df)
        assert train['Date'].max() <= test['Date'].min()

    def test_stratified_split_by_open_status(self, sample_train_data):
        """Test stratified split maintaining store distribution."""
        df = sample_train_data.copy()

        # Split maintaining open status ratio
        X = df[['Customers', 'DayOfWeek']].values
        y = df['Sales'].values
        open_status = df['Open'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=open_status, random_state=42
        )

        assert len(X_train) + len(X_test) == len(X)


class TestModelPerformanceThresholds:
    """Test model meets performance thresholds."""

    def test_model_r2_score_minimum(self, sample_train_data):
        """Test that model achieves minimum R2 score."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        r2 = r2_score(y, predictions)

        # Should achieve at least some positive R2 on training set
        assert r2 > 0.1, "Model should achieve reasonable fit"

    def test_model_mae_acceptable(self, sample_train_data):
        """Test that model MAE is within acceptable range."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)

        # MAE should be less than mean sales
        assert mae < y.mean(), "MAE should be reasonable"

    def test_random_forest_outperforms_baseline(self, sample_train_data):
        """Test that RF performs better than simple baseline."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        # Baseline: predict mean
        baseline_pred = np.full_like(y, y.mean(), dtype=float)
        baseline_mae = mean_absolute_error(y, baseline_pred)

        # RF model
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        rf_pred = model.predict(X)
        rf_mae = mean_absolute_error(y, rf_pred)

        assert rf_mae < baseline_mae, "RF should outperform mean baseline"


class TestPredictionConsistency:
    """Test prediction consistency."""

    def test_deterministic_predictions(self, sample_train_data):
        """Test that model gives same predictions each time."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        pred1 = model.predict(X[:5])
        pred2 = model.predict(X[:5])

        np.testing.assert_array_equal(pred1, pred2, \
            "Predictions should be deterministic")

    def test_random_forest_deterministic_with_seed(self, sample_train_data):
        """Test RF determinism with fixed random seed."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model1 = RandomForestRegressor(n_estimators=5, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X[:5])

        model2 = RandomForestRegressor(n_estimators=5, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X[:5])

        np.testing.assert_array_almost_equal(pred1, pred2, \
            "RF predictions should match with same seed")

    def test_no_nan_predictions(self, sample_train_data):
        """Test that predictions don't contain NaN."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        assert not np.isnan(predictions).any(), \
            "Predictions should not contain NaN"
