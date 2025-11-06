"""Tests for end-to-end pipeline integration."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class TestDataPipeline:
    """Test complete data pipeline."""

    def test_train_store_merge(self, sample_train_data, sample_store_data):
        """Test merging training data with store information."""
        merged = sample_train_data.merge(sample_store_data, on='Store', how='left')

        assert len(merged) == len(sample_train_data), "Merge should preserve rows"
        assert 'StoreType' in merged.columns, "Merged data should have StoreType"
        assert 'Assortment' in merged.columns, "Merged data should have Assortment"

    def test_full_data_preprocessing(self, sample_train_data, sample_store_data):
        """Test full preprocessing pipeline."""
        # Merge data
        df = sample_train_data.merge(sample_store_data, on='Store', how='left')

        # Extract date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['DayOfMonth'] = df['Date'].dt.day

        # Fill missing values
        df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
        df['Promo2SinceWeek'].fillna(0, inplace=True)
        df['Promo2SinceYear'].fillna(0, inplace=True)

        # Encode categorical features
        df['StoreType_encoded'] = pd.Categorical(df['StoreType']).codes
        df['Assortment_encoded'] = pd.Categorical(df['Assortment']).codes

        assert 'Year' in df.columns
        assert 'Month' in df.columns
        assert df['CompetitionDistance'].isnull().sum() == 0
        assert 'StoreType_encoded' in df.columns

    def test_feature_engineering_pipeline(self, sample_train_data):
        """Test feature engineering pipeline."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        # Create lag features
        for store in df['Store'].unique():
            mask = df['Store'] == store
            df.loc[mask, 'Sales_lag1'] = df.loc[mask, 'Sales'].shift(1)

        # Create rolling features
        for store in df['Store'].unique():
            mask = df['Store'] == store
            df.loc[mask, 'Sales_rolling_7'] = \
                df.loc[mask, 'Sales'].rolling(window=7, min_periods=1).mean()

        # Create calendar features
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter

        assert 'Sales_lag1' in df.columns
        assert 'Sales_rolling_7' in df.columns
        assert 'Month' in df.columns
        assert 'Quarter' in df.columns

    def test_data_quality_checks_in_pipeline(self, sample_train_data):
        """Test data quality checks throughout pipeline."""
        df = sample_train_data.copy()

        # Check no nulls in critical columns
        assert df['Store'].isnull().sum() == 0
        assert df['Sales'].isnull().sum() == 0

        # Check valid ranges
        assert (df['Sales'] >= 0).all()
        assert (df['Customers'] >= 0).all()
        assert (df['DayOfWeek'] >= 1).all() and (df['DayOfWeek'] <= 7).all()

        # Check binary features
        assert (df['Open'].isin([0, 1])).all()
        assert (df['Promo'].isin([0, 1])).all()


class TestTrainingPipeline:
    """Test model training pipeline."""

    def test_prepare_training_data(self, sample_train_data):
        """Test preparation of training data."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        feature_cols = ['Customers', 'Promo', 'DayOfWeek']
        target_col = 'Sales'

        X = df[feature_cols]
        y = df[target_col]

        assert len(X) == len(y)
        assert X.isnull().sum().sum() == 0
        assert y.isnull().sum() == 0

    def test_train_test_split_pipeline(self, sample_train_data):
        """Test train-test split in pipeline."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()
        df = df.sort_values('Date').reset_index(drop=True)

        feature_cols = ['Customers', 'Promo', 'DayOfWeek']
        target_col = 'Sales'

        split_idx = int(len(df) * 0.8)
        X_train = df.iloc[:split_idx][feature_cols]
        X_test = df.iloc[split_idx:][feature_cols]
        y_train = df.iloc[:split_idx][target_col]
        y_test = df.iloc[split_idx:][target_col]

        assert len(X_train) + len(X_test) == len(df)
        assert len(y_train) + len(y_test) == len(df)

    def test_model_training_pipeline(self, sample_train_data):
        """Test complete model training pipeline."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        feature_cols = ['Customers', 'Promo', 'DayOfWeek']
        X = df[feature_cols].values
        y = df['Sales'].values

        # Train linear regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)

        # Train random forest
        rf_model = RandomForestRegressor(n_estimators=5, random_state=42)
        rf_model.fit(X, y)

        assert lr_model is not None
        assert rf_model is not None

    def test_model_evaluation_pipeline(self, sample_train_data):
        """Test model evaluation pipeline."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        assert mae > 0
        assert -1 <= r2 <= 1


class TestPredictionPipeline:
    """Test prediction pipeline."""

    def test_make_predictions_on_test_set(self, sample_train_data, sample_test_data):
        """Test making predictions on test set."""
        # Train on full training data
        df_train = sample_train_data[sample_train_data['Open'] == 1].copy()
        X_train = df_train[['Customers', 'Promo', 'DayOfWeek']].values
        y_train = df_train['Sales'].values

        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on test data
        X_test = sample_test_data[['Customers', 'Promo', 'DayOfWeek']].values
        predictions = model.predict(X_test)

        assert len(predictions) == len(sample_test_data)
        assert not np.isnan(predictions).any()

    def test_create_submission_format(self, sample_test_data):
        """Test creating submission in correct format."""
        df_test = sample_test_data.copy()

        # Create predictions (dummy)
        predictions = np.random.randint(1000, 10000, len(df_test))

        # Create submission dataframe
        submission = pd.DataFrame({
            'Id': range(len(df_test)),
            'Sales': predictions
        })

        assert len(submission) == len(df_test)
        assert 'Sales' in submission.columns
        assert submission['Sales'].isnull().sum() == 0

    def test_prediction_postprocessing(self, sample_train_data):
        """Test prediction postprocessing."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)

        # Clip negative predictions
        predictions = np.maximum(predictions, 0)

        assert (predictions >= 0).all()
        assert not np.isnan(predictions).any()


class TestMultiStoreScenarios:
    """Test pipeline with multiple stores."""

    def test_pipeline_with_multiple_stores(self, sample_train_data):
        """Test pipeline handles multiple stores correctly."""
        df = sample_train_data.copy()

        # Process each store separately
        store_models = {}
        for store_id in df['Store'].unique():
            store_data = df[df['Store'] == store_id]

            if len(store_data[store_data['Open'] == 1]) > 5:
                X = store_data[store_data['Open'] == 1][['Customers', 'Promo']].values
                y = store_data[store_data['Open'] == 1]['Sales'].values

                model = LinearRegression()
                model.fit(X, y)
                store_models[store_id] = model

        assert len(store_models) > 0

    def test_store_specific_predictions(self, sample_train_data):
        """Test making store-specific predictions."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        # Train global model
        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        # Make predictions for each store
        for store_id in df['Store'].unique():
            store_data = df[df['Store'] == store_id]
            X_store = store_data[['Customers', 'Promo', 'DayOfWeek']].values

            if len(X_store) > 0:
                predictions = model.predict(X_store)
                assert len(predictions) == len(store_data)

    def test_aggregated_metrics_by_store(self, sample_train_data):
        """Test computing aggregated metrics by store."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        model = LinearRegression()
        model.fit(X, y)

        df['predictions'] = model.predict(X)

        # Compute metrics per store
        store_metrics = []
        for store_id in df['Store'].unique():
            store_df = df[df['Store'] == store_id]
            mae = mean_absolute_error(store_df['Sales'], store_df['predictions'])
            r2 = r2_score(store_df['Sales'], store_df['predictions'])

            store_metrics.append({
                'Store': store_id,
                'MAE': mae,
                'R2': r2
            })

        assert len(store_metrics) > 0


class TestEndToEndIntegration:
    """Test complete end-to-end pipeline."""

    def test_complete_pipeline_execution(self, sample_train_data, sample_store_data, sample_test_data):
        """Test running complete pipeline from start to finish."""
        # 1. Load and merge data
        df_train = sample_train_data.merge(sample_store_data, on='Store', how='left')

        # 2. Preprocessing
        df_train['Month'] = df_train['Date'].dt.month
        df_train['CompetitionDistance'].fillna(df_train['CompetitionDistance'].median(), inplace=True)

        # 3. Filter to open stores
        df_train_open = df_train[df_train['Open'] == 1].copy()

        # 4. Feature selection
        features = ['Customers', 'Promo', 'DayOfWeek', 'Month']
        X_train = df_train_open[features].values
        y_train = df_train_open['Sales'].values

        # 5. Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. Prepare test data
        df_test = sample_test_data.copy()
        df_test['Month'] = df_test['Date'].dt.month
        X_test = df_test[features].values

        # 7. Make predictions
        predictions = model.predict(X_test)

        # 8. Postprocess predictions
        predictions = np.maximum(predictions, 0)

        # 9. Validate results
        assert len(predictions) == len(sample_test_data)
        assert (predictions >= 0).all()
        assert not np.isnan(predictions).any()

    def test_pipeline_error_handling(self, sample_train_data):
        """Test pipeline error handling."""
        df = sample_train_data.copy()

        # Handle case with no open stores in subset
        subset = df[df['Open'] == 0]
        assert len(subset) > 0, "Should have closed stores for testing"

        # Should handle gracefully
        try:
            if len(subset[subset['Open'] == 1]) == 0:
                # Can't train on no data, should handle
                pass
        except Exception as e:
            pytest.fail(f"Pipeline should handle missing open stores: {e}")

    def test_pipeline_reproducibility(self, sample_train_data):
        """Test that pipeline produces reproducible results."""
        df1 = sample_train_data.copy()
        df2 = sample_train_data.copy()

        X1 = df1[df1['Open'] == 1][['Customers', 'Promo']].values
        y1 = df1[df1['Open'] == 1]['Sales'].values

        X2 = df2[df2['Open'] == 1][['Customers', 'Promo']].values
        y2 = df2[df2['Open'] == 1]['Sales'].values

        model1 = LinearRegression()
        model1.fit(X1, y1)
        pred1 = model1.predict(X1[:5])

        model2 = LinearRegression()
        model2.fit(X2, y2)
        pred2 = model2.predict(X2[:5])

        np.testing.assert_array_almost_equal(pred1, pred2, \
            "Pipeline should be reproducible")

    def test_pipeline_with_feature_scaling(self, sample_train_data):
        """Test pipeline with feature scaling."""
        df = sample_train_data[sample_train_data['Open'] == 1].copy()

        X = df[['Customers', 'Promo', 'DayOfWeek']].values
        y = df['Sales'].values

        # Standardize features
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_scaled = (X - X_mean) / (X_std + 1e-10)

        model = LinearRegression()
        model.fit(X_scaled, y)

        predictions = model.predict(X_scaled)

        assert not np.isnan(predictions).any()
        assert len(predictions) == len(y)
