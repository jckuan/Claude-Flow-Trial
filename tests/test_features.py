"""Tests for feature engineering functions."""

import pytest
import pandas as pd
import numpy as np


class TestTimeSeriesFeatures:
    """Test time series feature engineering."""

    def test_lag_feature_generation(self, sample_train_data):
        """Test generation of lag features."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        # Create lag features
        for store in df['Store'].unique():
            store_data = df[df['Store'] == store]
            df.loc[store_data.index, 'Sales_lag1'] = \
                store_data['Sales'].shift(1).values

        assert 'Sales_lag1' in df.columns
        assert pd.isna(df['Sales_lag1'].iloc[0]), "First lag should be NaN"

    def test_rolling_mean_feature(self, sample_train_data):
        """Test rolling mean feature generation."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        # Create rolling mean for each store
        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            df.loc[store_mask, 'Sales_rolling_mean'] = \
                df.loc[store_mask, 'Sales'].rolling(window=7, min_periods=1).mean()

        assert 'Sales_rolling_mean' in df.columns
        assert df['Sales_rolling_mean'].notna().sum() > 0

    def test_exponential_moving_average(self, sample_train_data):
        """Test exponential moving average feature."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            df.loc[store_mask, 'Sales_ema'] = \
                df.loc[store_mask, 'Sales'].ewm(span=7).mean()

        assert 'Sales_ema' in df.columns
        assert df['Sales_ema'].notna().all()

    def test_differencing_feature(self, sample_train_data):
        """Test differencing for stationarity."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            df.loc[store_mask, 'Sales_diff'] = \
                df.loc[store_mask, 'Sales'].diff()

        assert 'Sales_diff' in df.columns
        # First value should be NaN after differencing
        assert pd.isna(df.groupby('Store')['Sales_diff'].first().iloc[0])

    def test_seasonal_feature(self, sample_train_data):
        """Test seasonal feature engineering."""
        df = sample_train_data.copy()
        df['Month'] = df['Date'].dt.month

        # Create seasonal indicators
        df['is_q1'] = (df['Month'] <= 3).astype(int)
        df['is_q2'] = ((df['Month'] > 3) & (df['Month'] <= 6)).astype(int)
        df['is_q3'] = ((df['Month'] > 6) & (df['Month'] <= 9)).astype(int)
        df['is_q4'] = (df['Month'] > 9).astype(int)

        assert df['is_q1'].sum() > 0
        assert (df[['is_q1', 'is_q2', 'is_q3', 'is_q4']].sum(axis=1) == 1).all()

    def test_trend_feature(self, sample_train_data):
        """Test trend feature engineering."""
        df = sample_train_data.copy()
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

        # Create trend as days since start for each store
        for store in df['Store'].unique():
            store_mask = df['Store'] == store
            store_indices = df[store_mask].index
            df.loc[store_indices, 'trend'] = \
                np.arange(store_mask.sum())

        assert 'trend' in df.columns
        assert df['trend'].min() == 0


class TestInteractionFeatures:
    """Test interaction feature engineering."""

    def test_promo_day_interaction(self, sample_train_data):
        """Test interaction between promo and day of week."""
        df = sample_train_data.copy()
        df['promo_dow'] = df['Promo'] * df['DayOfWeek']

        assert 'promo_dow' in df.columns
        assert (df['promo_dow'] >= 0).all()
        assert df['promo_dow'].max() <= 7

    def test_open_promo_interaction(self, sample_train_data):
        """Test interaction between open status and promo."""
        df = sample_train_data.copy()
        df['open_promo'] = df['Open'] * df['Promo']

        assert 'open_promo' in df.columns
        assert (df['open_promo'].isin([0, 1])).all()

    def test_holiday_promo_interaction(self, sample_train_data):
        """Test interaction between holidays and promo."""
        df = sample_train_data.copy()
        df['holiday'] = (df['StateHoliday'] != '0').astype(int)
        df['holiday_promo'] = df['holiday'] * df['Promo']

        assert 'holiday_promo' in df.columns
        assert (df['holiday_promo'].isin([0, 1])).all()

    def test_customer_store_type_interaction(self, merged_data):
        """Test interaction between customers and store type."""
        df = merged_data.copy()

        # One-hot encode store type and interact with customers
        store_types = pd.get_dummies(df['StoreType'], prefix='StoreType')
        for col in store_types.columns:
            df[f'customers_{col}'] = df['Customers'] * store_types[col]

        new_cols = [col for col in df.columns if col.startswith('customers_')]
        assert len(new_cols) > 0


class TestCompetitionFeatures:
    """Test competition-related feature engineering."""

    def test_competition_days_since_open(self, sample_store_data):
        """Test calculation of days since competition opened."""
        df = sample_store_data.copy()

        # Calculate days since competition opened (simplified)
        current_year = 2015
        current_month = 7

        df['days_since_comp_open'] = \
            (current_year - df['CompetitionOpenSinceYear']) * 365 + \
            (current_month - df['CompetitionOpenSinceMonth']) * 30

        assert 'days_since_comp_open' in df.columns
        assert (df['days_since_comp_open'] >= 0).all() or \
               df['days_since_comp_open'].isnull().any()

    def test_competition_distance_categories(self, sample_store_data):
        """Test categorization of competition distance."""
        df = sample_store_data.copy()

        # Create distance categories
        df['comp_dist_category'] = pd.cut(
            df['CompetitionDistance'].fillna(df['CompetitionDistance'].median()),
            bins=[0, 500, 1000, 5000, 30000, np.inf],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        )

        assert 'comp_dist_category' in df.columns
        assert df['comp_dist_category'].notna().all()

    def test_has_competition_flag(self, sample_store_data):
        """Test creation of competition existence flag."""
        df = sample_store_data.copy()
        df['has_competition'] = \
            (~df['CompetitionDistance'].isnull()).astype(int)

        assert 'has_competition' in df.columns
        assert (df['has_competition'].isin([0, 1])).all()


class TestPromo2Features:
    """Test Promo2 feature engineering."""

    def test_promo2_participation_flag(self, sample_store_data):
        """Test Promo2 participation indicator."""
        df = sample_store_data.copy()
        df['active_promo2'] = df['Promo2'].copy()

        assert 'active_promo2' in df.columns
        assert (df['active_promo2'].isin([0, 1])).all()

    def test_promo2_duration(self, sample_store_data):
        """Test calculation of Promo2 duration."""
        df = sample_store_data.copy()

        current_year = 2015
        current_week = 31

        df['promo2_weeks_active'] = \
            (current_year - df['Promo2SinceYear']) * 52 + \
            (current_week - df['Promo2SinceWeek'])

        # Fill negative values (future dates) with 0
        df.loc[df['promo2_weeks_active'] < 0, 'promo2_weeks_active'] = 0
        df.loc[df['Promo2'] == 0, 'promo2_weeks_active'] = 0

        assert 'promo2_weeks_active' in df.columns
        assert (df['promo2_weeks_active'] >= 0).all()


class TestStoreCharacteristicFeatures:
    """Test store characteristic feature engineering."""

    def test_store_type_encoding(self, sample_store_data):
        """Test store type one-hot encoding."""
        df = sample_store_data.copy()
        encoded = pd.get_dummies(df['StoreType'], prefix='StoreType')

        assert encoded.shape[0] == len(df)
        assert (encoded.sum(axis=1) == 1).all()

    def test_assortment_encoding(self, sample_store_data):
        """Test assortment one-hot encoding."""
        df = sample_store_data.copy()
        encoded = pd.get_dummies(df['Assortment'], prefix='Assortment')

        assert encoded.shape[0] == len(df)
        assert (encoded.sum(axis=1) == 1).all()

    def test_promo_interval_parsing(self, sample_store_data):
        """Test parsing of promo interval."""
        df = sample_store_data.copy()

        df['promo_in_jan'] = df['PromoInterval'].fillna('').str.contains('Jan').astype(int)
        df['promo_in_apr'] = df['PromoInterval'].fillna('').str.contains('Apr').astype(int)
        df['promo_in_jul'] = df['PromoInterval'].fillna('').str.contains('Jul').astype(int)
        df['promo_in_oct'] = df['PromoInterval'].fillna('').str.contains('Oct').astype(int)

        assert df['promo_in_jan'].sum() >= 0
        assert df['promo_in_apr'].sum() >= 0


class TestFeatureValidation:
    """Test validation of generated features."""

    def test_no_infinite_values_in_features(self, sample_train_data):
        """Test that features don't contain infinite values."""
        df = sample_train_data.copy()

        # Add some engineered features
        df['sales_per_customer'] = df['Sales'] / (df['Customers'] + 1)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not np.isinf(df[col]).any(), f"Column {col} contains infinite values"

    def test_no_nan_in_critical_features(self, sample_train_data):
        """Test that critical features don't have NaNs."""
        df = sample_train_data.copy()

        critical_cols = ['Store', 'Date', 'Sales', 'Open', 'Promo']
        for col in critical_cols:
            assert df[col].isnull().sum() == 0, \
                f"Critical column {col} has null values"

    def test_feature_value_ranges(self, sample_train_data):
        """Test that features are within expected ranges."""
        df = sample_train_data.copy()
        df['Month'] = df['Date'].dt.month

        # Month should be 1-12
        assert (df['Month'] >= 1).all() and (df['Month'] <= 12).all()

        # DayOfWeek should be 1-7
        assert (df['DayOfWeek'] >= 1).all() and (df['DayOfWeek'] <= 7).all()

        # Binary features should be 0 or 1
        for col in ['Open', 'Promo', 'SchoolHoliday']:
            assert (df[col].isin([0, 1])).all()

    def test_feature_cardinality(self, sample_train_data):
        """Test reasonable cardinality of categorical features."""
        df = sample_train_data.copy()

        # Store should have limited unique values
        assert df['Store'].nunique() < 2000

        # DayOfWeek should have exactly 7 values
        assert df['DayOfWeek'].nunique() == 7

        # StateHoliday should have limited values
        assert df['StateHoliday'].nunique() <= 5
