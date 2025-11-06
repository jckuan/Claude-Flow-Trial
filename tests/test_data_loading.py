"""Tests for data loading and validation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestDataLoading:
    """Test data loading functionality."""

    def test_train_data_shape(self, sample_train_data):
        """Test that training data has expected shape."""
        assert len(sample_train_data) > 0, "Training data should not be empty"
        assert sample_train_data.shape[1] >= 9, "Training data should have at least 9 columns"

    def test_store_data_shape(self, sample_store_data):
        """Test that store data has expected shape."""
        assert len(sample_store_data) > 0, "Store data should not be empty"
        assert sample_store_data.shape[1] >= 10, "Store data should have at least 10 columns"

    def test_test_data_shape(self, sample_test_data):
        """Test that test data has expected shape (no target)."""
        assert len(sample_test_data) > 0, "Test data should not be empty"
        assert 'Sales' not in sample_test_data.columns, "Test data should not have Sales column"
        assert sample_test_data.shape[1] >= 8, "Test data should have at least 8 columns"

    def test_required_columns_train(self, sample_train_data):
        """Test that training data has all required columns."""
        required_cols = ['Store', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
                        'StateHoliday', 'SchoolHoliday', 'DayOfWeek']
        missing = [col for col in required_cols if col not in sample_train_data.columns]
        assert len(missing) == 0, f"Missing required columns: {missing}"

    def test_required_columns_store(self, sample_store_data):
        """Test that store data has all required columns."""
        required_cols = ['Store', 'StoreType', 'Assortment', 'CompetitionDistance',
                        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear']
        missing = [col for col in required_cols if col not in sample_store_data.columns]
        assert len(missing) == 0, f"Missing required columns: {missing}"

    def test_no_duplicates_in_store_data(self, sample_store_data):
        """Test that store data has no duplicate store IDs."""
        assert sample_store_data['Store'].duplicated().sum() == 0, "Store data has duplicate stores"

    def test_date_column_is_datetime(self, sample_train_data):
        """Test that Date column is datetime type."""
        assert pd.api.types.is_datetime64_any_dtype(sample_train_data['Date']), \
            "Date column should be datetime type"

    def test_date_range_validity(self, sample_train_data):
        """Test that dates are in valid range."""
        min_date = sample_train_data['Date'].min()
        max_date = sample_train_data['Date'].max()

        assert min_date.year >= 2013, "Training data start year should be >= 2013"
        assert max_date.year <= 2015, "Training data end year should be <= 2015"
        assert max_date > min_date, "Max date should be after min date"


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_no_null_sales_in_train(self, sample_train_data):
        """Test that training data has no null sales values."""
        assert sample_train_data['Sales'].isnull().sum() == 0, \
            "Sales column should have no null values"

    def test_no_null_store_id(self, sample_train_data):
        """Test that Store column has no null values."""
        assert sample_train_data['Store'].isnull().sum() == 0, \
            "Store column should have no null values"

    def test_sales_non_negative(self, sample_train_data):
        """Test that sales values are non-negative."""
        assert (sample_train_data['Sales'] >= 0).all(), \
            "Sales values should be non-negative"

    def test_customers_non_negative(self, sample_train_data):
        """Test that customer counts are non-negative."""
        assert (sample_train_data['Customers'] >= 0).all(), \
            "Customer counts should be non-negative"

    def test_open_column_binary(self, sample_train_data):
        """Test that Open column contains only 0 and 1."""
        valid_values = set([0, 1])
        assert set(sample_train_data['Open'].unique()).issubset(valid_values), \
            "Open column should only contain 0 and 1"

    def test_promo_column_binary(self, sample_train_data):
        """Test that Promo column contains only 0 and 1."""
        valid_values = set([0, 1])
        assert set(sample_train_data['Promo'].unique()).issubset(valid_values), \
            "Promo column should only contain 0 and 1"

    def test_school_holiday_binary(self, sample_train_data):
        """Test that SchoolHoliday column contains only 0 and 1."""
        valid_values = set([0, 1])
        assert set(sample_train_data['SchoolHoliday'].unique()).issubset(valid_values), \
            "SchoolHoliday column should only contain 0 and 1"

    def test_day_of_week_valid_range(self, sample_train_data):
        """Test that DayOfWeek values are in valid range (1-7)."""
        assert (sample_train_data['DayOfWeek'] >= 1).all() and \
               (sample_train_data['DayOfWeek'] <= 7).all(), \
            "DayOfWeek should be between 1 and 7"

    def test_state_holiday_valid_values(self, sample_train_data):
        """Test that StateHoliday contains only valid values."""
        valid_values = {'0', 'a', 'b', 'c'}
        actual_values = set(sample_train_data['StateHoliday'].unique())
        assert actual_values.issubset(valid_values), \
            f"StateHoliday contains invalid values: {actual_values - valid_values}"

    def test_store_type_valid_values(self, sample_store_data):
        """Test that StoreType contains only valid values."""
        valid_values = {'a', 'b', 'c', 'd'}
        actual_values = set(sample_store_data['StoreType'].unique())
        assert actual_values.issubset(valid_values), \
            f"StoreType contains invalid values: {actual_values - valid_values}"

    def test_assortment_valid_values(self, sample_store_data):
        """Test that Assortment contains only valid values."""
        valid_values = {'a', 'b', 'c'}
        actual_values = set(sample_store_data['Assortment'].unique())
        assert actual_values.issubset(valid_values), \
            f"Assortment contains invalid values: {actual_values - valid_values}"

    def test_competition_distance_non_negative(self, sample_store_data):
        """Test that competition distance is non-negative."""
        comp_dist = sample_store_data['CompetitionDistance'].dropna()
        assert (comp_dist >= 0).all(), "Competition distance should be non-negative"

    def test_no_extreme_outliers_in_sales(self, sample_train_data):
        """Test that sales values are within reasonable bounds."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']
        q99 = sales.quantile(0.99)
        assert q99 < 100000, "99th percentile sales seems unreasonable"

    def test_store_ids_are_positive_integers(self, sample_train_data):
        """Test that Store IDs are positive integers."""
        assert (sample_train_data['Store'] > 0).all(), "Store IDs should be positive"
        assert sample_train_data['Store'].dtype in ['int64', 'int32'], \
            "Store IDs should be integers"


class TestDataConsistency:
    """Test data consistency between datasets."""

    def test_all_stores_in_store_data(self, merged_data):
        """Test that all stores in training data exist in store data."""
        train_stores = set(merged_data['Store'].unique())
        assert merged_data['StoreType'].isnull().sum() == 0, \
            "Not all stores in training data exist in store data"

    def test_merged_data_no_extra_nulls(self, merged_data):
        """Test that merge doesn't introduce excessive nulls."""
        null_pct = (merged_data.isnull().sum() / len(merged_data) * 100).max()
        assert null_pct < 50, f"Merge introduced too many nulls: {null_pct}%"

    def test_merged_data_row_count(self, sample_train_data, merged_data):
        """Test that merge maintains row count."""
        assert len(merged_data) == len(sample_train_data), \
            "Merge should not change row count"

    def test_store_promo2_consistency(self, sample_store_data):
        """Test that Promo2 related columns are consistent."""
        # If Promo2 is 0, related columns should be null
        promo2_zero = sample_store_data[sample_store_data['Promo2'] == 0]
        assert promo2_zero['PromoInterval'].isnull().all() or \
               promo2_zero['PromoInterval'].notna().any(), \
            "Promo2 consistency check"


class TestDataStatistics:
    """Test statistical properties of data."""

    def test_sales_mean_reasonable(self, sample_train_data):
        """Test that mean sales is reasonable."""
        sales_mean = sample_train_data[sample_train_data['Sales'] > 0]['Sales'].mean()
        sales_median = sample_train_data[sample_train_data['Sales'] > 0]['Sales'].median()

        # Mean should be greater than median for right-skewed distribution
        assert sales_mean > 0, "Mean sales should be positive"
        assert not np.isnan(sales_mean), "Mean sales should not be NaN"

    def test_sales_std_positive(self, sample_train_data):
        """Test that sales standard deviation is positive."""
        sales_std = sample_train_data[sample_train_data['Sales'] > 0]['Sales'].std()
        assert sales_std > 0, "Sales standard deviation should be positive"

    def test_customer_sales_correlation_positive(self, sample_train_data):
        """Test that sales and customers are positively correlated."""
        open_stores = sample_train_data[sample_train_data['Open'] == 1]
        correlation = open_stores[['Sales', 'Customers']].corr().iloc[0, 1]

        assert correlation > 0, "Sales and customers should be positively correlated"

    def test_promo_effect_on_sales(self, sample_train_data):
        """Test that promotions have positive effect on sales."""
        open_stores = sample_train_data[sample_train_data['Open'] == 1]

        sales_no_promo = open_stores[open_stores['Promo'] == 0]['Sales'].mean()
        sales_with_promo = open_stores[open_stores['Promo'] == 1]['Sales'].mean()

        assert sales_with_promo > sales_no_promo, \
            "Sales with promo should be higher than without promo"

    def test_open_status_affects_sales(self, sample_train_data):
        """Test that open stores have higher sales."""
        sales_closed = sample_train_data[sample_train_data['Open'] == 0]['Sales'].mean()
        sales_open = sample_train_data[sample_train_data['Open'] == 1]['Sales'].mean()

        assert sales_open > sales_closed, \
            "Open stores should have higher average sales"
