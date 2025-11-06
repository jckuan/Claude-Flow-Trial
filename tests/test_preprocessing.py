"""Tests for data preprocessing functions."""

import pytest
import pandas as pd
import numpy as np


class TestDateFeatureExtraction:
    """Test date feature extraction."""

    def test_date_to_datetime_conversion(self, sample_train_data):
        """Test that Date column can be converted to datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_train_data['Date']), \
            "Date should be datetime type"

    def test_extract_year_feature(self, sample_train_data):
        """Test year extraction from date."""
        sample_train_data['Year'] = sample_train_data['Date'].dt.year

        assert 'Year' in sample_train_data.columns
        assert sample_train_data['Year'].min() >= 2013
        assert sample_train_data['Year'].max() <= 2015

    def test_extract_month_feature(self, sample_train_data):
        """Test month extraction from date."""
        sample_train_data['Month'] = sample_train_data['Date'].dt.month

        assert 'Month' in sample_train_data.columns
        assert sample_train_data['Month'].min() >= 1
        assert sample_train_data['Month'].max() <= 12

    def test_extract_day_feature(self, sample_train_data):
        """Test day extraction from date."""
        sample_train_data['Day'] = sample_train_data['Date'].dt.day

        assert 'Day' in sample_train_data.columns
        assert sample_train_data['Day'].min() >= 1
        assert sample_train_data['Day'].max() <= 31

    def test_extract_day_of_week_feature(self, sample_train_data):
        """Test day of week extraction."""
        sample_train_data['DayOfWeek_dt'] = sample_train_data['Date'].dt.dayofweek + 1

        assert sample_train_data['DayOfWeek_dt'].min() >= 1
        assert sample_train_data['DayOfWeek_dt'].max() <= 7

    def test_extract_week_of_year_feature(self, sample_train_data):
        """Test week of year extraction."""
        sample_train_data['WeekOfYear'] = sample_train_data['Date'].dt.isocalendar().week

        assert sample_train_data['WeekOfYear'].min() >= 1
        assert sample_train_data['WeekOfYear'].max() <= 53


class TestMissingValueHandling:
    """Test handling of missing values."""

    def test_missing_values_in_competition_data(self, sample_store_data):
        """Test detection of missing competition data."""
        missing_comp_month = sample_store_data['CompetitionOpenSinceMonth'].isnull().sum()
        missing_comp_year = sample_store_data['CompetitionOpenSinceYear'].isnull().sum()

        # Should have some missing values
        assert missing_comp_month >= 0, "Should detect missing competition month values"
        assert missing_comp_year >= 0, "Should detect missing competition year values"

    def test_missing_values_in_promo2_data(self, sample_store_data):
        """Test detection of missing Promo2 data."""
        missing_promo2_week = sample_store_data['Promo2SinceWeek'].isnull().sum()
        missing_promo2_year = sample_store_data['Promo2SinceYear'].isnull().sum()
        missing_promo_interval = sample_store_data['PromoInterval'].isnull().sum()

        assert missing_promo2_week >= 0, "Should detect missing Promo2 week values"
        assert missing_promo2_year >= 0, "Should detect missing Promo2 year values"
        assert missing_promo_interval >= 0, "Should detect missing PromoInterval values"

    def test_fill_missing_competition_distance(self, sample_store_data):
        """Test filling missing competition distance."""
        original_missing = sample_store_data['CompetitionDistance'].isnull().sum()

        # Fill with median
        filled_data = sample_store_data.copy()
        median_dist = filled_data['CompetitionDistance'].median()
        filled_data['CompetitionDistance'].fillna(median_dist, inplace=True)

        assert filled_data['CompetitionDistance'].isnull().sum() == 0, \
            "All missing competition distances should be filled"

    def test_fill_missing_competition_dates(self, sample_store_data):
        """Test filling missing competition date fields."""
        filled_data = sample_store_data.copy()

        # Fill with 0 (no competition)
        filled_data['CompetitionOpenSinceMonth'].fillna(0, inplace=True)
        filled_data['CompetitionOpenSinceYear'].fillna(0, inplace=True)

        assert filled_data['CompetitionOpenSinceMonth'].isnull().sum() == 0
        assert filled_data['CompetitionOpenSinceYear'].isnull().sum() == 0

    def test_fill_missing_promo2_fields(self, sample_store_data):
        """Test filling missing Promo2 fields."""
        filled_data = sample_store_data.copy()

        # Fill with 0/dummy values
        filled_data['Promo2SinceWeek'].fillna(0, inplace=True)
        filled_data['Promo2SinceYear'].fillna(0, inplace=True)
        filled_data['PromoInterval'].fillna('None', inplace=True)

        assert filled_data['Promo2SinceWeek'].isnull().sum() == 0
        assert filled_data['Promo2SinceYear'].isnull().sum() == 0
        assert filled_data['PromoInterval'].isnull().sum() == 0


class TestEncodingCategorical:
    """Test encoding of categorical variables."""

    def test_encode_store_type(self, sample_store_data):
        """Test encoding of StoreType."""
        encoded = pd.Categorical(sample_store_data['StoreType']).codes

        assert len(encoded) == len(sample_store_data)
        assert encoded.min() >= 0, "Encoded values should be non-negative"

    def test_encode_assortment(self, sample_store_data):
        """Test encoding of Assortment."""
        encoded = pd.Categorical(sample_store_data['Assortment']).codes

        assert len(encoded) == len(sample_store_data)
        assert encoded.min() >= 0

    def test_encode_state_holiday(self, sample_train_data):
        """Test encoding of StateHoliday."""
        encoded = pd.Categorical(sample_train_data['StateHoliday']).codes

        assert len(encoded) == len(sample_train_data)
        assert encoded.min() >= 0

    def test_one_hot_encode_state_holiday(self, sample_train_data):
        """Test one-hot encoding of StateHoliday."""
        encoded = pd.get_dummies(sample_train_data['StateHoliday'],
                                 prefix='StateHoliday')

        assert encoded.shape[0] == len(sample_train_data)
        assert encoded.sum().sum() == len(sample_train_data), \
            "One-hot encoded values should sum to dataset length"

    def test_one_hot_encode_store_type(self, sample_store_data):
        """Test one-hot encoding of StoreType."""
        encoded = pd.get_dummies(sample_store_data['StoreType'],
                                 prefix='StoreType')

        assert encoded.shape[0] == len(sample_store_data)
        assert all((encoded.sum(axis=1) == 1).values), \
            "Each row should have exactly one hot value"


class TestNormalization:
    """Test normalization and scaling."""

    def test_minmax_scaling(self, sample_train_data):
        """Test min-max scaling."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        min_val = sales.min()
        max_val = sales.max()
        scaled = (sales - min_val) / (max_val - min_val)

        assert scaled.min() >= 0, "Min-max scaled values should be >= 0"
        assert scaled.max() <= 1, "Min-max scaled values should be <= 1"

    def test_standardization_scaling(self, sample_train_data):
        """Test standardization (z-score) scaling."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        mean = sales.mean()
        std = sales.std()
        standardized = (sales - mean) / std

        assert abs(standardized.mean()) < 0.1, "Mean of standardized data should be ~0"
        assert abs(standardized.std() - 1.0) < 0.1, "Std of standardized data should be ~1"

    def test_log_transformation(self, sample_train_data):
        """Test log transformation for right-skewed data."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        log_sales = np.log1p(sales)  # log1p to handle zeros

        assert (log_sales >= 0).all(), "Log-transformed values should be non-negative"
        assert log_sales.std() < sales.std(), \
            "Log transformation should reduce variance for skewed data"


class TestOutlierDetection:
    """Test outlier detection methods."""

    def test_iqr_outlier_detection(self, sample_train_data):
        """Test IQR-based outlier detection."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        Q1 = sales.quantile(0.25)
        Q3 = sales.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = sales[(sales < lower_bound) | (sales > upper_bound)]

        # Should have some outliers but not too many
        outlier_pct = len(outliers) / len(sales) * 100
        assert outlier_pct < 10, "Outlier percentage should be reasonable"

    def test_zscore_outlier_detection(self, sample_train_data):
        """Test z-score based outlier detection."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        mean = sales.mean()
        std = sales.std()
        z_scores = np.abs((sales - mean) / std)

        outliers = z_scores > 3
        outlier_pct = outliers.sum() / len(sales) * 100

        assert outlier_pct < 1, "Z-score outliers should be < 1%"

    def test_high_value_outliers(self, sample_train_data):
        """Test identification of high-value outliers."""
        sales = sample_train_data[sample_train_data['Sales'] > 0]['Sales']

        p99 = sales.quantile(0.99)
        high_outliers = sales > p99

        assert high_outliers.sum() > 0, "Should have high-value outliers"
        assert high_outliers.sum() / len(sales) < 0.02, \
            "High-value outliers should be < 2% of data"


class TestDataSplitting:
    """Test train/validation/test data splitting."""

    def test_temporal_split_train_test(self, sample_train_data):
        """Test temporal split for time series data."""
        split_date = sample_train_data['Date'].quantile(0.8)

        train = sample_train_data[sample_train_data['Date'] <= split_date]
        test = sample_train_data[sample_train_data['Date'] > split_date]

        assert len(train) + len(test) == len(sample_train_data), \
            "Train and test should cover all data"
        assert len(train) > len(test), "Train set should be larger than test set"
        assert train['Date'].max() <= test['Date'].min(), \
            "Train dates should be before test dates"

    def test_stratified_split_by_store(self, sample_train_data):
        """Test stratified split to maintain store distribution."""
        stores = sample_train_data['Store'].unique()
        np.random.seed(42)
        split_idx = int(len(sample_train_data) * 0.8)

        train = sample_train_data.iloc[:split_idx]
        test = sample_train_data.iloc[split_idx:]

        train_stores = set(train['Store'].unique())
        test_stores = set(test['Store'].unique())

        # Both sets should have most stores represented
        assert len(train_stores) > len(stores) * 0.8, \
            "Train set should have most stores"
        assert len(test_stores) > len(stores) * 0.5, \
            "Test set should have reasonable store coverage"

    def test_validation_set_creation(self, sample_train_data):
        """Test creation of validation set."""
        train_size = int(len(sample_train_data) * 0.6)
        val_size = int(len(sample_train_data) * 0.2)

        train = sample_train_data.iloc[:train_size]
        val = sample_train_data.iloc[train_size:train_size + val_size]
        test = sample_train_data.iloc[train_size + val_size:]

        assert len(train) + len(val) + len(test) == len(sample_train_data), \
            "All splits should cover entire dataset"
        assert len(train) > len(val) > len(test) or \
               (len(train) >= len(val) and len(val) >= len(test)), \
            "Train should be largest, then validation, then test"
