"""
Unit tests for data loading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader, load_data
from utils.config import config


class TestDataLoader:
    """Test cases for DataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a DataLoader instance."""
        return DataLoader()

    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader.config is not None
        assert loader.train is None
        assert loader.test is None
        assert loader.store is None

    def test_load_train(self, loader):
        """Test loading training data."""
        train = loader.load_train()

        assert isinstance(train, pd.DataFrame)
        assert not train.empty
        assert "Store" in train.columns
        assert "Date" in train.columns
        assert "Sales" in train.columns

    def test_load_test(self, loader):
        """Test loading test data."""
        test = loader.load_test()

        assert isinstance(test, pd.DataFrame)
        assert not test.empty
        assert "Store" in test.columns
        assert "Date" in test.columns

    def test_load_store(self, loader):
        """Test loading store data."""
        store = loader.load_store()

        assert isinstance(store, pd.DataFrame)
        assert not store.empty
        assert "Store" in store.columns
        assert "StoreType" in store.columns
        assert "Assortment" in store.columns

    def test_load_all(self, loader):
        """Test loading all datasets."""
        train, test, store = loader.load_all()

        assert all(isinstance(df, pd.DataFrame) for df in [train, test, store])
        assert all(not df.empty for df in [train, test, store])

    def test_merge_with_store(self, loader):
        """Test merging transaction data with store metadata."""
        train = loader.load_train()
        merged = loader.merge_with_store(train)

        assert isinstance(merged, pd.DataFrame)
        assert "StoreType" in merged.columns
        assert "Assortment" in merged.columns
        assert len(merged) == len(train)

    def test_data_validation_train(self, loader):
        """Test that training data passes validation."""
        train = loader.load_train()

        # Check required columns
        assert all(col in train.columns for col in ["Store", "Date", "Sales"])

        # Check data types
        assert train["Store"].dtype in [np.int32, np.int64]
        assert train["Sales"].dtype in [np.int32, np.int64, np.float32, np.float64]

    def test_data_validation_test(self, loader):
        """Test that test data passes validation."""
        test = loader.load_test()

        # Check required columns
        assert all(col in test.columns for col in ["Store", "Date"])

    def test_data_validation_store(self, loader):
        """Test that store data passes validation."""
        store = loader.load_store()

        # Check required columns
        assert "Store" in store.columns

        # Check unique stores
        assert store["Store"].is_unique

    def test_get_data_summary(self, loader):
        """Test data summary generation."""
        loader.load_all()
        summary = loader.get_data_summary()

        assert isinstance(summary, dict)
        assert "train" in summary
        assert "test" in summary
        assert "store" in summary

        # Check train summary
        assert "shape" in summary["train"]
        assert "memory_mb" in summary["train"]
        assert "date_range" in summary["train"]

    def test_convenience_function(self):
        """Test the load_data convenience function."""
        train, test, store = load_data()

        assert all(isinstance(df, pd.DataFrame) for df in [train, test, store])
        assert all(not df.empty for df in [train, test, store])


class TestDataValidation:
    """Test data validation logic."""

    def test_train_no_missing_values(self):
        """Test that training data has no missing values."""
        loader = DataLoader()
        train = loader.load_train()

        # Training data should have no missing values
        missing_count = train.isnull().sum().sum()
        assert missing_count == 0, f"Found {missing_count} missing values in training data"

    def test_store_expected_missing(self):
        """Test that store data has expected missing values."""
        loader = DataLoader()
        store = loader.load_store()

        # Store data is expected to have some missing values
        # in competition and promo2 columns
        assert store.isnull().sum().sum() > 0

    def test_no_duplicates(self):
        """Test that there are no duplicate rows."""
        loader = DataLoader()
        train, test, store = loader.load_all()

        assert not train.duplicated().any()
        assert not test.duplicated().any()
        assert not store.duplicated().any()

    def test_store_ids_valid(self):
        """Test that store IDs are valid and consistent."""
        loader = DataLoader()
        train, test, store = loader.load_all()

        # All store IDs in train/test should be in store data
        train_stores = set(train["Store"].unique())
        test_stores = set(test["Store"].unique())
        store_stores = set(store["Store"].unique())

        assert train_stores.issubset(store_stores)
        assert test_stores.issubset(store_stores)

    def test_date_ranges(self):
        """Test that date ranges are logical."""
        loader = DataLoader()
        train, test, _ = loader.load_all()

        # Convert dates
        train["Date"] = pd.to_datetime(train["Date"])
        test["Date"] = pd.to_datetime(test["Date"])

        # Test data should be after training data
        assert test["Date"].min() > train["Date"].max()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
