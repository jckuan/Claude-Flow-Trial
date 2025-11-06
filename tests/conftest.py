"""Pytest configuration and shared fixtures."""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_train_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2013-01-01', periods=1000, freq='D')

    data = {
        'Store': np.tile(np.arange(1, 6), 200),  # 5 stores
        'Date': np.repeat(dates, 1),
        'DayOfWeek': np.tile(np.arange(1, 8), 143)[:1000],
        'Sales': np.random.randint(0, 20000, 1000),
        'Customers': np.random.randint(0, 1000, 1000),
        'Open': np.random.choice([0, 1], 1000, p=[0.17, 0.83]),
        'Promo': np.random.choice([0, 1], 1000, p=[0.62, 0.38]),
        'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], 1000, p=[0.95, 0.03, 0.01, 0.01]),
        'SchoolHoliday': np.random.choice([0, 1], 1000, p=[0.82, 0.18]),
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['Store', 'Date']).reset_index(drop=True)


@pytest.fixture
def sample_store_data():
    """Create sample store metadata for testing."""
    np.random.seed(42)

    data = {
        'Store': np.arange(1, 6),
        'StoreType': np.random.choice(['a', 'b', 'c', 'd'], 5),
        'Assortment': np.random.choice(['a', 'b', 'c'], 5),
        'CompetitionDistance': np.random.uniform(100, 30000, 5),
        'CompetitionOpenSinceMonth': np.random.choice([1, 3, 6, 9, 12], 5),
        'CompetitionOpenSinceYear': np.random.choice([2006, 2008, 2010, 2012, 2014], 5),
        'Promo2': np.random.choice([0, 1], 5),
        'Promo2SinceWeek': np.random.choice([1, 10, 20, 30, 40], 5),
        'Promo2SinceYear': np.random.choice([2009, 2011, 2013, 2015], 5),
        'PromoInterval': np.random.choice(['Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', np.nan], 5),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_test_data():
    """Create sample test data (without target variable)."""
    np.random.seed(42)
    dates = pd.date_range('2015-08-01', periods=100, freq='D')

    data = {
        'Store': np.tile(np.arange(1, 6), 20),
        'Date': np.repeat(dates, 1),
        'DayOfWeek': np.tile(np.arange(1, 8), 15)[:100],
        'Customers': np.random.randint(100, 1000, 100),
        'Open': np.random.choice([0, 1], 100, p=[0.1, 0.9]),
        'Promo': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
        'StateHoliday': np.random.choice(['0', 'a'], 100, p=[0.98, 0.02]),
        'SchoolHoliday': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values(['Store', 'Date']).reset_index(drop=True)


@pytest.fixture
def merged_data(sample_train_data, sample_store_data):
    """Create merged training data with store information."""
    return sample_train_data.merge(sample_store_data, on='Store', how='left')
