"""Script to create test fixtures."""

import pandas as pd
import numpy as np
from pathlib import Path

# Create fixtures directory
fixtures_dir = Path(__file__).parent
fixtures_dir.mkdir(exist_ok=True)


def create_sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    dates = pd.date_range('2013-01-01', periods=500, freq='D')

    data = {
        'Store': np.tile(np.arange(1, 6), 100),
        'Date': np.repeat(dates, 1),
        'DayOfWeek': np.tile(np.arange(1, 8), 72)[:500],
        'Sales': np.random.randint(0, 20000, 500),
        'Customers': np.random.randint(0, 1000, 500),
        'Open': np.random.choice([0, 1], 500, p=[0.17, 0.83]),
        'Promo': np.random.choice([0, 1], 500, p=[0.62, 0.38]),
        'StateHoliday': np.random.choice(['0', 'a', 'b', 'c'], 500, p=[0.95, 0.03, 0.01, 0.01]),
        'SchoolHoliday': np.random.choice([0, 1], 500, p=[0.82, 0.18]),
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    df.to_csv(fixtures_dir / 'sample_train.csv', index=False)
    print(f"Created sample_train.csv with {len(df)} records")


def create_sample_store_data():
    """Create sample store metadata."""
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
        'PromoInterval': ['Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', np.nan, 'Jan,Apr,Jul,Oct', np.nan],
    }

    df = pd.DataFrame(data)
    df.to_csv(fixtures_dir / 'sample_store.csv', index=False)
    print(f"Created sample_store.csv with {len(df)} records")


def create_sample_test_data():
    """Create sample test data."""
    np.random.seed(42)
    dates = pd.date_range('2015-08-01', periods=50, freq='D')

    data = {
        'Store': np.tile(np.arange(1, 6), 10),
        'Date': np.repeat(dates, 1),
        'DayOfWeek': np.tile(np.arange(1, 8), 8)[:50],
        'Customers': np.random.randint(100, 1000, 50),
        'Open': np.random.choice([0, 1], 50, p=[0.1, 0.9]),
        'Promo': np.random.choice([0, 1], 50, p=[0.6, 0.4]),
        'StateHoliday': np.random.choice(['0', 'a'], 50, p=[0.98, 0.02]),
        'SchoolHoliday': np.random.choice([0, 1], 50, p=[0.85, 0.15]),
    }

    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    df.to_csv(fixtures_dir / 'sample_test.csv', index=False)
    print(f"Created sample_test.csv with {len(df)} records")


if __name__ == '__main__':
    create_sample_train_data()
    create_sample_store_data()
    create_sample_test_data()
    print("\nAll fixtures created successfully!")
