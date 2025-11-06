"""
Lag and Rolling Features Engineering

Creates lag features and rolling statistics for time series forecasting:
- Sales lag features (1 day, 1 week, 1 month)
- Customer lag features
- Rolling averages (7-day, 30-day windows)
- Rolling standard deviations
- Exponential moving averages
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import warnings


class LagFeatureEngineer:
    """
    Engineer lag and rolling window features for time series

    Features created:
    - Lag features: previous day, week, month sales
    - Rolling means: 7-day, 14-day, 30-day averages
    - Rolling stds: volatility measures
    - Expanding means: cumulative averages
    - Trend features
    """

    def __init__(
        self,
        target_column: str = 'Sales',
        date_column: str = 'Date',
        store_column: str = 'Store',
        lag_periods: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        """
        Initialize lag feature engineer

        Args:
            target_column: Name of target variable column
            date_column: Name of date column
            store_column: Name of store identifier column
            lag_periods: List of lag periods in days
            rolling_windows: List of rolling window sizes in days
        """
        self.target_column = target_column
        self.date_column = date_column
        self.store_column = store_column
        self.lag_periods = lag_periods or [1, 7, 14, 30]
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the lag feature engineer (stateless operation)

        Args:
            X: Input dataframe
            y: Target variable (unused)

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by adding lag and rolling features

        Args:
            X: Input dataframe with date and target columns

        Returns:
            Dataframe with additional lag features
        """
        df = X.copy()
        created_features = []

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Sort by store and date for proper lag calculation
        df = df.sort_values([self.store_column, self.date_column])

        # Create lag features by store
        for lag in self.lag_periods:
            col_name = f'{self.target_column}_Lag{lag}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].shift(lag)
            created_features.append(col_name)

        # Create rolling mean features by store
        for window in self.rolling_windows:
            col_name = f'{self.target_column}_RollingMean{window}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            created_features.append(col_name)

        # Create rolling std features by store
        for window in self.rolling_windows:
            col_name = f'{self.target_column}_RollingStd{window}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=2).std()
            )
            # Fill NaN with 0 for initial periods
            df[col_name].fillna(0, inplace=True)
            created_features.append(col_name)

        # Create rolling max/min features
        for window in [7, 30]:
            # Rolling max
            col_name = f'{self.target_column}_RollingMax{window}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            created_features.append(col_name)

            # Rolling min
            col_name = f'{self.target_column}_RollingMin{window}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            created_features.append(col_name)

        # Exponential moving average
        for span in [7, 30]:
            col_name = f'{self.target_column}_EMA{span}'
            df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
            created_features.append(col_name)

        # Expanding mean (cumulative average per store)
        col_name = f'{self.target_column}_ExpandingMean'
        df[col_name] = df.groupby(self.store_column)[self.target_column].transform(
            lambda x: x.expanding(min_periods=1).mean()
        )
        created_features.append(col_name)

        # Add customer lag features if available
        if 'Customers' in df.columns:
            for lag in [1, 7]:
                col_name = f'Customers_Lag{lag}'
                df[col_name] = df.groupby(self.store_column)['Customers'].shift(lag)
                created_features.append(col_name)

        # Ratio of current to lag features (momentum indicators)
        if f'{self.target_column}_Lag7' in df.columns:
            df[f'{self.target_column}_Momentum7'] = (
                df[self.target_column] / (df[f'{self.target_column}_Lag7'] + 1)
            )
            created_features.append(f'{self.target_column}_Momentum7')

        if f'{self.target_column}_Lag30' in df.columns:
            df[f'{self.target_column}_Momentum30'] = (
                df[self.target_column] / (df[f'{self.target_column}_Lag30'] + 1)
            )
            created_features.append(f'{self.target_column}_Momentum30')

        # Trend feature (difference from moving average)
        if f'{self.target_column}_RollingMean7' in df.columns:
            df[f'{self.target_column}_Trend7'] = (
                df[self.target_column] - df[f'{self.target_column}_RollingMean7']
            )
            created_features.append(f'{self.target_column}_Trend7')

        self.feature_names = created_features
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            X: Input dataframe
            y: Target variable (unused)

        Returns:
            Transformed dataframe
        """
        return self.fit(X, y).transform(X)

    def create_day_of_week_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create day-of-week specific lag features

        Args:
            df: Input dataframe

        Returns:
            Dataframe with day-of-week lag features
        """
        result = df.copy()

        # Same day of week in previous week
        col_name = f'{self.target_column}_SameDayLastWeek'
        result[col_name] = result.groupby(self.store_column)[self.target_column].shift(7)

        # Same day 2 weeks ago
        col_name = f'{self.target_column}_SameDay2WeeksAgo'
        result[col_name] = result.groupby(self.store_column)[self.target_column].shift(14)

        # Average of same day in last 4 weeks
        col_name = f'{self.target_column}_SameDayAvg4Weeks'
        result[col_name] = result.groupby(self.store_column)[self.target_column].transform(
            lambda x: (x.shift(7) + x.shift(14) + x.shift(21) + x.shift(28)) / 4
        )

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names

        Returns:
            List of feature names
        """
        return self.feature_names.copy()
