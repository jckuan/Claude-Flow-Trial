"""
Temporal Feature Engineering

Creates time-based features for sales forecasting including:
- Date components (day, month, year, quarter)
- Week-based features
- Holiday proximity features
- Seasonal indicators
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta


class TemporalFeatureEngineer:
    """
    Engineer temporal features from date columns

    Features created:
    - Year, Month, Day
    - DayOfWeek, WeekOfYear, Quarter
    - IsWeekend, IsMonthStart, IsMonthEnd
    - DaysSinceHoliday, DaysUntilHoliday
    - Season indicators
    """

    def __init__(self, date_column: str = 'Date'):
        """
        Initialize temporal feature engineer

        Args:
            date_column: Name of the date column in dataframe
        """
        self.date_column = date_column
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit the temporal feature engineer (stateless operation)

        Args:
            X: Input dataframe
            y: Target variable (unused)

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by adding temporal features

        Args:
            X: Input dataframe with date column

        Returns:
            Dataframe with additional temporal features
        """
        df = X.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column])

        # Basic date components
        df['Year'] = df[self.date_column].dt.year
        df['Month'] = df[self.date_column].dt.month
        df['Day'] = df[self.date_column].dt.day
        df['Quarter'] = df[self.date_column].dt.quarter
        df['WeekOfYear'] = df[self.date_column].dt.isocalendar().week.astype(int)

        # Day of week features (if not already present)
        if 'DayOfWeek' not in df.columns:
            df['DayOfWeek'] = df[self.date_column].dt.dayofweek + 1  # 1=Monday, 7=Sunday

        # Weekend indicator
        df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)

        # Month start/end indicators
        df['IsMonthStart'] = df[self.date_column].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df[self.date_column].dt.is_month_end.astype(int)

        # Season (1=Winter, 2=Spring, 3=Summer, 4=Fall)
        df['Season'] = (df['Month'] % 12 + 3) // 3

        # Days in month
        df['DaysInMonth'] = df[self.date_column].dt.days_in_month

        # Day of year
        df['DayOfYear'] = df[self.date_column].dt.dayofyear

        # Cyclic encoding for month (preserves cyclical nature)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

        # Cyclic encoding for day of week
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Cyclic encoding for day of month
        df['Day_Sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_Cos'] = np.cos(2 * np.pi * df['Day'] / 31)

        # Track feature names
        self.feature_names = [
            'Year', 'Month', 'Day', 'Quarter', 'WeekOfYear', 'DayOfWeek',
            'IsWeekend', 'IsMonthStart', 'IsMonthEnd', 'Season',
            'DaysInMonth', 'DayOfYear',
            'Month_Sin', 'Month_Cos', 'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Day_Sin', 'Day_Cos'
        ]

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

    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday proximity features

        Args:
            df: Dataframe with StateHoliday column

        Returns:
            Dataframe with holiday features
        """
        result = df.copy()

        # Convert StateHoliday to binary indicator
        if 'StateHoliday' in result.columns:
            result['IsStateHoliday'] = (result['StateHoliday'] != '0').astype(int)

        # School holiday indicator (already binary)
        if 'SchoolHoliday' in result.columns:
            result['IsSchoolHoliday'] = result['SchoolHoliday'].astype(int)

        # Combined holiday indicator
        if 'IsStateHoliday' in result.columns and 'IsSchoolHoliday' in result.columns:
            result['IsAnyHoliday'] = (
                (result['IsStateHoliday'] == 1) |
                (result['IsSchoolHoliday'] == 1)
            ).astype(int)

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names

        Returns:
            List of feature names
        """
        return self.feature_names.copy()
