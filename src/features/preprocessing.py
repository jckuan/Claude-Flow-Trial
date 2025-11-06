"""
Data Preprocessing and Scaling

Handles data cleaning, missing value imputation, and feature scaling:
- Missing value handling strategies
- Outlier detection and treatment
- Feature scaling and normalization
- Data validation
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings


class DataPreprocessor:
    """
    Preprocess data for machine learning

    Operations performed:
    - Handle missing values
    - Remove/cap outliers
    - Scale features
    - Validate data quality
    - Create train/validation/test splits
    """

    def __init__(
        self,
        scaling_method: str = 'standard',
        handle_outliers: bool = True,
        outlier_threshold: float = 3.0
    ):
        """
        Initialize data preprocessor

        Args:
            scaling_method: Method for scaling ('standard', 'robust', 'minmax', 'none')
            handle_outliers: Whether to handle outliers
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        self.imputers = {}
        self.feature_stats = {}

        # Initialize scaler
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit preprocessing transformers

        Args:
            X: Input dataframe
            y: Target variable (unused)

        Returns:
            self
        """
        df = X.copy()

        # Store feature statistics
        self.feature_stats = {
            'means': df.mean(numeric_only=True).to_dict(),
            'medians': df.median(numeric_only=True).to_dict(),
            'stds': df.std(numeric_only=True).to_dict()
        }

        # Fit imputers for numeric columns with missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                imputer = SimpleImputer(strategy='median')
                imputer.fit(df[[col]])
                self.imputers[col] = imputer

        # Fit scaler on numeric columns (excluding identifiers)
        if self.scaler is not None:
            scale_cols = [
                col for col in numeric_cols
                if col not in ['Store', 'Year', 'Month', 'Day']
            ]
            if scale_cols:
                self.scaler.fit(df[scale_cols].fillna(df[scale_cols].median()))

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations

        Args:
            X: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        df = X.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Handle outliers
        if self.handle_outliers:
            df = self._handle_outliers(df)

        # Scale features
        if self.scaler is not None:
            df = self._scale_features(df)

        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            X: Input dataframe
            y: Target variable

        Returns:
            Preprocessed dataframe
        """
        return self.fit(X, y).transform(X)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies

        Args:
            df: Input dataframe

        Returns:
            Dataframe with imputed values
        """
        result = df.copy()

        # Impute numeric columns
        for col, imputer in self.imputers.items():
            if col in result.columns:
                result[col] = imputer.transform(result[[col]])

        # Store-specific imputation strategies
        if 'CompetitionDistance' in result.columns:
            # Use median for missing competition distance
            median_dist = result['CompetitionDistance'].median()
            result['CompetitionDistance'].fillna(median_dist, inplace=True)

        # Competition date features - fill with 0 (no competition)
        comp_cols = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear',
                     'MonthsSinceCompetition']
        for col in comp_cols:
            if col in result.columns:
                result[col].fillna(0, inplace=True)

        # Promo2 features - fill with 0 (no promo2)
        promo_cols = ['Promo2SinceWeek', 'Promo2SinceYear', 'MonthsSincePromo2']
        for col in promo_cols:
            if col in result.columns:
                result[col].fillna(0, inplace=True)

        return result

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and handle outliers using z-score method

        Args:
            df: Input dataframe

        Returns:
            Dataframe with handled outliers
        """
        result = df.copy()

        # Only handle outliers in continuous numeric features
        outlier_cols = ['Sales', 'Customers', 'CompetitionDistance']

        for col in outlier_cols:
            if col in result.columns and col in self.feature_stats['means']:
                mean = self.feature_stats['means'][col]
                std = self.feature_stats['stds'][col]

                if std > 0:
                    # Calculate z-scores
                    z_scores = np.abs((result[col] - mean) / std)

                    # Cap outliers at threshold
                    upper_bound = mean + (self.outlier_threshold * std)
                    lower_bound = mean - (self.outlier_threshold * std)

                    result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)

        return result

    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numeric features

        Args:
            df: Input dataframe

        Returns:
            Dataframe with scaled features
        """
        result = df.copy()

        # Get numeric columns to scale (excluding identifiers)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        scale_cols = [
            col for col in numeric_cols
            if col not in ['Store', 'Year', 'Month', 'Day', 'Sales']  # Don't scale target
        ]

        if scale_cols:
            # Create scaled versions
            scaled_data = self.scaler.transform(result[scale_cols].fillna(0))
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=[f'{col}_Scaled' for col in scale_cols],
                index=result.index
            )
            result = pd.concat([result, scaled_df], axis=1)

        return result

    @staticmethod
    def create_splits(
        df: pd.DataFrame,
        date_column: str = 'Date',
        val_days: int = 48,
        test_days: int = 48
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based train/validation/test splits

        Args:
            df: Input dataframe with date column
            date_column: Name of date column
            val_days: Number of days for validation set
            test_days: Number of days for test set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])

        # Sort by date
        df_sorted = df.sort_values(date_column)

        # Calculate split dates
        max_date = df_sorted[date_column].max()
        test_start = max_date - pd.Timedelta(days=test_days - 1)
        val_start = test_start - pd.Timedelta(days=val_days)

        # Create splits
        train = df_sorted[df_sorted[date_column] < val_start].copy()
        val = df_sorted[
            (df_sorted[date_column] >= val_start) &
            (df_sorted[date_column] < test_start)
        ].copy()
        test = df_sorted[df_sorted[date_column] >= test_start].copy()

        return train, val, test

    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality

        Args:
            df: Input dataframe

        Returns:
            Dictionary with validation results
        """
        results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns),
            'date_range': None,
            'issues': []
        }

        # Check date range
        if 'Date' in df.columns:
            try:
                date_series = pd.to_datetime(df['Date'])
                results['date_range'] = {
                    'start': date_series.min(),
                    'end': date_series.max(),
                    'days': (date_series.max() - date_series.min()).days
                }
            except:
                results['issues'].append('Invalid date format')

        # Check for high missing value columns
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > 50].to_dict()
        if high_missing:
            results['issues'].append(f'Columns with >50% missing: {list(high_missing.keys())}')

        # Check for constant columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() == 1:
                results['issues'].append(f'Constant column: {col}')

        return results
