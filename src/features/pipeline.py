"""
Feature Engineering Pipeline

Main orchestrator for complete feature engineering workflow:
- Data loading and merging
- Temporal feature creation
- Categorical encoding
- Lag and rolling features
- Preprocessing and scaling
- Data splitting
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import os
from pathlib import Path
import warnings

from .temporal_features import TemporalFeatureEngineer
from .categorical_features import CategoricalFeatureEngineer
from .lag_features import LagFeatureEngineer
from .preprocessing import DataPreprocessor


class FeatureEngineeringPipeline:
    """
    Complete feature engineering pipeline for Rossmann sales prediction

    Pipeline stages:
    1. Data loading and merging
    2. Temporal features
    3. Categorical features
    4. Lag and rolling features
    5. Preprocessing and scaling
    6. Data splitting
    """

    def __init__(
        self,
        data_path: str = 'data/rossmann-store-sales',
        use_target_encoding: bool = False,
        scaling_method: str = 'standard',
        create_lag_features: bool = True
    ):
        """
        Initialize feature engineering pipeline

        Args:
            data_path: Path to data directory
            use_target_encoding: Whether to use target encoding for categoricals
            scaling_method: Scaling method ('standard', 'robust', 'minmax', 'none')
            create_lag_features: Whether to create lag features
        """
        self.data_path = data_path
        self.use_target_encoding = use_target_encoding
        self.scaling_method = scaling_method
        self.create_lag_features = create_lag_features

        # Initialize feature engineers
        self.temporal_engineer = TemporalFeatureEngineer()
        self.categorical_engineer = CategoricalFeatureEngineer(
            use_target_encoding=use_target_encoding
        )
        self.lag_engineer = LagFeatureEngineer() if create_lag_features else None
        self.preprocessor = DataPreprocessor(scaling_method=scaling_method)

        self.is_fitted = False
        self.feature_names = []

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load train, test, and store data

        Returns:
            Tuple of (train_df, test_df, store_df)
        """
        train_path = os.path.join(self.data_path, 'train.csv')
        test_path = os.path.join(self.data_path, 'test.csv')
        store_path = os.path.join(self.data_path, 'store.csv')

        print(f"Loading data from {self.data_path}...")
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        store = pd.read_csv(store_path)

        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        print(f"Store shape: {store.shape}")

        return train, test, store

    def prepare_data(
        self,
        train: pd.DataFrame,
        store: pd.DataFrame,
        is_test: bool = False
    ) -> pd.DataFrame:
        """
        Merge train/test data with store information

        Args:
            train: Training or test dataframe
            store: Store information dataframe
            is_test: Whether this is test data

        Returns:
            Merged dataframe
        """
        # Merge with store data
        df = train.merge(store, on='Store', how='left')

        # Filter out closed stores (unless test data)
        if not is_test and 'Open' in df.columns:
            df = df[df['Open'] == 1].copy()

        # Filter out rows with zero sales in training
        if not is_test and 'Sales' in df.columns:
            df = df[df['Sales'] > 0].copy()

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit all feature engineering transformers

        Args:
            X: Input dataframe
            y: Target variable (for target encoding)

        Returns:
            self
        """
        print("\n=== Fitting Feature Engineering Pipeline ===")

        df = X.copy()

        # 1. Temporal features
        print("Fitting temporal features...")
        df = self.temporal_engineer.fit_transform(df)
        df = self.temporal_engineer.add_holiday_features(df)

        # 2. Categorical features
        print("Fitting categorical features...")
        df = self.categorical_engineer.fit_transform(df, y)

        # 3. Lag features (only if requested)
        if self.create_lag_features and self.lag_engineer is not None:
            print("Fitting lag features...")
            df = self.lag_engineer.fit_transform(df)
            df = self.lag_engineer.create_day_of_week_features(df)

        # 4. Preprocessing
        print("Fitting preprocessor...")
        self.preprocessor.fit(df, y)

        self.is_fitted = True

        # Collect feature names
        self._collect_feature_names(df)

        print(f"Pipeline fitted successfully. Total features: {len(self.feature_names)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted pipeline

        Args:
            X: Input dataframe

        Returns:
            Transformed dataframe
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")

        print("\n=== Transforming Data ===")

        df = X.copy()

        # 1. Temporal features
        print("Creating temporal features...")
        df = self.temporal_engineer.transform(df)
        df = self.temporal_engineer.add_holiday_features(df)

        # 2. Categorical features
        print("Encoding categorical features...")
        df = self.categorical_engineer.transform(df)

        # 3. Lag features
        if self.create_lag_features and self.lag_engineer is not None:
            print("Creating lag features...")
            df = self.lag_engineer.transform(df)
            df = self.lag_engineer.create_day_of_week_features(df)

        # 4. Preprocessing
        print("Preprocessing data...")
        df = self.preprocessor.transform(df)

        print(f"Transformation complete. Shape: {df.shape}")

        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step

        Args:
            X: Input dataframe
            y: Target variable

        Returns:
            Transformed dataframe
        """
        return self.fit(X, y).transform(X)

    def create_train_val_test_splits(
        self,
        df: pd.DataFrame,
        val_days: int = 48,
        test_days: int = 48
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create time-based data splits

        Args:
            df: Full dataframe
            val_days: Days for validation set
            test_days: Days for test set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print(f"\nCreating data splits (val={val_days} days, test={test_days} days)...")

        train, val, test = self.preprocessor.create_splits(
            df,
            date_column='Date',
            val_days=val_days,
            test_days=test_days
        )

        print(f"Train shape: {train.shape}")
        print(f"Validation shape: {val.shape}")
        print(f"Test shape: {test.shape}")

        return train, val, test

    def run_full_pipeline(
        self,
        save_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete pipeline from data loading to final splits

        Args:
            save_path: Optional path to save processed data

        Returns:
            Dictionary with train, val, test dataframes
        """
        print("\n" + "="*80)
        print("RUNNING COMPLETE FEATURE ENGINEERING PIPELINE")
        print("="*80)

        # 1. Load data
        train_raw, test_raw, store = self.load_data()

        # 2. Prepare data (merge with store info)
        print("\nPreparing training data...")
        train_full = self.prepare_data(train_raw, store, is_test=False)

        # 3. Fit and transform training data
        y_train = train_full['Sales'].copy()
        X_train = train_full.drop('Sales', axis=1)

        X_train_transformed = self.fit_transform(X_train, y_train)
        X_train_transformed['Sales'] = y_train

        # 4. Create splits
        train_df, val_df, test_df = self.create_train_val_test_splits(
            X_train_transformed,
            val_days=48,
            test_days=48
        )

        # 5. Validate data
        print("\nValidating data quality...")
        validation_results = self.preprocessor.validate_data(train_df)
        print(f"Validation results: {len(validation_results['issues'])} issues found")
        if validation_results['issues']:
            for issue in validation_results['issues']:
                print(f"  - {issue}")

        # 6. Save processed data
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"\nSaving processed data to {save_path}...")

            train_df.to_csv(os.path.join(save_path, 'train_processed.csv'), index=False)
            val_df.to_csv(os.path.join(save_path, 'val_processed.csv'), index=False)
            test_df.to_csv(os.path.join(save_path, 'test_processed.csv'), index=False)

            # Save feature names
            with open(os.path.join(save_path, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(self.feature_names))

            print("Data saved successfully!")

        # 7. Print summary
        self._print_summary(train_df, val_df, test_df)

        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

    def _collect_feature_names(self, df: pd.DataFrame):
        """
        Collect all feature names from transformed dataframe

        Args:
            df: Transformed dataframe
        """
        # Exclude target and identifier columns
        exclude_cols = ['Sales', 'Date', 'Store', 'Customers']
        self.feature_names = [
            col for col in df.columns
            if col not in exclude_cols
        ]

    def _print_summary(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ):
        """
        Print pipeline summary statistics

        Args:
            train: Training dataframe
            val: Validation dataframe
            test: Test dataframe
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING PIPELINE SUMMARY")
        print("="*80)

        print(f"\nDataset Sizes:")
        print(f"  Training:   {len(train):,} rows")
        print(f"  Validation: {len(val):,} rows")
        print(f"  Test:       {len(test):,} rows")

        print(f"\nFeature Counts:")
        print(f"  Total features: {len(self.feature_names)}")
        print(f"  Temporal features: {len(self.temporal_engineer.get_feature_names())}")
        print(f"  Categorical features: {len(self.categorical_engineer.get_feature_names())}")
        if self.lag_engineer:
            print(f"  Lag features: {len(self.lag_engineer.get_feature_names())}")

        print(f"\nTarget Statistics (Training):")
        if 'Sales' in train.columns:
            print(f"  Mean: ${train['Sales'].mean():.2f}")
            print(f"  Median: ${train['Sales'].median():.2f}")
            print(f"  Std: ${train['Sales'].std():.2f}")
            print(f"  Min: ${train['Sales'].min():.2f}")
            print(f"  Max: ${train['Sales'].max():.2f}")

        print(f"\nDate Ranges:")
        if 'Date' in train.columns:
            print(f"  Training:   {train['Date'].min()} to {train['Date'].max()}")
            print(f"  Validation: {val['Date'].min()} to {val['Date'].max()}")
            print(f"  Test:       {test['Date'].min()} to {test['Date'].max()}")

        print("\n" + "="*80)

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by type for importance analysis

        Returns:
            Dictionary mapping feature types to feature names
        """
        groups = {
            'temporal': self.temporal_engineer.get_feature_names(),
            'categorical': self.categorical_engineer.get_feature_names(),
        }

        if self.lag_engineer:
            groups['lag'] = self.lag_engineer.get_feature_names()

        return groups


# Convenience function for quick pipeline execution
def create_features(
    data_path: str = 'data/rossmann-store-sales',
    save_path: Optional[str] = None,
    use_target_encoding: bool = False,
    scaling_method: str = 'standard'
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run complete feature engineering pipeline

    Args:
        data_path: Path to raw data directory
        save_path: Path to save processed data
        use_target_encoding: Whether to use target encoding
        scaling_method: Scaling method to use

    Returns:
        Dictionary with train, val, test dataframes
    """
    pipeline = FeatureEngineeringPipeline(
        data_path=data_path,
        use_target_encoding=use_target_encoding,
        scaling_method=scaling_method
    )

    return pipeline.run_full_pipeline(save_path=save_path)
