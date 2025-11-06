"""
Feature engineering for Rossmann sales prediction.
Creates temporal, lag, and interaction features.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.config import config
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class FeatureEngineer:
    """Create features for sales prediction."""

    def __init__(self, feature_config=None):
        """
        Initialize feature engineer.

        Args:
            feature_config: FeatureConfig instance
        """
        self.config = feature_config or config.features
        self.feature_names = []

    @log_execution_time(logger)
    def create_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Create all features for the dataset.

        Args:
            df: Input dataframe
            is_train: Whether this is training data

        Returns:
            Dataframe with engineered features
        """
        logger.info(f"Creating features (is_train={is_train})...")
        df = df.copy()

        # Temporal features
        if self.config.extract_date_features:
            df = self._create_temporal_features(df)

        # Store features
        df = self._create_store_features(df)

        # Promo features
        df = self._create_promo_features(df)

        # Competition features
        df = self._create_competition_features(df)

        # Lag features (only for training or if we have history)
        if self.config.create_lag_features and is_train:
            df = self._create_lag_features(df)

        # Interaction features
        if self.config.create_interactions:
            df = self._create_interaction_features(df)

        # Encode categorical variables
        if self.config.encode_categorical:
            df = self._encode_categorical(df)

        logger.info(f"Feature engineering complete: {df.shape}")
        return df

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating temporal features...")
        df = df.copy()

        # Extract date components
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week
        df["Quarter"] = df["Date"].dt.quarter

        # Cyclical encoding for month and day
        df["MonthSin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["MonthCos"] = np.cos(2 * np.pi * df["Month"] / 12)
        df["DayOfWeekSin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeekCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

        # Additional temporal features
        df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
        df["IsMonthStart"] = (df["Day"] <= 7).astype(int)
        df["IsMonthEnd"] = (df["Day"] >= 23).astype(int)

        # Days in month
        df["DaysInMonth"] = df["Date"].dt.days_in_month

        return df

    def _create_store_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create store-related features."""
        logger.info("Creating store features...")
        df = df.copy()

        # Store type and assortment are already present
        # Create store group features
        df["StoreAssortment"] = df["StoreType"].astype(str) + "_" + df["Assortment"].astype(str)

        return df

    def _create_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create promotion-related features."""
        logger.info("Creating promo features...")
        df = df.copy()

        # Promo active
        df["PromoActive"] = df["Promo"]

        # Promo2 active (check if current month is in promo interval)
        if "PromoInterval" in df.columns and "Month" in df.columns:
            df["Promo2Active"] = 0

            # Map month to interval names
            month_to_interval = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }

            df["MonthName"] = df["Month"].map(month_to_interval)

            # Check if month is in promo interval
            mask = (
                (df["Promo2"] == 1) &
                (df["PromoInterval"] != "None") &
                (df["PromoInterval"].notna())
            )

            for idx in df[mask].index:
                interval = df.loc[idx, "PromoInterval"]
                month = df.loc[idx, "MonthName"]
                if month in interval:
                    df.loc[idx, "Promo2Active"] = 1

            df.drop("MonthName", axis=1, inplace=True)

        # Combined promo indicator
        df["AnyPromo"] = ((df["Promo"] == 1) | (df.get("Promo2Active", 0) == 1)).astype(int)

        return df

    def _create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competition-related features."""
        logger.info("Creating competition features...")
        df = df.copy()

        # Competition distance bins
        if "CompetitionDistance" in df.columns:
            df["CompDistanceBin"] = pd.cut(
                df["CompetitionDistance"],
                bins=[0, 500, 1000, 2000, 5000, np.inf],
                labels=["very_close", "close", "medium", "far", "very_far"]
            )

            # Has nearby competition
            df["HasNearbyCompetition"] = (df["CompetitionDistance"] < 1000).astype(int)

        # Competition duration (months since competition opened)
        if all(col in df.columns for col in ["Year", "Month", "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]):
            df["CompetitionOpenMonths"] = (
                (df["Year"] - df["CompetitionOpenSinceYear"]) * 12 +
                (df["Month"] - df["CompetitionOpenSinceMonth"])
            )
            df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].clip(lower=0)

            # Has established competition
            df["HasEstablishedCompetition"] = (df["CompetitionOpenMonths"] >= 12).astype(int)

        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag and rolling features."""
        logger.info("Creating lag features...")
        df = df.copy()

        # Sort by store and date
        df = df.sort_values(["Store", "Date"])

        # Only create lags if Sales column exists (training data)
        if "Sales" not in df.columns:
            logger.warning("Sales column not found, skipping lag features")
            return df

        # Create lag features for each store
        for lag in self.config.lag_periods:
            df[f"Sales_Lag{lag}"] = df.groupby("Store")["Sales"].shift(lag)

        # Create rolling mean features
        for window in self.config.rolling_windows:
            df[f"Sales_RollingMean{window}"] = (
                df.groupby("Store")["Sales"]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Fill NaN values with 0 (for early records without history)
        lag_cols = [col for col in df.columns if "Lag" in col or "Rolling" in col]
        df[lag_cols] = df[lag_cols].fillna(0)

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        logger.info("Creating interaction features...")
        df = df.copy()

        # Promo x DayOfWeek
        df["Promo_DayOfWeek"] = df["Promo"].astype(str) + "_" + df["DayOfWeek"].astype(str)

        # StoreType x Promo
        df["StoreType_Promo"] = df["StoreType"].astype(str) + "_" + df["Promo"].astype(str)

        # SchoolHoliday x DayOfWeek
        df["SchoolHoliday_DayOfWeek"] = (
            df["SchoolHoliday"].astype(str) + "_" + df["DayOfWeek"].astype(str)
        )

        return df

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables."""
        logger.info("Encoding categorical variables...")
        df = df.copy()

        categorical_cols = [
            "StoreType", "Assortment", "CompDistanceBin",
            "StoreAssortment", "Promo_DayOfWeek", "StoreType_Promo",
            "SchoolHoliday_DayOfWeek"
        ]

        # Only encode columns that exist
        cols_to_encode = [col for col in categorical_cols if col in df.columns]

        if cols_to_encode:
            df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
            logger.info(f"Encoded {len(cols_to_encode)} categorical columns")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature column names (excluding target and metadata).

        Args:
            df: Dataframe with features

        Returns:
            List of feature column names
        """
        exclude_cols = [
            "Sales", "Customers", "Date", "Store",
            "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
            "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"
        ]

        features = [col for col in df.columns if col not in exclude_cols]
        return features


def create_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Convenience function to create features.

    Args:
        df: Input dataframe
        is_train: Whether this is training data

    Returns:
        Dataframe with features
    """
    engineer = FeatureEngineer()
    return engineer.create_features(df, is_train)


if __name__ == "__main__":
    from data.data_loader import load_data
    from data.preprocessing import preprocess_data

    # Load and preprocess data
    train, test, store = load_data()
    train_proc, test_proc = preprocess_data(train, test, store)

    # Create features
    engineer = FeatureEngineer()
    train_features = engineer.create_features(train_proc.head(1000), is_train=True)

    print(f"\nOriginal shape: {train_proc.head(1000).shape}")
    print(f"With features: {train_features.shape}")
    print(f"\nFeature columns: {engineer.get_feature_names(train_features)}")
    print(f"\nSample features:")
    print(train_features.head())
