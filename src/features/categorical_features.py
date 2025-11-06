"""
Categorical Feature Engineering

Handles encoding and feature extraction from categorical variables:
- Store type and assortment encoding
- Promo encoding
- Competition features
- Store-specific features
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CategoricalFeatureEngineer:
    """
    Engineer features from categorical variables

    Features created:
    - Label encoded categoricals
    - One-hot encoded categoricals (optional)
    - Target encoded categoricals (optional)
    - Competition-based features
    - Promo features
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        use_onehot: bool = False,
        use_target_encoding: bool = False
    ):
        """
        Initialize categorical feature engineer

        Args:
            categorical_columns: List of categorical column names
            use_onehot: Whether to use one-hot encoding
            use_target_encoding: Whether to use target encoding
        """
        self.categorical_columns = categorical_columns or [
            'StoreType', 'Assortment', 'StateHoliday', 'PromoInterval'
        ]
        self.use_onehot = use_onehot
        self.use_target_encoding = use_target_encoding
        self.label_encoders = {}
        self.target_encoders = {}
        self.feature_names = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit encoders on training data

        Args:
            X: Input dataframe
            y: Target variable (for target encoding)

        Returns:
            self
        """
        df = X.copy()

        # Fit label encoders
        for col in self.categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values by treating them as a separate category
                mask = df[col].notna()
                if mask.any():
                    le.fit(df.loc[mask, col].astype(str))
                    self.label_encoders[col] = le

        # Fit target encoders if requested
        if self.use_target_encoding and y is not None:
            for col in self.categorical_columns:
                if col in df.columns:
                    target_means = df.groupby(col)[y.name].mean().to_dict()
                    self.target_encoders[col] = target_means

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical features

        Args:
            X: Input dataframe

        Returns:
            Transformed dataframe
        """
        df = X.copy()
        created_features = []

        # Label encoding
        for col in self.categorical_columns:
            if col in df.columns and col in self.label_encoders:
                new_col = f'{col}_Encoded'
                mask = df[col].notna()
                df[new_col] = -1  # Default for missing values
                if mask.any():
                    try:
                        df.loc[mask, new_col] = self.label_encoders[col].transform(
                            df.loc[mask, col].astype(str)
                        )
                    except ValueError:
                        # Handle unseen categories
                        df.loc[mask, new_col] = -1
                created_features.append(new_col)

        # Target encoding
        if self.use_target_encoding:
            for col in self.categorical_columns:
                if col in df.columns and col in self.target_encoders:
                    new_col = f'{col}_TargetEnc'
                    df[new_col] = df[col].map(self.target_encoders[col])
                    # Fill unseen categories with global mean
                    global_mean = np.mean(list(self.target_encoders[col].values()))
                    df[new_col].fillna(global_mean, inplace=True)
                    created_features.append(new_col)

        # Competition features
        df = self._engineer_competition_features(df)
        created_features.extend([
            'HasCompetition', 'CompetitionDistance_Log',
            'MonthsSinceCompetition', 'CompetitionAge_Binned'
        ])

        # Promo features
        df = self._engineer_promo_features(df)
        created_features.extend([
            'IsPromo', 'IsPromo2Active', 'MonthsSincePromo2'
        ])

        # Store type interaction features
        if 'StoreType_Encoded' in df.columns and 'Assortment_Encoded' in df.columns:
            df['StoreType_Assortment'] = (
                df['StoreType_Encoded'].astype(str) + '_' +
                df['Assortment_Encoded'].astype(str)
            )
            df['StoreType_Assortment_Encoded'] = df['StoreType_Assortment'].factorize()[0]
            created_features.append('StoreType_Assortment_Encoded')

        self.feature_names = created_features
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

    def _engineer_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create competition-related features

        Args:
            df: Input dataframe

        Returns:
            Dataframe with competition features
        """
        result = df.copy()

        # Competition presence indicator
        result['HasCompetition'] = result['CompetitionDistance'].notna().astype(int)

        # Log transform competition distance (handle missing values)
        result['CompetitionDistance_Log'] = np.log1p(
            result['CompetitionDistance'].fillna(result['CompetitionDistance'].median())
        )

        # Months since competition opened
        if 'CompetitionOpenSinceMonth' in result.columns and 'CompetitionOpenSinceYear' in result.columns:
            # Calculate months since competition
            if 'Date' in result.columns:
                result['Date_temp'] = pd.to_datetime(result['Date'])
                result['CompetitionOpenDate'] = pd.to_datetime(
                    result['CompetitionOpenSinceYear'].astype(str) + '-' +
                    result['CompetitionOpenSinceMonth'].astype(str) + '-01',
                    errors='coerce'
                )
                result['MonthsSinceCompetition'] = (
                    (result['Date_temp'] - result['CompetitionOpenDate']).dt.days / 30.44
                )
                result['MonthsSinceCompetition'] = result['MonthsSinceCompetition'].clip(lower=0)
                result['MonthsSinceCompetition'].fillna(0, inplace=True)

                # Binned competition age
                result['CompetitionAge_Binned'] = pd.cut(
                    result['MonthsSinceCompetition'],
                    bins=[-1, 0, 6, 12, 24, 48, np.inf],
                    labels=[0, 1, 2, 3, 4, 5]
                ).astype(int)

                result.drop(['Date_temp', 'CompetitionOpenDate'], axis=1, inplace=True)

        return result

    def _engineer_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create promotion-related features

        Args:
            df: Input dataframe

        Returns:
            Dataframe with promo features
        """
        result = df.copy()

        # Basic promo indicator (ensure it's binary)
        if 'Promo' in result.columns:
            result['IsPromo'] = result['Promo'].astype(int)

        # Promo2 active indicator
        if 'Promo2' in result.columns:
            result['IsPromo2Active'] = result['Promo2'].astype(int)

        # Months since Promo2 started
        if 'Promo2SinceWeek' in result.columns and 'Promo2SinceYear' in result.columns:
            if 'Date' in result.columns:
                result['Date_temp'] = pd.to_datetime(result['Date'])
                # Approximate promo2 start date (week 1 = Jan 1st)
                result['Promo2StartDate'] = pd.to_datetime(
                    result['Promo2SinceYear'].astype(str) + '-01-01',
                    errors='coerce'
                ) + pd.to_timedelta(result['Promo2SinceWeek'] * 7, unit='D')

                result['MonthsSincePromo2'] = (
                    (result['Date_temp'] - result['Promo2StartDate']).dt.days / 30.44
                )
                result['MonthsSincePromo2'] = result['MonthsSincePromo2'].clip(lower=0)
                result['MonthsSincePromo2'].fillna(0, inplace=True)

                result.drop(['Date_temp', 'Promo2StartDate'], axis=1, inplace=True)

        # Promo interval encoding
        if 'PromoInterval' in result.columns and 'Month' in result.columns:
            # Check if current month is in promo interval
            promo_month_map = {
                'Jan,Apr,Jul,Oct': [1, 4, 7, 10],
                'Feb,May,Aug,Nov': [2, 5, 8, 11],
                'Mar,Jun,Sept,Dec': [3, 6, 9, 12]
            }

            def is_promo_month(row):
                if pd.isna(row['PromoInterval']):
                    return 0
                months = promo_month_map.get(row['PromoInterval'], [])
                return 1 if row['Month'] in months else 0

            result['IsPromo2Month'] = result.apply(is_promo_month, axis=1)

        return result

    def get_feature_names(self) -> List[str]:
        """
        Get list of created feature names

        Returns:
            List of feature names
        """
        return self.feature_names.copy()
