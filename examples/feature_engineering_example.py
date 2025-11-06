"""
Feature Engineering Example

Simple example demonstrating the feature engineering pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.pipeline import create_features


def main():
    """Run simple feature engineering example"""

    print("\n" + "="*60)
    print("Simple Feature Engineering Example")
    print("="*60)

    # Run pipeline
    print("\nRunning feature engineering pipeline...")

    datasets = create_features(
        data_path='../data/rossmann-store-sales',
        save_path='../data/processed',
        use_target_encoding=False,
        scaling_method='standard'
    )

    # Access datasets
    train = datasets['train']
    val = datasets['val']
    test = datasets['test']

    print(f"\n✅ Pipeline complete!")
    print(f"   Train: {train.shape}")
    print(f"   Val: {val.shape}")
    print(f"   Test: {test.shape}")

    # Show sample features
    print("\n📊 Sample Features:")
    print(train[['Sales', 'Year', 'Month', 'IsWeekend',
                 'Sales_Lag7', 'Sales_RollingMean7']].head())


if __name__ == '__main__':
    main()
