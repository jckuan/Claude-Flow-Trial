"""
Run Feature Engineering Pipeline

This script demonstrates how to run the complete feature engineering pipeline
for the Rossmann sales prediction dataset.

Usage:
    python src/run_feature_pipeline.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from features.pipeline import FeatureEngineeringPipeline, create_features


def main():
    """
    Run the complete feature engineering pipeline
    """
    print("\n" + "="*80)
    print("ROSSMANN SALES PREDICTION - FEATURE ENGINEERING PIPELINE")
    print("="*80)

    # Configuration
    data_path = 'data/rossmann-store-sales'
    output_path = 'data/processed'

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\n❌ Error: Data directory not found at {data_path}")
        print("Please ensure the Rossmann dataset is available.")
        return

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    print("\n📊 Starting feature engineering pipeline...")
    print(f"   Data path: {data_path}")
    print(f"   Output path: {output_path}")

    try:
        # Option 1: Use convenience function
        print("\n🚀 Running pipeline with convenience function...")
        datasets = create_features(
            data_path=data_path,
            save_path=output_path,
            use_target_encoding=False,
            scaling_method='standard'
        )

        # Access results
        train_df = datasets['train']
        val_df = datasets['val']
        test_df = datasets['test']

        print("\n✅ Pipeline completed successfully!")
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        print(f"   Test samples: {len(test_df):,}")
        print(f"   Total features: {train_df.shape[1]}")

        # Print feature statistics
        print("\n📈 Feature Statistics:")
        print(f"   Temporal features: {len([c for c in train_df.columns if any(t in c for t in ['Year', 'Month', 'Day', 'Week', 'Quarter', 'Season'])])}")
        print(f"   Categorical features: {len([c for c in train_df.columns if 'Encoded' in c])}")
        print(f"   Lag features: {len([c for c in train_df.columns if 'Lag' in c])}")
        print(f"   Rolling features: {len([c for c in train_df.columns if 'Rolling' in c])}")
        print(f"   Scaled features: {len([c for c in train_df.columns if 'Scaled' in c])}")

        # Print sample of features
        print("\n🔍 Sample Features (first 5 rows of training data):")
        feature_cols = [c for c in train_df.columns if c not in ['Date', 'Sales', 'Store']]
        print(train_df[feature_cols[:10]].head())

        # Print saved files
        print("\n💾 Saved Files:")
        for file in os.listdir(output_path):
            file_path = os.path.join(output_path, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"   {file}: {size_mb:.2f} MB")

    except Exception as e:
        print(f"\n❌ Error running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    # Option 2: Show manual pipeline usage
    print("\n" + "="*80)
    print("ALTERNATIVE: Manual Pipeline Usage")
    print("="*80)

    print("\nYou can also use the pipeline manually for more control:")
    print("""
    from features.pipeline import FeatureEngineeringPipeline

    # Initialize pipeline
    pipeline = FeatureEngineeringPipeline(
        data_path='data/rossmann-store-sales',
        use_target_encoding=False,
        scaling_method='standard',
        create_lag_features=True
    )

    # Load data
    train, test, store = pipeline.load_data()

    # Prepare and merge data
    train_full = pipeline.prepare_data(train, store, is_test=False)

    # Fit and transform
    y_train = train_full['Sales']
    X_train = train_full.drop('Sales', axis=1)
    X_transformed = pipeline.fit_transform(X_train, y_train)

    # Create splits
    train_df, val_df, test_df = pipeline.create_train_val_test_splits(
        X_transformed
    )

    # Access feature groups
    feature_groups = pipeline.get_feature_importance_groups()
    """)

    print("\n" + "="*80)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*80)
    print(f"\nProcessed data is ready at: {output_path}/")
    print("You can now proceed to model training.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
