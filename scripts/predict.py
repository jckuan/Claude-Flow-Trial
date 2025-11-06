#!/usr/bin/env python3
"""
Generate predictions on test set using trained model.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate predictions for test set'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.pkl)'
    )
    
    parser.add_argument(
        '--test-data',
        type=str,
        default='data/test_processed.csv',
        help='Path to processed test data (default: data/test_processed.csv)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='submission.csv',
        help='Output filename for predictions (default: submission.csv)'
    )
    
    parser.add_argument(
        '--original-test',
        type=str,
        default='data/rossmann-store-sales/test.csv',
        help='Original test file for ID column (default: data/rossmann-store-sales/test.csv)'
    )
    
    return parser.parse_args()


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def main():
    """Main prediction script."""
    args = parse_args()
    
    print("="*80)
    print("ROSSMANN STORE SALES - PREDICTION GENERATION")
    print("="*80)
    
    model_path = Path(args.model)
    test_path = Path(args.test_data)
    original_test_path = Path(args.original_test)
    output_path = Path(args.output)
    
    # Check if files exist
    if not model_path.exists():
        print(f"\n❌ Model file not found: {model_path}")
        print("\nAvailable models:")
        models_dir = Path('models')
        if models_dir.exists():
            for model in models_dir.glob('*.pkl'):
                print(f"  - {model}")
        return 1
    
    if not test_path.exists():
        print(f"\n❌ Processed test data not found: {test_path}")
        print("Please run feature engineering first:")
        print("  python src/run_feature_pipeline.py")
        return 1
    
    try:
        # Load model
        print(f"\nLoading model from {model_path}...")
        model = load_model(model_path)
        print("✓ Model loaded successfully")
        
        # Load test data
        print(f"\nLoading test data from {test_path}...")
        test_df = pd.read_csv(test_path)
        print(f"✓ Test set shape: {test_df.shape}")
        
        # Remove target if present
        if 'Sales' in test_df.columns:
            test_df = test_df.drop('Sales', axis=1)
        
        # Generate predictions
        print("\nGenerating predictions...")
        predictions = model.predict(test_df)
        print(f"✓ Generated {len(predictions)} predictions")
        
        # Load original test file for IDs
        if original_test_path.exists():
            print(f"\nLoading original test file from {original_test_path}...")
            original_test = pd.read_csv(original_test_path)
            
            # Create submission file
            submission = pd.DataFrame({
                'Id': original_test['Id'] if 'Id' in original_test.columns else range(1, len(predictions) + 1),
                'Sales': predictions
            })
        else:
            print(f"\n⚠️  Original test file not found at {original_test_path}")
            print("Creating submission without IDs...")
            submission = pd.DataFrame({
                'Id': range(1, len(predictions) + 1),
                'Sales': predictions
            })
        
        # Ensure non-negative predictions
        submission['Sales'] = submission['Sales'].clip(lower=0)
        
        # Save submission
        submission.to_csv(output_path, index=False)
        
        print("\n" + "="*80)
        print("✅ PREDICTIONS GENERATED SUCCESSFULLY")
        print("="*80)
        print(f"\nOutput file: {output_path}")
        print(f"Number of predictions: {len(submission)}")
        print(f"\nPrediction Statistics:")
        print(f"  Mean: ${submission['Sales'].mean():.2f}")
        print(f"  Median: ${submission['Sales'].median():.2f}")
        print(f"  Min: ${submission['Sales'].min():.2f}")
        print(f"  Max: ${submission['Sales'].max():.2f}")
        print(f"  Std: ${submission['Sales'].std():.2f}")
        
        # Show sample predictions
        print("\nSample Predictions:")
        print(submission.head(10).to_string(index=False))
        
        print("\n" + "="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
