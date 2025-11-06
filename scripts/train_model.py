#!/usr/bin/env python3
"""
Train Rossmann sales prediction models with command-line options.
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Rossmann sales prediction models'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'baseline', 'linear', 'rf', 'xgboost', 'lightgbm', 'ensemble'],
        default='all',
        help='Model type to train (default: all)'
    )
    
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models/)'
    )
    
    return parser.parse_args()


def main():
    """Main training script."""
    args = parse_args()
    
    print("="*80)
    print("ROSSMANN STORE SALES - MODEL TRAINING")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Cross-validation folds: {args.n_folds}")
    print(f"Hyperparameter tuning: {'Yes' if args.tune else 'No'}")
    print(f"Output directory: {args.output_dir}")
    print("")
    
    # Import training module
    try:
        from train_models import main as train_main
        return train_main()
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("\nPlease ensure you have installed all requirements:")
        print("  pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
