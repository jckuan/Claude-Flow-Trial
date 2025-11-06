#!/usr/bin/env python3
"""
Evaluate trained models and generate comparison reports.
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
        description='Evaluate trained models'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained models (default: models/)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing processed data (default: data/)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='docs/visualizations',
        help='Directory for output visualizations (default: docs/visualizations/)'
    )
    
    return parser.parse_args()


def load_model(model_path):
    """Load a trained model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def main():
    """Main evaluation script."""
    args = parse_args()
    
    print("="*80)
    print("ROSSMANN STORE SALES - MODEL EVALUATION")
    print("="*80)
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Check if models exist
    model_files = list(model_dir.glob('*.pkl'))
    if not model_files:
        print(f"\n❌ No trained models found in {model_dir}")
        print("Please train models first using: python scripts/train_model.py")
        return 1
    
    print(f"\nFound {len(model_files)} trained models")
    
    # Load validation data
    val_path = data_dir / 'val_processed.csv'
    if not val_path.exists():
        print(f"\n❌ Validation data not found at {val_path}")
        print("Please run feature engineering first:")
        print("  python src/run_feature_pipeline.py")
        return 1
    
    try:
        from models.evaluator import ModelEvaluator
        
        print(f"Loading validation data from {val_path}...")
        val_df = pd.read_csv(val_path)
        
        # Separate features and target
        X_val = val_df.drop('Sales', axis=1)
        y_val = val_df['Sales']
        
        print(f"Validation set: {X_val.shape}")
        
        # Create evaluator
        evaluator = ModelEvaluator()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Evaluate each model
        results = []
        for model_path in model_files:
            model_name = model_path.stem
            print(f"\nEvaluating {model_name}...")
            
            try:
                model = load_model(model_path)
                predictions = model.predict(X_val)
                
                # Calculate metrics
                metrics = evaluator.calculate_metrics(y_val, predictions)
                metrics['model'] = model_name
                results.append(metrics)
                
                # Generate visualizations
                viz_path = output_dir / model_name
                viz_path.mkdir(exist_ok=True)
                
                evaluator.plot_predictions(
                    y_val, predictions,
                    title=f"{model_name} - Actual vs Predicted",
                    save_path=viz_path / 'predictions.png'
                )
                
                evaluator.plot_residuals(
                    y_val, predictions,
                    title=f"{model_name} - Residuals",
                    save_path=viz_path / 'residuals.png'
                )
                
                print(f"  RMSE: {metrics['rmse']:.2f}")
                print(f"  MAE: {metrics['mae']:.2f}")
                print(f"  R²: {metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"  ⚠️  Error evaluating {model_name}: {e}")
                continue
        
        # Create comparison table
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('rmse')
            
            print("\n" + "="*80)
            print("MODEL COMPARISON")
            print("="*80)
            print(results_df.to_string(index=False))
            
            # Save results
            results_path = output_dir / 'model_comparison.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\nResults saved to {results_path}")
            
            # Find best model
            best_model = results_df.iloc[0]
            print("\n" + "="*80)
            print(f"✅ BEST MODEL: {best_model['model']}")
            print(f"   RMSE: {best_model['rmse']:.2f}")
            print(f"   MAE: {best_model['mae']:.2f}")
            print(f"   R²: {best_model['r2']:.4f}")
            print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
