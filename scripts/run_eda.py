#!/usr/bin/env python3
"""
Run exploratory data analysis on Rossmann store sales data.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Run EDA analysis."""
    print("="*80)
    print("ROSSMANN STORE SALES - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    print("\nFor interactive EDA, please use the Jupyter notebook:")
    print("  jupyter notebook rossmann_eda.ipynb")
    print("\nOr view the EDA reports in the docs/ directory:")
    print("  - docs/eda_report.md")
    print("  - docs/eda_key_insights.md")
    print("  - docs/phase1_summary.md")
    
    print("\n" + "="*80)
    print("EDA resources are available. Please review the notebook or reports.")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
