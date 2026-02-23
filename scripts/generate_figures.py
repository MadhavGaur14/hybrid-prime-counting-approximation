#!/usr/bin/env python3
"""
Regenerate all figures from paper Section 5.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.visualization import (
    plot_absolute_error,
    plot_relative_error,
    plot_irregular_error
)

def main():
    print("Generating all paper figures...")

    # Note: Requires data files from experiments
    if not os.path.exists('data/irregular_inputs_results.csv'):
        print("ERROR: Run experiments first: python scripts/run_all_experiments.py")
        return

    plot_absolute_error('data/irregular_inputs_results.csv')
    plot_relative_error('data/irregular_inputs_results.csv')
    plot_irregular_error('data/irregular_inputs_results.csv')

    print("\n✅ All figures generated in results/figures/")

if __name__ == "__main__":
    main()
