#!/usr/bin/env python3
"""
Run all experiments from paper.
Executes irregular inputs validation and Cipolla comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.robust_experimental_extension import main as run_irregular
from src.cipolla_comparison import main as run_cipolla

def main():
    print("="*80)
    print("RUNNING ALL EXPERIMENTS FROM PAPER")
    print("="*80)

    print("\n[1/2] Running irregular inputs validation (Section 5.6)...")
    run_irregular()

    print("\n[2/2] Running Cipolla comparison (Section 5.4)...")
    run_cipolla()

    print("\n" + "="*80)
    print("✅ ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - data/irregular_inputs_results.csv")
    print("  - data/cipolla_comparison_results.csv")
    print("  - results/figures/*.pdf")

if __name__ == "__main__":
    main()
