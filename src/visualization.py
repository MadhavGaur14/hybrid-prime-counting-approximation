"""
Visualization utilities for paper figures.
Generates all plots from Section 5.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpmath import log10

def plot_absolute_error(data_file, output_file='results/figures/fig_abs_error.pdf'):
    """Generate Figure 1 from paper (absolute error vs x)."""
    df = pd.read_csv(data_file)

    plt.figure(figsize=(10, 6))
    plt.semilogy(df['log10_x'], df['abs_error_pi_g'], 'o-', label='$|\pi_g(x) - \pi(x)|$')
    plt.semilogy(df['log10_x'], df['abs_error_pi_h'], 's-', label='$|\pi_h(x) - \pi(x)|$')
    plt.semilogy(df['log10_x'], df['abs_error_li'], '^-', label='$|Li(x) - \pi(x)|$')

    plt.xlabel('$\log_{10}(x)$', fontsize=12)
    plt.ylabel('Absolute Error', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"✓ Saved: {output_file}")
