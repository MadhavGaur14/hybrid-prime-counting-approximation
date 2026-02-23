"""
Cipolla Approximation Implementation and Comparison
====================================================

This script implements Cipolla's 1902 approximation to π(x) and compares it 
with π_h^(N)(x) from the main paper.

Cipolla's formula:
π_Cipolla(x) = Li(x) - Σ (μ(n)/n) * Li(x^(1/n))

where μ(n) is the Möbius function and the sum is over n ≥ 2.

Author: Madhav Gaur
Date: February 21, 2026
"""

from mpmath import mp, li, ln, log10, sqrt, pi as mp_pi
from collections import defaultdict
import csv
from datetime import datetime

# Set precision
mp.dps = 100  # 100 decimal places for high precision

# ===========================
# PART 1: MÖBIUS FUNCTION
# ===========================

def mobius(n):
    """
    Compute the Möbius function μ(n).
    
    μ(n) = 1 if n is square-free with even number of prime factors
    μ(n) = -1 if n is square-free with odd number of prime factors
    μ(n) = 0 if n has a squared prime factor
    
    Args:
        n: Positive integer
        
    Returns:
        μ(n) ∈ {-1, 0, 1}
    """
    if n == 1:
        return 1
    
    # Factor n to detect square factors
    factors = []
    temp_n = n
    d = 2
    
    while d * d <= temp_n:
        count = 0
        while temp_n % d == 0:
            count += 1
            temp_n //= d
        
        if count > 1:
            # Square factor found
            return 0
        elif count == 1:
            factors.append(d)
        
        d += 1
    
    if temp_n > 1:
        factors.append(temp_n)
    
    # n is square-free, return (-1)^k where k is number of prime factors
    return (-1) ** len(factors)


def compute_mobius_up_to(max_n):
    """
    Precompute Möbius function values up to max_n.
    
    Args:
        max_n: Maximum n to compute
        
    Returns:
        Dictionary mapping n -> μ(n)
    """
    mobius_values = {}
    for n in range(1, max_n + 1):
        mobius_values[n] = mobius(n)
    return mobius_values


# ===========================
# PART 2: CIPOLLA'S FORMULA
# ===========================

def cipolla_approximation(x, N_terms=20, mobius_cache=None):
    """
    Compute Cipolla's approximation to π(x).
    
    π_Cipolla(x) = Li(x) - Σ_{n=2}^N (μ(n)/n) * Li(x^(1/n))
    
    Args:
        x: Argument (positive real number)
        N_terms: Number of terms to include (default 20)
        mobius_cache: Precomputed Möbius values (optional)
        
    Returns:
        Approximation to π(x)
    """
    # Convert x to mpmath type
    x_mp = mp.mpf(x)
    
    # Start with Li(x)
    result = li(x_mp)
    
    # Subtract correction terms
    for n in range(2, N_terms + 1):
        # Get Möbius value
        if mobius_cache and n in mobius_cache:
            mu_n = mobius_cache[n]
        else:
            mu_n = mobius(n)
        
        if mu_n == 0:
            continue  # Skip if μ(n) = 0
        
        # Compute x^(1/n)
        x_power = x_mp ** (mp.mpf(1) / mp.mpf(n))
        
        # Add term: -μ(n)/n * Li(x^(1/n))
        term = -mu_n * li(x_power) / mp.mpf(n)
        result += term
    
    return result


# ===========================
# PART 3: π_h(x) FROM PAPER
# ===========================

def pi_g(x):
    """
    Base rational approximant from the paper.
    
    π_g(x) = x(ln x + 1) / ((ln x)^2 + 1)
    """
    x_mp = mp.mpf(x)
    ln_x = ln(x_mp)
    
    numerator = x_mp * (ln_x + 1)
    denominator = ln_x**2 + 1
    
    return numerator / denominator


def pi_h_N(x, N=3):
    """
    Hybrid approximation from the paper.
    
    π_h^(N)(x) = Σ_{n=1}^N [π_g(x) + x^(1/n)] / (ln x)^n
    """
    x_mp = mp.mpf(x)
    ln_x = ln(x_mp)
    pi_g_val = pi_g(x_mp)
    
    result = mp.mpf(0)
    
    for n in range(1, N + 1):
        x_power = x_mp ** (mp.mpf(1) / mp.mpf(n))
        term = (pi_g_val + x_power) / (ln_x ** n)
        result += term
    
    return result


# ===========================
# PART 4: COMPARISON FRAMEWORK
# ===========================

# Known exact values of π(x) for x = 10^n
EXACT_PI_VALUES = {
    1: 4,
    2: 25,
    3: 168,
    4: 1229,
    5: 9592,
    6: 78498,
    7: 664579,
    8: 5761455,
    9: 50847534,
    10: 455052511,
    11: 4118054813,
    12: 37607912018,
    13: 346065536839,
    14: 3204941750802,
    15: 29844570422669,
    16: 279238341033925,
    17: 2623557157654233,
    18: 24739954287740860,
    19: 234057667276344607,
    20: 2220819602560918840,
}


def compare_approximations(test_points, N_cipolla=20, N_hybrid=3):
    """
    Compare Cipolla, π_h(x), and Li(x) approximations.
    
    Args:
        test_points: List of n values (x = 10^n will be tested)
        N_cipolla: Number of terms for Cipolla (default 20)
        N_hybrid: Truncation level for π_h (default 3)
        
    Returns:
        List of dictionaries containing results
    """
    # Precompute Möbius values
    print(f"Precomputing Möbius function values up to {N_cipolla}...")
    mobius_cache = compute_mobius_up_to(N_cipolla)
    
    results = []
    
    for n in test_points:
        x = mp.mpf(10) ** n
        pi_exact = EXACT_PI_VALUES.get(n, None)
        
        print(f"\nComputing for x = 10^{n}...")
        
        # Compute approximations
        print("  - Computing Li(x)...")
        li_val = li(x)
        
        print(f"  - Computing Cipolla (N={N_cipolla})...")
        cipolla_val = cipolla_approximation(x, N_cipolla, mobius_cache)
        
        print(f"  - Computing π_h^({N_hybrid})(x)...")
        pi_h_val = pi_h_N(x, N_hybrid)
        
        # Compute errors if exact value known
        result = {
            'n': n,
            'x': str(x),
            'pi_exact': pi_exact,
            'li': float(li_val),
            'cipolla': float(cipolla_val),
            'pi_h': float(pi_h_val),
        }
        
        if pi_exact is not None:
            result['error_li'] = abs(float(li_val) - pi_exact)
            result['error_cipolla'] = abs(float(cipolla_val) - pi_exact)
            result['error_pi_h'] = abs(float(pi_h_val) - pi_exact)
            
            # Relative errors in ppm
            result['rel_error_li_ppm'] = (result['error_li'] / pi_exact) * 1e6
            result['rel_error_cipolla_ppm'] = (result['error_cipolla'] / pi_exact) * 1e6
            result['rel_error_pi_h_ppm'] = (result['error_pi_h'] / pi_exact) * 1e6
        
        results.append(result)
        
        # Print summary
        if pi_exact:
            print(f"  Results:")
            print(f"    π({n}) = {pi_exact}")
            print(f"    |Li - π| = {result['error_li']:.2e} ({result['rel_error_li_ppm']:.2f} ppm)")
            print(f"    |Cipolla - π| = {result['error_cipolla']:.2e} ({result['rel_error_cipolla_ppm']:.2f} ppm)")
            print(f"    |π_h - π| = {result['error_pi_h']:.2e} ({result['rel_error_pi_h_ppm']:.2f} ppm)")
    
    return results


# ===========================
# PART 5: OUTPUT FORMATTING
# ===========================

def save_results_csv(results, filename='cipolla_comparison_results.csv'):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    fieldnames = results[0].keys()
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n✓ Results saved to {filename}")


def format_latex_table(results):
    """
    Generate LaTeX table code for inclusion in paper.
    """
    print("\n" + "="*80)
    print("LATEX TABLE CODE (copy to paper)")
    print("="*80)
    
    print(r"""
\begin{table}[htbp]
\centering
\caption{Comparison of Cipolla, $\pi_h^{(3)}(x)$, and Li$(x)$ approximations against known $\pi(10^n)$ values. Errors reported in parts per million (ppm).}
\label{tab:cipolla_comparison}
\begin{tabular}{@{}crrrrr@{}}
\toprule
$n$ & $\pi(10^n)$ & Li$(x)$ (ppm) & Cipolla (ppm) & $\pi_h^{(3)}(x)$ (ppm) \\
\midrule
""")
    
    for r in results:
        if r['pi_exact'] is not None:
            n = r['n']
            pi_val = r['pi_exact']
            li_ppm = r['rel_error_li_ppm']
            cipolla_ppm = r['rel_error_cipolla_ppm']
            pi_h_ppm = r['rel_error_pi_h_ppm']
            
            print(f"{n} & {pi_val} & {li_ppm:.2f} & {cipolla_ppm:.2f} & {pi_h_ppm:.2f} \\\\")
    
    print(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print("="*80 + "\n")


def generate_summary_statistics(results):
    """Generate summary statistics for the comparison."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    valid_results = [r for r in results if r['pi_exact'] is not None]
    
    if not valid_results:
        print("No valid results with exact π(x) values.")
        return
    
    # Compute average relative errors
    avg_li = sum(r['rel_error_li_ppm'] for r in valid_results) / len(valid_results)
    avg_cipolla = sum(r['rel_error_cipolla_ppm'] for r in valid_results) / len(valid_results)
    avg_pi_h = sum(r['rel_error_pi_h_ppm'] for r in valid_results) / len(valid_results)
    
    print(f"\nAverage Relative Error (ppm) across {len(valid_results)} test points:")
    print(f"  Li(x):           {avg_li:>12.2f} ppm")
    print(f"  Cipolla:         {avg_cipolla:>12.2f} ppm")
    print(f"  π_h^(3)(x):      {avg_pi_h:>12.2f} ppm")
    
    # Compute improvement factors
    print(f"\nImprovement Factors (relative to Li(x)):")
    print(f"  Cipolla vs Li:   {avg_li / avg_cipolla:.2f}× better")
    print(f"  π_h vs Li:       {avg_li / avg_pi_h:.2f}× better")
    
    print(f"\nComparison (Cipolla vs π_h):")
    if avg_cipolla < avg_pi_h:
        print(f"  Cipolla is {avg_pi_h / avg_cipolla:.2f}× more accurate than π_h^(3)(x)")
    else:
        print(f"  π_h^(3)(x) is {avg_cipolla / avg_pi_h:.2f}× more accurate than Cipolla")
    
    print("="*80 + "\n")


# ===========================
# PART 6: MAIN EXECUTION
# ===========================

def main():
    """
    Main execution function.
    Run comparison for selected test points.
    """
    print("="*80)
    print("CIPOLLA APPROXIMATION COMPARISON")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Precision: {mp.dps} decimal places")
    print("="*80)
    
    # Configuration
    TEST_POINTS = [5, 10, 12, 15, 18, 20]  # Start with subset for testing
    N_CIPOLLA_TERMS = 20  # Number of Cipolla terms
    N_HYBRID = 3  # Truncation level for π_h
    
    print(f"\nConfiguration:")
    print(f"  Test points (n): {TEST_POINTS}")
    print(f"  Cipolla terms (N): {N_CIPOLLA_TERMS}")
    print(f"  Hybrid level (N): {N_HYBRID}")
    
    # Run comparison
    results = compare_approximations(
        test_points=TEST_POINTS,
        N_cipolla=N_CIPOLLA_TERMS,
        N_hybrid=N_HYBRID
    )
    
    # Save and display results
    save_results_csv(results)
    generate_summary_statistics(results)
    format_latex_table(results)
    
    print(f"\n✓ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run the comparison
    results = main()
    
    print("\nNext steps:")
    print("1. Review the CSV file: cipolla_comparison_results.csv")
    print("2. Copy the LaTeX table code above into your paper")
    print("3. Add discussion of results in Section 5")
    print("4. Update conclusions based on comparison")
