"""
Test convergence properties (Theorem 2 from paper).
"""

import pytest
from src.prime_approximations import pi_h
from mpmath import ln

def test_increasing_N_improves_accuracy():
    """Verify increasing N reduces error (Theorem 2)."""
    x = 1e12
    exact = 37607912018  # π(10^12)

    errors = []
    for N in [1, 2, 3, 4]:
        approx = float(pi_h(x, N))
        error = abs(approx - exact)
        errors.append(error)

    # Errors should generally decrease with N
    assert errors[2] < errors[0], "N=3 should be better than N=1"

def test_convergence_rate():
    """Test geometric decay O((ln x)^(-n))."""
    x = 1e9
    ln_x = float(ln(x))

    # Compute successive approximations
    vals = [float(pi_h(x, N)) for N in range(1, 5)]

    # Differences should decay geometrically
    diffs = [abs(vals[i+1] - vals[i]) for i in range(len(vals)-1)]

    # Ratio should be approximately 1/ln(x)
    ratios = [diffs[i+1]/diffs[i] for i in range(len(diffs)-1)]
    expected_ratio = 1/ln_x

    # Check ratio is within 50% of expected (loose bound)
    for r in ratios:
        assert 0.5*expected_ratio < r < 2*expected_ratio
