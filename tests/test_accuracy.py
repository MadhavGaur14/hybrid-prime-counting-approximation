"""
Test accuracy against known π(x) values from paper Table 3.
"""

import pytest
from src.prime_approximations import pi_h, pi_g, EXACT_PI_VALUES, compute_relative_error_ppm

def test_pi_h_accuracy_10_9():
    """Test π_h(10^9) matches paper results."""
    x = 1e9
    result = float(pi_h(x, N=3))
    exact = EXACT_PI_VALUES[9]

    error_ppm = compute_relative_error_ppm(result, exact)

    # From paper Table 3: should be ~492 ppm
    assert error_ppm < 1000, f"Error too large: {error_ppm} ppm"
    assert abs(result - exact) < 40000, "Absolute error too large"

def test_all_exact_values():
    """Test against all known π(10^n) values."""
    for n in range(5, 21):  # Test 10^5 to 10^20
        x = 10**n
        exact = EXACT_PI_VALUES[n]
        approx = float(pi_h(x, N=3))

        rel_error = abs(approx - exact) / exact
        assert rel_error < 0.01, f"Failed at 10^{n}: {rel_error:.4%}"
