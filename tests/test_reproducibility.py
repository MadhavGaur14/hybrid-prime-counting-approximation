"""
Test deterministic reproducibility.
"""

from src.prime_approximations import pi_h, pi_g

def test_deterministic_output():
    """Verify same inputs give same outputs."""
    x = 1e9

    result1 = float(pi_h(x, N=3))
    result2 = float(pi_h(x, N=3))

    assert result1 == result2, "Non-deterministic behavior detected"

def test_precision_stability():
    """Test high precision doesn't introduce errors."""
    from mpmath import mp
    original_dps = mp.dps

    mp.dps = 50
    result_50 = float(pi_h(1e6, N=3))

    mp.dps = 100
    result_100 = float(pi_h(1e6, N=3))

    mp.dps = original_dps

    # Should agree to at least 10 significant figures
    rel_diff = abs(result_50 - result_100) / result_100
    assert rel_diff < 1e-10
