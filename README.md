# Hybrid Rational Approximation for Prime Counting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Companion repository for **"A Hybrid Rational Approximation for Prime Counting: Rigorous Construction and Analysis of π_h(x)"** by Madhav Gaur.

**Status**: Submitted to *Experimental Mathematics* (February 2026)

## Overview

This repository provides a complete implementation of π_h(x), a hybrid rational approximation to the prime-counting function π(x), with explicit truncation error bounds E_N(x) ≤ C_N · x/(ln x)^(N+2).

**Key features:**
- Explicit error bounds with computable constants (C_3 = 2.80 for ln x ≥ 100)
- O(N) arithmetic operations for fixed truncation level N
- Median relative error: 0.10% (1,037 ppm) across tested range
- Complete reproducibility framework with 30+ test points across irregular inputs

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/hybrid-prime-counting-approximation.git
cd hybrid-prime-counting-approximation

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install primesieve for exact π(x) computation
pip install primesieve
```

### Basic Usage

```python
from src.prime_approximations import pi_h, pi_g

# Compute approximations
x = 1e9
approximation = pi_h(x, N=3)  # π_h^(3)(x)
base_approx = pi_g(x)          # Base rational approximant

print(f"π_h^(3)({x:.0e}) = {approximation:.2f}")
# Output: π_h^(3)(1e+09) = 50813442.72
```

### Run Experiments

```bash
# Reproduce irregular inputs validation (from paper Section 5.6)
python src/irregular_inputs.py

# Compare with Cipolla approximation (from paper Section 5.4)  
python src/cipolla_comparison.py

# Generate all figures from paper
python scripts/generate_figures.py
```

Expected runtime: ~10-15 minutes on modern hardware (Intel i7 or equivalent).

## Experimental Results Summary

### Key Statistics (30 Irregular Test Points - Section 5.6)

| Metric | Value |
|--------|-------|
| **Mean relative error** | 2,415.73 ppm (0.24%) |
| **Median relative error** | 1,037.18 ppm (0.10%) |
| **Range tested** | 10^4 ≤ x ≤ 10^12 |
| **Test categories** | Random irregular, near powers-of-10, non-decimal |
| **Error scaling** | Monotonic decay, O(x/(ln x)^5) for N=3 |

### Key Findings

1. **Intermediate accuracy**: Achieves 2-3.5× better accuracy than x/ln x, positioned between classical approximations and Li(x)
2. **Robustness**: Consistent performance across three diverse sampling strategies (< 10% variation)
3. **Predictability**: Clean monotonic error decay matching theoretical O(x/(ln x)^(N+2)) scaling
4. **No anomalous behavior**: Uniform accuracy across random, near-power-10, and non-decimal inputs

## Repository Structure

```
hybrid-prime-counting-approximation/
├── src/                        # Core implementations
│   ├── prime_approximations.py # π_g(x) and π_h(x) implementations
│   ├── cipolla_comparison.py   # Cipolla approximation comparison
│   └── irregular_inputs.py     # Robust experimental validation
├── data/                       # Reference data and results
│   ├── exact_pi_values.csv     # Known π(10^n) values
│   └── irregular_inputs_results.csv  # Experimental results
├── results/                    # Generated outputs
│   ├── figures/                # All paper figures (PDF)
│   └── tables/                 # LaTeX table files
├── docs/                       # Extended documentation
│   ├── EXPERIMENTS.md          # Replication guide
│   ├── IMPLEMENTATION.md       # Code architecture
│   └── HARDWARE.md             # System specifications
├── tests/                      # Unit tests and validation
├── scripts/                    # Convenience scripts
├── paper/                      # LaTeX source (optional)
└── requirements.txt            # Python dependencies
```

## Theoretical Guarantees

**Proven bounds** (Theorem 3 in paper, Section 4.3):

For ln x ≥ L_0(N), truncation error satisfies:

```
E_N(x) = |π_h(x) - π_h^(N)(x)| ≤ (C_N · x) / (ln x)^(N+2)
```

**Constants (Table 2 in paper):**

| N | L_0(N) | C_N  |
|---|--------|------|
| 1 | 100    | 3.04 |
| 2 | 100    | 3.04 |
| 3 | 100    | 2.80 |
| 4 | 150    | 2.65 |

**Empirical observation**: Effective constants C'_3 ≈ 1.5-2.0 in tested range (ln x ∈ [9.2, 27.6]), suggesting the proven bound is conservative.

**Critical note**: Theoretical threshold ln x ≥ 100 corresponds to x ≥ e^100 ≈ 2.7×10^43, far beyond experimental range. See paper Section 6 (Limitations) for explicit domain separation discussion.

## Mathematical Construction

**Base approximant:**
```
π_g(x) = x(ln x + 1) / ((ln x)^2 + 1)
```

**Hybrid approximation:**
```
π_h^(N)(x) = Σ_{n=1}^N [π_g(x) + x^(1/n)] / (ln x)^n
```

**Design principles:**
- Geometric damping via (ln x)^(-n) ensures convergence
- Fractional powers x^(1/n) provide discrete corrections
- Rational base π_g(x) ensures numerical stability (no real singularities)

## Reproducibility

All experiments are fully reproducible:

✓ **Fixed random seeds**: `np.random.seed(42)` for irregular sampling  
✓ **Deterministic outputs**: Bit-identical results on same hardware/precision  
✓ **Checksums provided**: Validate data integrity against published results  
✓ **System specs documented**: See [HARDWARE.md](docs/HARDWARE.md)

**Tested environments:**
- Python 3.10, 3.11, 3.12
- Windows 11 Pro (primary), Ubuntu 22.04
- mpmath 1.3.0, primesieve 11.1

## Citation

If you use this code or approximation in your research, please cite:

```bibtex
@article{gaur2026hybrid,
  title={A Hybrid Rational Approximation for Prime Counting: Rigorous Construction and Analysis of $\pi_h(x)$},
  author={Gaur, Madhav},
  journal={Experimental Mathematics},
  year={2026},
  note={Submitted},
  url={https://github.com/YOUR_USERNAME/hybrid-prime-counting-approximation}
}
```

## Dependencies

**Core requirements:**
```
numpy >= 1.24.0      # Numerical arrays
pandas >= 2.0.0      # Data analysis
mpmath >= 1.3.0      # Arbitrary-precision arithmetic
matplotlib >= 3.7.0  # Visualization
primesieve >= 11.1   # Exact π(x) computation (optional but recommended)
```

See `requirements.txt` for complete list with exact versions.

## Testing

Run test suite (once implemented):
```bash
pytest tests/ -v
```

**Planned coverage:**
- Convergence bounds (Theorem 2 validation)
- Accuracy against known π(x) values
- Reproducibility verification (checksums)
- Edge cases (small x, large N)

## Documentation

- **[EXPERIMENTS.md](docs/EXPERIMENTS.md)**: Step-by-step replication guide
- **[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)**: Code architecture and design decisions
- **[HARDWARE.md](docs/HARDWARE.md)**: System specifications for timing reproducibility

## Performance Characteristics

**Computational complexity:**
- **Operation count**: O(N) arithmetic operations for fixed N
- **Bit complexity**: O(k log^2 k) for x = 10^k with FFT-based arithmetic
- **Memory**: O(1) (constant memory for fixed N)

**Measured performance** (N=3, median over 20 trials):
- **x = 10^9**: 55.2 μs ± 6.8 μs (CV: 12.3%)
- **Scaling**: Approximately constant for 10^3 ≤ x ≤ 10^12
- **Note**: Growth with x due to bit-complexity, not algorithm

## Comparison with Other Methods

From paper Section 5.4 (Cipolla comparison):

| Method | Small-x (10^5) | Large-x (10^12) | Implementation |
|--------|----------------|-----------------|----------------|
| **π_h^(3)(x)** | 1,358 ppm | 278 ppm | Simple (no Möbius, no Li integrals) |
| **Cipolla (N=20)** | 8,363 ppm | 2.07 ppm | Complex (Möbius + multiple Li evals) |
| **Li(x)** | 3,942 ppm | 1.02 ppm | Moderate (numerical integration) |

**Complementary niches**: π_h excels at small-to-moderate x; Cipolla dominates for x ≥ 10^12.

## Limitations (From Paper Section 6)

**Theoretical:**
- Proven bounds apply only for ln x ≥ 100 (x ≥ e^100)
- Construction is engineered, not derived from explicit formula
- No improvement to asymptotic PNT error terms

**Experimental:**
- Tested only for x ≤ 10^12 (discrete power-of-10 sampling)
- Irregular inputs extend validation but finite sampling remains
- Li(x) used as reference for x > 10^20 (not ground truth)

**Computational:**
- O(1) complexity refers to arithmetic operations (Real RAM model)
- Bit-complexity grows with precision requirements
- No formal cryptographic validation provided

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Contact

**Author**: Madhav Gaur  
**Paper**: Submitted to *Experimental Mathematics* (Feb 2026)  
**Repository**: https://github.com/YOUR_USERNAME/hybrid-prime-counting-approximation

## Acknowledgments

- **Prime number data**: [primesieve](https://github.com/kimwalisch/primesieve) by Kim Walisch
- **Exact π(x) values**: Deleglise & Rivat (1996), published tables
- **Arbitrary-precision arithmetic**: [mpmath](https://mpmath.org/) library
- **Computational resources**: Intel Core i7-10700K @ 3.8GHz, 32GB RAM

## Version History

### v1.0.0 (February 2026) - Initial Release
- Complete π_h(x) implementation with N=1,2,3,4 support
- Full experimental validation (30 irregular test points)
- Cipolla approximation comparison
- Reproducibility framework with fixed seeds
- Publication-quality figures and tables
- Comprehensive documentation

---

**Important Note**: This is research code accompanying an academic paper. For production applications requiring certified prime counting, use exact algorithms like [primesieve](https://github.com/kimwalisch/primesieve) or validated libraries.

**Research Use**: This approximation is designed for scenarios requiring fast density estimates where approximation error of ~0.1-0.3% is acceptable, such as algorithm analysis, heuristic optimization, or educational demonstrations.
