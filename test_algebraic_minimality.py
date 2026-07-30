#!/usr/bin/env python3
"""
Explicit test of algebraic minimality: reduce y*(x^2+y^2-1) to x^2+y^2-1.
"""

import numpy as np
from sympy import symbols, parse_expr, factor, simplify, Poly
from sr_gb import reduce_to_minimal_generator, exact_recovery
from sr_gb import build_monomial_library, sr_gb

def test_minimality():
    x, y = symbols('x y')
    # Create data on the circle
    theta = np.random.uniform(0, 2*np.pi, 500)
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    # Add small noise
    data += np.random.normal(0, 1e-6, data.shape)

    # Generate polynomial p = y*(x^2+y^2-1)
    p = y*(x**2 + y**2 - 1)
    # We'll simulate that this is the recovered polynomial, then apply reduction.
    reduced = reduce_to_minimal_generator(p, data, sigma_estimate=1e-6, sym_vars=[x, y])
    print("Original:", p)
    print("Reduced:", reduced)
    # Check if reduced equals x^2+y^2-1 (up to sign)
    true_p = x**2 + y**2 - 1
    ratio = simplify(reduced / true_p)
    assert ratio.is_number or ratio.is_constant(), (
        f"FAIL: minimality did not reduce correctly. "
        f"Original={p}, reduced={reduced}, expected {true_p} (up to a scalar)."
    )
    print("SUCCESS: minimality reduced to true generator.")

if __name__ == "__main__":
    test_minimality()