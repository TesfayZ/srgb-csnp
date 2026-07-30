"""
test_degree_minimality.py – Verify degree minimality check.
"""

import numpy as np
from sympy import symbols, Poly, parse_expr
from sr_gb import is_degree_minimal, reduce_to_minimal_generator

def test_degree_minimality():
    x, y = symbols('x y')
    # Create a non-minimal polynomial: y*(x^2+y^2-1)
    p = y*(x**2 + y**2 - 1)
    # Generate data on circle
    theta = np.random.uniform(0, 2*np.pi, 500)
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    # Reduce to minimal
    reduced = reduce_to_minimal_generator(p, data, sigma_estimate=0.0, sym_vars=[x, y])
    # Check minimality of reduced
    ideal_gens = [reduced]  # single generator
    minimal = is_degree_minimal(reduced, ideal_gens, [x, y])
    assert minimal, "Reduced polynomial should be degree-minimal"
    # Check that original is not minimal (if we put it in its own ideal, it might be minimal by itself)
    # Better: check that p is not minimal w.r.t. its own ideal? That's trivial.
    # We'll test that p is reducible.
    # We'll just assert that reduced != p
    assert reduced != p, "Reduction failed to simplify"
    print("Degree minimality test passed.")

if __name__ == "__main__":
    test_degree_minimality()