#!/usr/bin/env python3
"""
Test that Branch-and-Bound finds the globally optimal support on a small
synthetic problem where we can brute-force all supports.
"""

import numpy as np
from scipy.linalg import svd
from sr_gb import build_monomial_library, estimate_rank, bb_search, snap_round, evaluate_support
from sympy import parse_expr, groebner, Poly, symbols, simplify
import itertools

def generate_synthetic_problem():
    """
    Create a small problem with a known sparse invariant.
    The data lies on the circle x^2 + y^2 - 1 = 0, with a degree-2 monomial
    library, which gives M=6 (1, x, y, x^2, xy, y^2).
    The invariant is -1 + x^2 + y^2.
    """
    N = 1000
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta)
    y = np.sin(theta)
    data = np.column_stack([x, y])
    # add small noise? For test, no noise.
    var_names = ["x", "y"]
    sym_vars, monomials, evaluate = build_monomial_library(var_names, max_degree=2, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data)
    U, s, Vt = svd(Phi, full_matrices=False)
    r = estimate_rank(s, sigma_estimate=0.0, N=N)
    null_dim = Phi.shape[1] - r
    V_null = Vt[r:, :].T
    d = null_dim
    # The true invariant coefficients: [-1, 0, 0, 1, 0, 1] (for monomials: 1, x, y, x^2, xy, y^2)
    true_c = np.array([-1.0, 0, 0, 1.0, 0, 1.0])
    true_c = true_c / np.linalg.norm(true_c)
    true_support = {3, 5, 0}  # indices of x^2, y^2, constant
    return Phi, V_null, d, true_support, true_c, monomials

def brute_force(V_null, d, k_max, max_denom, eps, sigma_estimate, Phi, N):
    M = V_null.shape[0]
    best_cost = (float('inf'), float('inf'))
    best_S = None
    best_c = None
    # Enumerate all supports up to size k_max
    for k in range(1, k_max+1):
        for S in itertools.combinations(range(M), k):
            S_set = set(S)
            res = evaluate_support(S_set, V_null, d, max_denom, eps, sigma_estimate, Phi, N)
            if res['status'] == 'FEASIBLE' or res['status'] == 'PRIMAL':
                if res['cost'] < best_cost:
                    best_cost = res['cost']
                    best_S = S_set
                    best_c = res['c']
    return best_S, best_c, best_cost

def test_bb_vs_bruteforce():
    Phi, V_null, d, true_support, true_c, monomials = generate_synthetic_problem()
    N = Phi.shape[0]
    M = V_null.shape[0]
    k_max = 4  # support size up to 4
    eps = 1e-3
    sigma_estimate = 0.0
    max_denom = 16

    # Run BB
    best_S_bb, best_c_bb, best_cost_bb = bb_search(
        V_null, d, k_max, max_denom, eps, sigma_estimate, Phi, N, M)

    # Brute force
    best_S_bf, best_c_bf, best_cost_bf = brute_force(V_null, d, k_max, max_denom, eps, sigma_estimate, Phi, N)

    print("BB best cost:", best_cost_bb)
    print("Brute force best cost:", best_cost_bf)
    print("BB support:", best_S_bb)
    print("Brute force support:", best_S_bf)

    # Compare costs with tolerance
    if best_cost_bb[0] == float('inf') and best_cost_bf[0] != float('inf'):
        # BB didn't find a feasible candidate, fallback to L1 -> we'll check support equality
        # but we expect BB to find it.
        pass
    else:
        assert abs(best_cost_bb[0] - best_cost_bf[0]) < 1e-6, f"R differs: {best_cost_bb[0]} vs {best_cost_bf[0]}"
        assert best_cost_bb[1] == best_cost_bf[1], f"k differs: {best_cost_bb[1]} vs {best_cost_bf[1]}"

    # Check supports and coefficients
    assert best_S_bb == best_S_bf, f"Supports differ: {best_S_bb} vs {best_S_bf}"
    if best_c_bb is not None and best_c_bf is not None:
        # Normalize signs
        if np.dot(best_c_bb, best_c_bf) < 0:
            best_c_bb = -best_c_bb
        np.testing.assert_almost_equal(best_c_bb, best_c_bf, decimal=5)

    print("BB matches brute force. Global optimality verified.")

if __name__ == "__main__":
    test_bb_vs_bruteforce()