#!/usr/bin/env python3
"""
Demonstrate that the rationality cost resolves the dt-induced nullspace ambiguity
in the harmonic oscillator with fixed dt.
"""

import numpy as np
from scipy.linalg import svd
from sr_gb import build_monomial_library, estimate_rank, rationality_cost, snap_round
from sympy import Rational, symbols
import pandas as pd

def generate_harmonic_pairs(N=5000, dt=0.1, sigma=0.0, seed=42):
    np.random.seed(seed)
    x_t = np.random.uniform(-1, 1, N)
    v_t = np.random.uniform(-1, 1, N)
    c = np.cos(dt)
    s = np.sin(dt)
    x_n = c * x_t + s * v_t
    v_n = -s * x_t + c * v_t
    data = np.column_stack([x_t, v_t, x_n, v_n])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data

def test_rationality_ambiguity():
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    data = generate_harmonic_pairs(N=5000, dt=0.1, sigma=0.0, seed=0)
    sym_vars, monomials, evaluate = build_monomial_library(var_names, max_degree=2, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data)
    U, s, Vt = svd(Phi, full_matrices=False)
    r = estimate_rank(s, sigma_estimate=0.0, N=Phi.shape[0])
    d = Phi.shape[1] - r
    print(f"Nullspace dimension: {d}")
    V_null = Vt[r:, :].T  # (M, d)

    # Enumerate all supports of size 5 (the true invariant has 5 terms)
    from itertools import combinations
    best_candidates = []
    for S in combinations(range(Phi.shape[1]), 5):
        S_set = set(S)
        notS = [i for i in range(Phi.shape[1]) if i not in S_set]
        A = V_null[notS, :]
        Ua, sa, Vta = svd(A, full_matrices=False)
        if len(sa) >= d and np.sum(sa < 1e-8) == 1:  # rank = d-1
            alpha = Vta[-1, :]
            c = V_null @ alpha
            c = c / np.linalg.norm(c)
            R = rationality_cost(c, max_denom=16, eps=1e-3)
            if R != float('inf'):
                best_candidates.append((S_set, R, c))
                print(f"Support {S_set} cost {R}")

    # Among candidates, find the one with lowest R.
    if best_candidates:
        best = min(best_candidates, key=lambda x: x[1])
        print("\nBest by rationality cost:")
        print(f"Support: {best[0]}, cost: {best[1]}")
        print("Coefficients (rounded):", [float(Rational(v).limit_denominator(16)) for v in snap_round(best[2], 0.0)])
        # Correct expected support: constant, x_t^2, v_t^2, x_next^2, v_next^2
        # Indices: 0 (1), 5 (x_t^2), 9 (v_t^2), 12 (x_next^2), 14 (v_next^2)
        expected_support = {0, 5, 9, 12, 14}
        assert best[0] == expected_support, (
            f"Rationality cost did not select the true energy invariant: "
            f"got support {best[0]} (cost {best[1]}), expected {expected_support}"
        )
        print("SUCCESS: Rationality selected the true energy invariant.")
    else:
        assert False, "No feasible candidates found."

if __name__ == "__main__":
    test_rationality_ambiguity()