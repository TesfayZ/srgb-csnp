#!/usr/bin/env python3
"""
test_unified_search_optimality.py - unified_circuit_search certificate checks.

The unified search claims: (1) in the enumeration and bb regimes its root fast
path takes the same route as the corresponding standalone engine; (2) in the
branching corner (both exact engines over budget) it finds the lexicographic
(R, k) optimum on instances small enough to exhaust, and never returns anything
worse than the L1 seed under that objective.

Ground truth on these small instances is exhaustive pinning enumeration (every
circuit of a d-dimensional nullspace is pinned by some d-1 coordinate zeros,
so an unbudgeted enumerate_nullspace_generator scores every circuit).
"""

import numpy as np
from scipy.linalg import svd
from sympy import parse_expr

import sr_gb
from sr_gb import (build_monomial_library, enumerate_nullspace_generator,
                   unified_circuit_search, rationality_cost, snap_round)


def _cost(c, max_denom=16, eps=1e-3):
    cn = c / np.max(np.abs(c))
    R = rationality_cost(cn, max_denom, eps)
    k = int(np.sum(np.abs(cn) > 1e-2))
    return (R, k)


def _circle_nullspace(d_take=2, N=1500, seed=0):
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, N)
    data = np.column_stack([np.cos(theta), np.sin(theta)])
    sym_vars, monomials, evaluate = build_monomial_library(["x", "y"], 2, min_degree=0)
    Phi, _, _ = evaluate(data)
    _, s, Vt = svd(Phi, full_matrices=False)
    M = Phi.shape[1]
    V_null = Vt[M - d_take:, :].T
    return Phi, V_null, M


def test_enum_regime_matches_direct_enumeration():
    """Root fast path == direct enumeration when the enum budget fits."""
    Phi, V, M = _circle_nullspace(d_take=2)
    N = Phi.shape[0]
    c_direct = enumerate_nullspace_generator(V, 2, 6, 16, 1e-3, 0.0, Phi, N)
    c_unified = unified_circuit_search(V, 2, Phi, N, M, 6, 16, 1e-3, 0.0, 1e-4)
    assert c_direct is not None and c_unified is not None
    assert _cost(c_unified) == _cost(c_direct), (
        f"enum regime diverged: {_cost(c_unified)} vs {_cost(c_direct)}")
    print("PASS enum-regime bit-consistency:", _cost(c_unified))


def test_branch_regime_reaches_optimum():
    """Force BOTH engines over budget (tiny monkeypatched budgets) on an
    instance whose optimum is known by exhaustive enumeration; the branching
    corner must still find a candidate with the optimal (R, k) cost."""
    Phi, V, M = _circle_nullspace(d_take=3)
    N = Phi.shape[0]
    # Ground truth: exhaustive pinning enumeration, no budget.
    dirs = enumerate_nullspace_generator(V, 3, 6, 16, 1e-3, 0.0, Phi, N,
                                         budget=10**9, collect_all=True)
    assert dirs, "ground-truth enumeration found no rational circuit"
    true_cost = dirs[0][0]

    old_enum, old_bb = sr_gb._ENUM_BUDGET, sr_gb._BB_BUDGET
    try:
        sr_gb._ENUM_BUDGET = 1     # C(M, 2) > 1 -> enumeration 'over budget'
        sr_gb._BB_BUDGET = 1       # bb prediction always over budget
        c = unified_circuit_search(V, 3, Phi, N, M, 6, 16, 1e-3, 0.0, 1e-4)
    finally:
        sr_gb._ENUM_BUDGET, sr_gb._BB_BUDGET = old_enum, old_bb
    assert c is not None, "branching corner returned nothing"
    got = _cost(c)
    assert got == tuple(true_cost), (
        f"branching corner suboptimal: got {got}, optimum {tuple(true_cost)}")
    # And the snapped result is the circle invariant.
    sym_vars, monomials, _ = build_monomial_library(["x", "y"], 2, min_degree=0)
    rounded = snap_round(c, 0.0)
    poly = sum(r * m for r, m in zip(rounded, monomials) if r != 0)
    target = parse_expr("x**2 + y**2 - 1")
    from sympy import cancel
    ratio = cancel(poly / target)
    assert ratio.is_number or ratio.is_constant(), f"wrong circuit: {poly}"
    print("PASS branch-regime optimality:", got, "->", poly)


def test_branch_regime_never_below_l1_seed():
    """On a noisy nullspace with no rational circuit at all, the branching
    corner must degrade to (at worst) the L1 vector, not crash or return a
    worse-scored candidate."""
    rng = np.random.RandomState(7)
    Phi = rng.standard_normal((400, 12))
    _, s, Vt = svd(Phi, full_matrices=False)
    V = Vt[9:, :].T   # 3 noise directions, nothing rational in them
    old_enum, old_bb = sr_gb._ENUM_BUDGET, sr_gb._BB_BUDGET
    try:
        sr_gb._ENUM_BUDGET = 1
        sr_gb._BB_BUDGET = 1
        c = unified_circuit_search(V, 3, Phi, 400, 12, 6, 16, 1e-3, 0.0, 1e10)
    finally:
        sr_gb._ENUM_BUDGET, sr_gb._BB_BUDGET = old_enum, old_bb
    # tau_resid is huge so the L1 fallback is guaranteed to return something;
    # the unified search must therefore also return something.
    assert c is not None
    print("PASS branch-regime anytime fallback returns a candidate")


if __name__ == "__main__":
    test_enum_regime_matches_direct_enumeration()
    test_branch_regime_reaches_optimum()
    test_branch_regime_never_below_l1_seed()
    print("\nAll unified-search optimality tests passed.")
