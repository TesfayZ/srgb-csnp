#!/usr/bin/env python3
"""
benchmark_oracle_misclassification.py - E-ORACLE: does the rho^A_min=20
gap-test in evaluate_support (the oracle behind Theorem 4.4's dominance-
pruning soundness) correctly classify a TRUE generator's support as
feasible, under noise?

Design note, found empirically before this version: the Feynman degree-2
equations all have d=1, and progressive_nullspace_search takes the raw SVD
column directly for d_try==1 (sr_gb.py, "if d_try == 1: c = V_null_try[:, 0]"),
never calling evaluate_support at all. evaluate_support/_submatrix_rank is
only ever exercised via bb_search, which only runs unconditionally for
d_try>=3. The 2D-oscillator deflation system (benchmark_deflation_multi_invariant.py)
has a genuine d=4 nullspace and is the right target for this diagnostic.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from scipy.linalg import svd
from sympy import parse_expr, Symbol, Poly

from sr_gb import build_monomial_library, evaluate_support
from benchmark_deflation_multi_invariant import generate_2d_harmonic_pairs


def estimate_rank_robust(s, sigma_estimate=0.0, N=None):
    """Pure gap-based rank estimate with a rank-relative regularization floor.

    Kept local to this diagnostic on purpose. sr_gb.estimate_rank now guards
    the gap with min_gap_ratio=2.0 and falls back to an absolute threshold
    when the spectrum has no clear elbow; on this benchmark's noisy
    (sigma=0.02) nullity-4 deflation spectrum that fallback returns rank ~ M-1
    (nullspace dimension 1, not 4), which makes the feasibility oracle declare
    every one of the four true generators Hard Infeasible. The oracle needs
    the plain argmax-gap estimate the deflation benchmark used before its
    migration onto sr_gb(), so it is reproduced here verbatim rather than
    re-coupled to the changed core estimator.
    """
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return 1
    floor = max(s[0] * 1e-9, 1e-300) if s[0] > 0 else 1e-300
    ratios = s[:-1] / (s[1:] + floor)
    gap_idx = int(np.argmax(ratios))
    r = gap_idx + 1
    return max(1, min(r, len(s) - 1))

SIGMA = 0.02
N_SEEDS = 30
N = 5000
DT = 0.1

state_vars = ["x", "y", "vx", "vy"]
sym_vars, monomials, evaluate = build_monomial_library(state_vars, max_degree=2, min_degree=0, scale=False)
monomials_nc = monomials[1:]  # constant column dropped, matches the benchmark
mon_to_idx = {m: i for i, m in enumerate(monomials_nc)}

TRUE_GENERATORS = {
    "x^2+vx^2 (x-energy)": "x**2 + vx**2",
    "y^2+vy^2 (y-energy)": "y**2 + vy**2",
    "x*y+vx*vy (cross term)": "x*y + vx*vy",
    "x*vy-y*vx (ang. momentum)": "x*vy - y*vx",
}


def support_of(expr_str):
    syms = {v: Symbol(v) for v in state_vars}
    expr = parse_expr(expr_str, local_dict=syms)
    poly = Poly(expr, *[syms[v] for v in state_vars])
    S = set()
    for monom_tuple, coeff in poly.terms():
        m = 1
        for i, e in enumerate(monom_tuple):
            if e > 0:
                m *= syms[state_vars[i]] ** e
        assert m in mon_to_idx, f"monomial {m} not in library"
        S.add(mon_to_idx[m])
    return S


def run(n_seeds=N_SEEDS):
    rows = []
    for seed in range(n_seeds):
        data = generate_2d_harmonic_pairs(N=N, dt=DT, sigma=SIGMA, seed=seed)
        old, new = data[:, 0:4], data[:, 4:8]
        Phi_old, _, _ = evaluate(old)
        Phi_new, _, _ = evaluate(new)
        Phi_diff = (Phi_old - Phi_new)[:, 1:]
        M = Phi_diff.shape[1]

        Nrows = Phi_diff.shape[0]
        _, s, Vt = svd(Phi_diff, full_matrices=False)
        r = estimate_rank_robust(s, sigma_estimate=SIGMA, N=N)
        d = M - r
        if d < 1:
            continue
        V_null = Vt[r:, :].T  # (M, d)

        eps = max(1e-4, 3.0 * SIGMA)
        for label, expr_str in TRUE_GENERATORS.items():
            S_star = support_of(expr_str)
            result = evaluate_support(S_star, V_null, d, max_denom=16, eps=eps,
                                       sigma_estimate=SIGMA, Phi=Phi_diff, N=Nrows)
            status = result['status']
            rows.append({"seed": seed, "d": d, "M": M, "generator": label,
                         "status": status,
                         "misclassified_hard_infeasible": status == 'HARD_INFEASIBLE'})

    df = pd.DataFrame(rows)
    summary = df.groupby("generator").agg(
        n_trials=("seed", "count"),
        n_misclassified=("misclassified_hard_infeasible", "sum"),
        status_modes=("status", lambda s: dict(s.value_counts())),
    ).reset_index()
    summary["misclassification_rate"] = summary["n_misclassified"] / summary["n_trials"]

    print("=" * 70)
    print(f"Oracle misclassification on 2D-oscillator deflation (sigma={SIGMA}, {n_seeds} seeds, d=4)")
    print("=" * 70)
    print(summary.to_string(index=False))

    import os
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/oracle_misclassification_full.csv", index=False)
    summary.to_csv("Results/oracle_misclassification_summary.csv", index=False)
    print("\nSaved to Results/oracle_misclassification_{full,summary}.csv")
    return df, summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        n_seeds = 2
    elif len(sys.argv) > 1:
        n_seeds = int(sys.argv[1])
    else:
        n_seeds = N_SEEDS
    run(n_seeds=n_seeds)
