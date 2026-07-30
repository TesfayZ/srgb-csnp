#!/usr/bin/env python3
"""
test_l1_fallback_4sphere.py — Validate CSNP's L1 fallback on the 4-sphere invariant
(degree-3 library, M=35).

Benchmark: 4-variable degree-3 monomial library (M=35 monomials) containing
the 4-sphere invariant x^2+y^2+z^2+w^2-1 (5 active monomials).
C(35,5)*d^2 ~ 8e6 > 10^6 → combinatorial CSNP infeasible; L1 fallback invoked.
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from itertools import combinations_with_replacement
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_gb import estimate_rank, snap_round, _l1_nullspace_fallback
from utils_stats import wilson_interval
from sympy import (symbols, Rational, parse_expr, cancel,
                   groebner, Poly, simplify, factor)
import warnings
warnings.filterwarnings('ignore')

x_s, y_s, z_s, w_s = symbols("x y z w")
SYM_VARS = [x_s, y_s, z_s, w_s]


def generate_4sphere_data(N=5000, sigma=0.0, seed=42):
    """Uniform samples on the 4-sphere x^2+y^2+z^2+w^2=1."""
    np.random.seed(seed)
    phi_a = np.random.uniform(0, np.pi, N)
    phi_b = np.random.uniform(0, np.pi, N)
    theta  = np.random.uniform(0, 2*np.pi, N)
    x = np.sin(phi_a)*np.sin(phi_b)*np.cos(theta)
    y = np.sin(phi_a)*np.sin(phi_b)*np.sin(theta)
    z = np.sin(phi_a)*np.cos(phi_b)
    w = np.cos(phi_a)
    data = np.column_stack([x, y, z, w])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data


def build_phi(data, max_degree=3):
    """Build monomial design matrix up to given degree."""
    N, n_vars = data.shape
    exponents = []
    for d in range(0, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            exp = [0] * n_vars
            for idx in combo:
                exp[idx] += 1
            exponents.append(tuple(exp))
    M = len(exponents)
    Phi = np.ones((N, M))
    zero_exp = tuple([0] * n_vars)
    for i, exp in enumerate(exponents):
        if exp == zero_exp:
            continue
        val = np.ones(N)
        for vi, e in enumerate(exp):
            if e > 0:
                val *= data[:, vi] ** e
        Phi[:, i] = val
    return Phi, exponents


def build_polynomial(rounded, exponents):
    poly = 0
    for i, exp in enumerate(exponents):
        if rounded[i] == 0:
            continue
        m = 1
        for vi, e in enumerate(exp):
            if e > 0:
                m *= SYM_VARS[vi] ** e
        poly += Rational(rounded[i]) * m
    return poly


def exact_match(poly, target):
    if poly == 0:
        return False
    try:
        ratio = cancel(poly / target)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return False


def run_one(N, sigma, seed, max_degree=3):
    data = generate_4sphere_data(N=N, sigma=sigma, seed=seed)
    Phi, exponents = build_phi(data, max_degree=max_degree)
    M = Phi.shape[1]

    U, sv, Vt = svd(Phi, full_matrices=False)
    r = estimate_rank(sv, sigma_estimate=sigma, N=N)
    d_null = M - r
    V_null = Vt[r:, :].T  # shape (M, d_null)

    from math import comb as math_comb
    k_expected = 5   # x^2+y^2+z^2+w^2-1 has 5 terms
    n_subsets = math_comb(M, k_expected) * d_null**2
    mode = "L1 fallback" if n_subsets > 1e6 else "Combinatorial (C={:,})".format(math_comb(M, k_expected))

    tau_resid = max(1e-3, sigma * np.sqrt(N) * 3.0)
    c = _l1_nullspace_fallback(V_null, Phi, tau_resid)
    if c is None:
        return {"seed": seed, "sigma": sigma, "N": N, "M": M, "d_null": d_null,
                "mode": mode, "exact": False, "sparsity": -1, "residual": -1}

    c[np.abs(c) < 1e-3] = 0.0
    rounded = snap_round(c, sigma_estimate=sigma)
    poly = build_polynomial(rounded, exponents)
    sphere4 = parse_expr("x**2 + y**2 + z**2 + w**2 - 1")
    matched = exact_match(poly, sphere4)
    sparsity = int(np.sum(np.abs(c) > 1e-2))
    resid = float(np.linalg.norm(Phi @ c))

    return {"seed": seed, "sigma": sigma, "N": N, "M": M, "d_null": d_null,
            "mode": mode, "exact": matched, "sparsity": sparsity,
            "residual": resid, "poly": str(poly)}


def test_l1_fallback(noise_levels=(0.00, 0.01, 0.02, 0.05), n_seeds=30):
    print("L1 Fallback Validation")
    print("Problem: 4-sphere x²+y²+z²+w²=1, degree-3 library (M=35)")
    print("C(35,5)*d² ~ 8×10⁶ > 10⁶  →  L1 fallback automatically invoked")
    print("=" * 65)

    results = []
    for sigma in noise_levels:
        for seed in range(n_seeds):
            row = run_one(N=5000, sigma=sigma, seed=seed, max_degree=3)
            matched = row["exact"]
            print(f"  σ={sigma:.3f} seed={seed}: exact={matched}, "
                  f"sparsity={row['sparsity']}, resid={row['residual']:.2e}, "
                  f"mode={row['mode']}")
            results.append(row)

    df = pd.DataFrame(results)
    summary = df.groupby("sigma").agg(
        exact_rate=("exact", "mean"),
        mean_sparsity=("sparsity", "mean"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    for idx, row in summary.iterrows():
        sigma_val = row['sigma']
        k = df[df['sigma'] == sigma_val]['exact'].sum()
        ci = wilson_interval(k, n_seeds)
        summary.loc[idx, 'ci_low'] = ci[0]
        summary.loc[idx, 'ci_high'] = ci[1]
    print(summary[['sigma', 'exact_rate', 'ci_low', 'ci_high']].round(3))
    print("\nSummary (exact recovery rate):")
    print(summary.to_string(index=False))
    df.to_csv("Results/l1_fallback_4sphere_results.csv", index=False)

    # Noise-free sanity check: at sigma=0.0 the L1 fallback must recover the
    # true 4-sphere invariant on every seed (this is the regression bar; the
    # noisier rows are a robustness sweep, not a per-run pass/fail gate;
    # see the BLAS-variance gotcha in TODO.md).
    zero_noise_rate = summary.loc[summary['sigma'] == 0.0, 'exact_rate'].iloc[0]
    assert zero_noise_rate == 1.0, (
        f"L1 fallback should recover the 4-sphere invariant on every seed at "
        f"sigma=0.0; got exact_rate={zero_noise_rate}"
    )
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same noise levels as full run")
    args = parser.parse_args()
    if args.quick:
        test_l1_fallback(n_seeds=3)
    else:
        test_l1_fallback()
