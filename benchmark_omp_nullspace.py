#!/usr/bin/env python3
"""
benchmark_omp_nullspace.py – Compare OMP vs CSNP on nullspace recovery.

Tests:
  1. Harmonic oscillator (fixed dt) 
  2. Kepler angular momentum 

Runs 30 seeds per problem and saves results to CSV.
"""

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sympy import parse_expr
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sr_gb import (
    sr_gb, exact_recovery, build_monomial_library,
    snap_round, estimate_rank, Rational, groebner, Poly, symbols, simplify
)
# Correct import: function returns (pairs, trajectory); we only need pairs
from benchmark_harmonic_oscillator_vs_sindy import generate_harmonic_trajectory
from benchmark_kepler_angular_momentum import generate_kepler_pairs
from utils_stats import wilson_interval


def omp_nullspace(data, var_names, degree=2, sigma_estimate=0.0, eps=1e-3):
    """
    OMP baseline: apply Orthogonal Matching Pursuit to the nullspace basis.
    This is the baseline that fails on multi-dimensional nullspaces.
    """
    from sr_gb import _column_preconditioned_svd, _orthonormal_null_basis

    sym_vars, monomials, evaluate = build_monomial_library(
        var_names, degree, min_degree=0, scale=False
    )
    Phi, _, _ = evaluate(data)
    N, M = Phi.shape

    # Same column-preconditioned SVD front-end SR-GB+CSNP uses, so the
    # comparison is fair; OMP's structural failure is downstream of the
    # nullspace estimate, not in how the nullspace is computed.
    s, Vt = _column_preconditioned_svd(Phi)
    r = estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d = M - r

    if d <= 1:
        # If null_dim=1, OMP would just return the last column
        c = Vt[-1, :].copy()
    else:
        V_null = _orthonormal_null_basis(Vt, M, d)  # shape (M, d)
        
        # OMP: find sparse combination of columns of V_null
        # We want a sparse vector c such that c = V_null @ alpha
        # This is a sparse recovery problem: find sparse alpha such that V_null @ alpha is sparse
        # Simplest: apply OMP to the columns of V_null to build a sparse alpha
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(4, d), fit_intercept=False)
        
        # Use the last column as target (arbitrary) – this is the flawed approach
        # The structural issue is that OMP doesn't enforce V_null[notS] @ alpha = 0
        target = V_null[:, 0]  # arbitrary column
        
        # Fit OMP on the remaining columns
        X = V_null[:, 1:]
        if X.shape[1] > 0:
            omp.fit(X, target)
            alpha = np.zeros(d)
            alpha[0] = 1.0
            alpha[1:] = omp.coef_
        else:
            alpha = np.array([1.0])
        
        c = V_null @ alpha
        norm = np.linalg.norm(c)
        if norm > 1e-10:
            c = c / norm
        else:
            c = Vt[-1, :].copy()  # fallback
    
    # Post-processing (same as SR-GB+CSNP)
    c[np.abs(c) < eps] = 0.0
    rounded = snap_round(c, sigma_estimate=sigma_estimate)
    c_rounded = [Rational(val) for val in rounded]
    
    poly_expr = sum(coef * mon for coef, mon in zip(c_rounded, monomials) if coef != 0)
    if poly_expr == 0:
        return []
    
    poly_expr = poly_expr.expand()
    sym_vars = symbols(var_names)
    
    # Normalise leading coefficient
    terms = poly_expr.as_ordered_terms()
    lead_term = None
    max_deg_seen = -1
    for term in terms:
        if term == 0:
            continue
        tdeg = 0 if term.is_Number else int(sum(term.as_powers_dict().values()))
        if tdeg > max_deg_seen:
            max_deg_seen = tdeg
            lead_term = term
    if lead_term is not None:
        lead_coef = lead_term.as_coeff_Mul()[0] if not lead_term.is_Number else lead_term
        if lead_coef != 0:
            poly_expr = simplify(poly_expr / lead_coef)
    
    gb = groebner([Poly(poly_expr, *sym_vars)], *sym_vars, order='grevlex')
    return list(gb)


def run_omp_benchmark(n_seeds=30, N=5000, dt=0.1, sigma=0.0):
    """Run OMP vs CSNP on harmonic oscillator and Kepler."""
    
    # ---- Harmonic oscillator ----
    var_names_ho = ["x_t", "v_t", "x_next", "v_next"]
    true_ho = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")
    
    ho_results = []
    for seed in range(n_seeds):
        # generate_harmonic_trajectory returns (pairs, trajectory); we only need pairs
        pairs, _ = generate_harmonic_trajectory(N=N, dt=dt, sigma=sigma, seed=seed)
        
        # CSNP – adaptive (degree=None, D_max=2)
        try:
            gb_csnp = sr_gb(pairs, var_names_ho, degree=None, D_max=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  harmonic_oscillator seed {seed}: sr_gb error: {e}")
            gb_csnp = []
        csnp_exact = exact_recovery(gb_csnp, true_ho)

        # OMP
        try:
            gb_omp = omp_nullspace(pairs, var_names_ho, degree=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  harmonic_oscillator seed {seed}: omp_nullspace error: {e}")
            gb_omp = []
        omp_exact = exact_recovery(gb_omp, true_ho)
        
        ho_results.append({
            "seed": seed,
            "csnp_exact": csnp_exact,
            "omp_exact": omp_exact,
            "problem": "harmonic_oscillator"
        })
    
    # ---- Kepler ----
    var_names_kep = ["x_t", "y_t", "vx_t", "vy_t", "x_next", "y_next", "vx_next", "vy_next"]
    true_kep = parse_expr("x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next")
    
    kep_results = []
    for seed in range(n_seeds):
        data = generate_kepler_pairs(N_pairs=N, dt=dt, seed=seed)
        
        # CSNP – adaptive (degree=None, D_max=2)
        try:
            gb_csnp = sr_gb(data, var_names_kep, degree=None, D_max=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  kepler seed {seed}: sr_gb error: {e}")
            gb_csnp = []
        csnp_exact = exact_recovery(gb_csnp, true_kep)

        # OMP
        try:
            gb_omp = omp_nullspace(data, var_names_kep, degree=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  kepler seed {seed}: omp_nullspace error: {e}")
            gb_omp = []
        omp_exact = exact_recovery(gb_omp, true_kep)
        
        kep_results.append({
            "seed": seed,
            "csnp_exact": csnp_exact,
            "omp_exact": omp_exact,
            "problem": "kepler"
        })
    
    # Combine results
    df = pd.DataFrame(ho_results + kep_results)
    
    # Summary
    summary = df.groupby("problem").agg(
        csnp_rate=("csnp_exact", "mean"),
        omp_rate=("omp_exact", "mean"),
        n_seeds=("seed", "count")
    ).reset_index()
    
    # Compute Wilson CIs
    for idx, row in summary.iterrows():
        problem = row["problem"]
        mask = df["problem"] == problem
        csnp_k = df[mask]["csnp_exact"].sum()
        omp_k = df[mask]["omp_exact"].sum()
        n = len(df[mask])
        csnp_ci = wilson_interval(csnp_k, n)
        omp_ci = wilson_interval(omp_k, n)
        summary.loc[idx, "csnp_ci_low"] = csnp_ci[0]
        summary.loc[idx, "csnp_ci_high"] = csnp_ci[1]
        summary.loc[idx, "omp_ci_low"] = omp_ci[0]
        summary.loc[idx, "omp_ci_high"] = omp_ci[1]
    
    print("\n" + "=" * 70)
    print("OMP vs CSNP Nullspace Recovery Benchmark")
    print("=" * 70)
    print(f"Seeds per problem: {n_seeds}")
    print(f"Samples per run: {N}")
    print("\n" + summary.to_string(index=False))
    print("=" * 70)
    
    # Save results
    df.to_csv("Results/omp_nullspace_results.csv", index=False)
    summary.to_csv("Results/omp_nullspace_summary.csv", index=False)
    print("\nResults saved to Results/omp_nullspace_*.csv")
    
    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N as full run")
    args = parser.parse_args()
    if args.quick:
        print("Running OMP vs CSNP benchmark (quick: 2 seeds)...")
        df, summary = run_omp_benchmark(n_seeds=2, N=5000, dt=0.1, sigma=0.0)
    else:
        print("Running OMP vs CSNP benchmark with 30 seeds...")
        df, summary = run_omp_benchmark(n_seeds=30, N=5000, dt=0.1, sigma=0.0)
    
    omp_rate = summary[summary["problem"] == "harmonic_oscillator"]["omp_rate"].values[0]
    print(f"\nOMP harmonic oscillator recovery: {omp_rate:.0%} ")
    omp_rate = summary[summary["problem"] == "kepler"]["omp_rate"].values[0]
    print(f"OMP Kepler recovery: {omp_rate:.0%} ")