#!/usr/bin/env python3
"""
Noise ablation for circle invariant with 30 seeds per sigma level.
Uses snap_round and optionally original limit_denominator.
"""

import numpy as np
import pandas as pd
from sympy import parse_expr, Rational
from sr_gb import sr_gb, exact_recovery, build_monomial_library
from utils_stats import wilson_interval
import time
from fractions import Fraction

def generate_circle(N=5000, sigma=0.0, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta)
    y = np.sin(theta)
    if sigma > 0:
        x += np.random.normal(0, sigma, N)
        y += np.random.normal(0, sigma, N)
    return np.column_stack([x, y])

def original_sr_gb(data, var_names, degree=2, max_denom=32, eps=1e-3):
    """Original SR‑GB (d=1, no CSNP activated) with limit_denominator (for comparison)."""
    _, monomials, evaluate = build_monomial_library(var_names, degree, scale=False)
    Phi, _, _ = evaluate(data)
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
    c = Vt[-1, :]
    c[np.abs(c) < eps] = 0.0
    c_rounded = [Rational(val).limit_denominator(max_denom) for val in c]
    poly_expr = sum(coef * mon for coef, mon in zip(c_rounded, monomials) if coef != 0)
    if poly_expr == 0:
        return []
    poly_expr = poly_expr.expand()
    from sympy import Poly, groebner, symbols, simplify
    sym_vars = symbols(var_names)
    terms = poly_expr.as_ordered_terms()
    if terms:
        max_deg = -1
        lead_term = None
        for term in terms:
            if term == 0:
                continue
            tdeg = 0 if term.is_Number else sum(term.as_powers_dict().values())
            if tdeg > max_deg:
                max_deg = tdeg
                lead_term = term
        if lead_term:
            lead_coef = lead_term.as_coeff_Mul()[0] if not lead_term.is_Number else lead_term
            if lead_coef != 0:
                poly_expr = simplify(poly_expr / lead_coef)
    gb = groebner([Poly(poly_expr, *sym_vars)], *sym_vars, order='grevlex')
    return list(gb)

def run_ablation(sigmas, N=5000, seeds=30):
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    results = []
    for sigma in sigmas:
        for seed in range(seeds):
            data = generate_circle(N, sigma, seed)
            # New SR‑GB with snap_round
            start = time.time()
            try:
                gb_new = sr_gb(data, var_names, degree=2, sigma_estimate=sigma)
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: sr_gb error: {e}")
                gb_new = []
            new_time = time.time() - start
            new_exact = exact_recovery(gb_new, true_inv)
            # Original SR‑GB
            start = time.time()
            try:
                gb_orig = original_sr_gb(data, var_names, degree=2, max_denom=32)
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: original_sr_gb error: {e}")
                gb_orig = []
            orig_time = time.time() - start
            orig_exact = exact_recovery(gb_orig, true_inv)
            results.append({
                "sigma": sigma,
                "seed": seed,
                "new_exact": new_exact,
                "orig_exact": orig_exact,
                "new_time": new_time,
                "orig_time": orig_time
            })
            # Progress print every 10 seeds
            if (seed + 1) % 10 == 0:
                print(f"σ={sigma:.3f}, seed={seed+1}/{seeds}: new={new_exact}, orig={orig_exact}")
    df = pd.DataFrame(results)
    rates = df.groupby("sigma").agg(
        new_rate=("new_exact", "mean"),
        orig_rate=("orig_exact", "mean"),
        new_time=("new_time", "mean"),
        orig_time=("orig_time", "mean")
    ).reset_index()
    # Add confidence intervals
    for idx, row in rates.iterrows():
        sigma_val = row['sigma']
        new_k = df[df['sigma'] == sigma_val]['new_exact'].sum()
        orig_k = df[df['sigma'] == sigma_val]['orig_exact'].sum()
        new_ci = wilson_interval(new_k, seeds)
        orig_ci = wilson_interval(orig_k, seeds)
        rates.loc[idx, 'new_ci_low'] = new_ci[0]
        rates.loc[idx, 'new_ci_high'] = new_ci[1]
        rates.loc[idx, 'orig_ci_low'] = orig_ci[0]
        rates.loc[idx, 'orig_ci_high'] = orig_ci[1]
    return rates, df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same full sigma sweep/N as full run")
    args = parser.parse_args()

    # Sigmas from 0.00 to 0.40 in steps of 0.02 -- same grid in quick mode,
    # only the seed count shrinks.
    sigmas = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40]
    if args.quick:
        print("Running noise ablation (quick: 2 seeds, N=5000, full 21 sigma levels)...")
        rates, full_df = run_ablation(sigmas, N=5000, seeds=2)
    else:
        print("Running noise ablation with 30 seeds per sigma (circle, N=5000)...")
        rates, full_df = run_ablation(sigmas, N=5000, seeds=30)
    print("\n=== Exact recovery rates (30 seeds) ===")
    print(rates.to_string())
    rates.to_csv("Results/noise_ablation_circle_rates.csv", index=False)
    full_df.to_csv("Results/noise_ablation_circle_full.csv", index=False)