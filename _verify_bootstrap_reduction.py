#!/usr/bin/env python3
"""
Diagnostic (not a paper artifact): verifies the effect of reducing KRONIC's
ensemble from the library default n_bootstrap=15 to n_bootstrap=5, confirming
whether the smaller ensemble changes the outcome. Runs all 26 Feynman
equations at both settings, sigma in {0.02, 0.05}, 5 seeds.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sympy import parse_expr, Symbol, Poly, cancel
from data_generator import generate_variety_data
from feynman_polynomials import feynman_polynomials
from sindy_baselines import sindy_nullspace

N = 5000
N_SEEDS = 5
SIGMAS = [0.02, 0.05]


def get_degree(expr_str, var_names):
    syms = {v: Symbol(v) for v in var_names}
    try:
        return max(2, Poly(parse_expr(expr_str, local_dict=syms),
                            *[syms[v] for v in var_names]).total_degree())
    except Exception:
        return 2


def check_exact(cands, truth):
    if not cands:
        return False
    for p in cands:
        try:
            ratio = cancel(p / truth)
            if ratio.is_number or ratio.is_constant():
                return True
        except Exception:
            pass
    return False


rows = []
t_reduced_total = 0.0
t_full_total = 0.0
import time
import os

os.makedirs("Results", exist_ok=True)
OUT_CSV = "Results/bootstrap_reduction_verification.csv"

for fid, expr_str, var_names, ranges in feynman_polynomials:
    deg = get_degree(expr_str, var_names)
    syms = {v: Symbol(v) for v in var_names}
    truth = parse_expr(expr_str, local_dict=syms)
    for sigma in SIGMAS:
        for seed in range(N_SEEDS):
            X = generate_variety_data(expr_str, var_names, ranges, N=N, sigma=sigma, seed=seed)

            t0 = time.time()
            null_reduced = sindy_nullspace(X, var_names, deg, sigma_estimate=sigma, n_bootstrap=5, bootstrap_frac=0.8)
            t_reduced = time.time() - t0

            t0 = time.time()
            null_full = sindy_nullspace(X, var_names, deg, sigma_estimate=sigma, n_bootstrap=15, bootstrap_frac=0.8)
            t_full = time.time() - t0

            t_reduced_total += t_reduced
            t_full_total += t_full

            null_reduced_exact = check_exact(null_reduced, truth)
            null_full_exact = check_exact(null_full, truth)

            rows.append({
                "id": fid, "sigma": sigma, "seed": seed,
                "null_reduced_exact": null_reduced_exact, "null_full_exact": null_full_exact,
                "null_changed": null_reduced_exact != null_full_exact,
            })
            if null_reduced_exact != null_full_exact:
                print(f"CHANGED: {fid} sigma={sigma} seed={seed} "
                      f"null: {null_reduced_exact}->{null_full_exact}", flush=True)

    # Flush after every equation so a wall-clock timeout (this diagnostic is
    # the slowest probe in the suite) preserves all completed equations rather
    # than discarding the whole run and producing no CSV at all.
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"done {fid} (running totals: reduced={t_reduced_total:.0f}s full={t_full_total:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"Total trials: {len(df)}")
print(f"SINDy-null: reduced rate={df['null_reduced_exact'].mean():.1%}  full rate={df['null_full_exact'].mean():.1%}  "
      f"changed in {df['null_changed'].sum()}/{len(df)} trials")
print(f"\nWall time: reduced={t_reduced_total:.0f}s full={t_full_total:.0f}s (ratio {t_full_total/max(t_reduced_total,1):.2f}x)")
print("Saved to Results/bootstrap_reduction_verification.csv")
