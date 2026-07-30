#!/usr/bin/env python3
"""
Benchmark: over‑lifting the unit circle to degree 3.
Shows that SINDy‑null fails often, while Dense SVD+GB succeeds 100% thanks to algebraic minimality.
Saves results to CSV.
"""

import numpy as np
import pandas as pd
import os
from sr_gb import exact_recovery
from sindy_baselines import sindy_nullspace
from benchmark_redundancy_elimination import dense_svd_gb
from sympy import parse_expr, cancel
from utils_stats import wilson_interval

def generate_circle(N=5000, sigma=0.0, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta)
    y = np.sin(theta)
    if sigma > 0:
        x += np.random.normal(0, sigma, N)
        y += np.random.normal(0, sigma, N)
    return np.column_stack([x, y])

def main(seeds=30, N=5000, sigma=0.0):
    var_names = ["x", "y"]
    true_expr = parse_expr("x**2 + y**2 - 1")
    results = []

    print(f"Running over‑lifting benchmark (circle, D=3, N={N}, σ={sigma}, seeds={seeds})")
    for seed in range(seeds):
        data = generate_circle(N, sigma, seed)
        # SINDy‑null with D=3. Score ANY returned candidate against the
        # truth, the same any-of criterion every other benchmark applies to
        # this baseline (scoring only cands[0] would be a stricter rule than
        # the rest of the suite uses).
        try:
            cands = sindy_nullspace(data, var_names, degree=3, sigma_estimate=sigma)
        except Exception as e:
            print(f"  seed={seed}: sindy_nullspace error: {e}")
            cands = []
        sindy_exact = False
        for poly in cands:
            try:
                ratio = cancel(poly / true_expr)
                if ratio.is_number or ratio.is_constant():
                    sindy_exact = True
                    break
            except:
                pass
        
        # Dense SVD+GB: dominant SVD null vector, snap-round, factor out
        # redundant components, Groebner basis -- the actually-naive
        # ablation (no CSNP combinatorial search over supports), matching
        # the same function used for the non-over-lifted D=2 case in
        # benchmark_redundancy_elimination.py. Calling the full sr_gb()
        # pipeline under the "Dense SVD+GB" name would substitute CSNP's
        # combinatorial search for what this ablation is meant to isolate:
        # factorization alone.
        try:
            gb, _ = dense_svd_gb(data, var_names, degree=3, sigma_estimate=sigma)
        except Exception as e:
            print(f"  seed={seed}: dense_svd_gb error: {e}")
            gb = []
        dense_exact = exact_recovery(gb, true_expr)

        results.append({
            "seed": seed,
            "sindy_exact": sindy_exact,
            "dense_exact": dense_exact
        })
        print(f"  seed={seed:2d}: sindy={sindy_exact}, dense={dense_exact}")

    # Save full results
    df = pd.DataFrame(results)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/overlifting_circle_results.csv", index=False)

    # Summary
    sindy_k = df["sindy_exact"].sum()
    dense_k = df["dense_exact"].sum()
    sindy_rate = sindy_k / seeds
    dense_rate = dense_k / seeds
    ci_sindy = wilson_interval(sindy_k, seeds)
    ci_dense = wilson_interval(dense_k, seeds)

    summary_df = pd.DataFrame([{
        "degree": 3,
        "sigma": sigma,
        "sindy_rate": sindy_rate,
        "sindy_ci_low": ci_sindy[0],
        "sindy_ci_high": ci_sindy[1],
        "dense_rate": dense_rate,
        "dense_ci_low": ci_dense[0],
        "dense_ci_high": ci_dense[1]
    }])
    summary_df.to_csv("Results/overlifting_circle_summary.csv", index=False)

    print(f"\nSINDy‑null (D=3): {sindy_rate:.0%} 95% CI [{ci_sindy[0]:.0%}, {ci_sindy[1]:.0%}]")
    print(f"Dense SVD+GB (D=3): {dense_rate:.0%} 95% CI [{ci_dense[0]:.0%}, {ci_dense[1]:.0%}]")
    print("Results saved to Results/overlifting_circle_*.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N as full run")
    args = parser.parse_args()
    if args.quick:
        main(seeds=2)
    else:
        main()