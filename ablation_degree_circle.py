#!/usr/bin/env python3
"""
Ablation: Adaptive degree discovery + fixed‑degree recovery on the circle.

runs:
  1) Adaptive mode (degree=None, D_max=4) to test MDL selection.
  2) Fixed degrees 1..4 to see exact recovery rates at each degree.

Saves:
  - Results/ablation_adaptive_degree_circle.csv
  - Results/ablation_fixed_degree_results.csv
"""

import numpy as np
import pandas as pd
from sympy import parse_expr, Poly
import argparse
import os

from sr_gb import sr_gb, exact_recovery
from utils_stats import wilson_interval

def generate_circle(N=5000, sigma=0.01, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta)
    y = np.sin(theta)
    if sigma > 0:
        x += np.random.normal(0, sigma, N)
        y += np.random.normal(0, sigma, N)
    return np.column_stack([x, y])

def run_adaptive(seeds=30, N=5000, sigma=0.01, D_max=4):
    """Adaptive degree discovery: exact recovery and degree selection."""
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    results = []
    for seed in range(seeds):
        data = generate_circle(N, sigma, seed)
        try:
            gb = sr_gb(data, var_names, degree=None, D_max=D_max, sigma_estimate=sigma)
        except Exception as e:
            print(f"  seed {seed}: sr_gb error: {e}")
            gb = []
        exact = exact_recovery(gb, true_inv)
        # Determine recovered degree from first generator
        if gb:
            poly = gb[0].as_expr() if hasattr(gb[0], 'as_expr') else gb[0]
            if poly != 0:
                try:
                    p_poly = Poly(poly, *[parse_expr(v) for v in var_names])
                    deg = p_poly.total_degree()
                except:
                    deg = -1
            else:
                deg = -1
        else:
            deg = -1
        results.append({"seed": seed, "exact": exact, "recovered_degree": deg})
        if (seed+1) % 10 == 0:
            print(f"Adaptive: seed {seed+1}/{seeds} done")
    df = pd.DataFrame(results)
    exact_rate = df["exact"].mean()
    k_exact = df["exact"].sum()
    ci_exact = wilson_interval(k_exact, seeds)
    deg_ok = df[df["recovered_degree"] == 2].shape[0]
    deg_rate = deg_ok / seeds
    ci_deg = wilson_interval(deg_ok, seeds)
    print("\n=== Adaptive Degree Discovery ===")
    print(f"Exact recovery rate: {exact_rate:.0%} 95% CI [{ci_exact[0]:.0%}, {ci_exact[1]:.0%}]")
    print(f"Correct degree (2) selection rate: {deg_rate:.0%} 95% CI [{ci_deg[0]:.0%}, {ci_deg[1]:.0%}]")
    return df

def run_fixed_degrees(seeds=30, N=5000, sigma=0.01, degrees=[1,2,3,4]):
    """Fixed‑degree recovery for each D in degrees."""
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    results = []
    for D in degrees:
        for seed in range(seeds):
            data = generate_circle(N, sigma, seed)
            try:
                gb = sr_gb(data, var_names, degree=D, sigma_estimate=sigma)
            except Exception as e:
                print(f"  D={D} seed {seed}: sr_gb error: {e}")
                gb = []
            ok = exact_recovery(gb, true_inv)
            results.append({"degree": D, "seed": seed, "exact": ok})
        # Progress report
        print(f"Fixed D={D}: done")
    df = pd.DataFrame(results)
    summary = df.groupby("degree").agg(
        rate=("exact", "mean"),
        k=("exact", "sum")
    ).reset_index()
    # Add Wilson CIs
    def ci_row(row):
        lo, hi = wilson_interval(row["k"], seeds)
        return pd.Series({"ci_low": lo, "ci_high": hi})
    summary[["ci_low", "ci_high"]] = summary.apply(ci_row, axis=1)
    summary = summary[["degree", "rate", "ci_low", "ci_high"]]
    print("\n=== Fixed‑Degree Recovery ===")
    for _, row in summary.iterrows():
        print(f"D={int(row['degree'])}: {row['rate']:.0%} 95% CI [{row['ci_low']:.0%}, {row['ci_high']:.0%}]")
    return df, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--Dmax", type=int, default=4, help="Maximum degree for adaptive and fixed sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N/sigma/Dmax as full run")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 2

    print("=" * 70)
    print("ABLATION: Degree discovery on the circle")
    print(f"Seeds = {args.seeds}, N = {args.N}, σ = {args.sigma}, Dmax = {args.Dmax}")
    print("=" * 70)

    # 1. Adaptive
    df_adapt = run_adaptive(seeds=args.seeds, N=args.N, sigma=args.sigma, D_max=args.Dmax)

    # 2. Fixed degrees 1..Dmax
    degrees = list(range(1, args.Dmax+1))
    df_fixed, summary_fixed = run_fixed_degrees(seeds=args.seeds, N=args.N,
                                                sigma=args.sigma, degrees=degrees)

    # Save
    os.makedirs("Results", exist_ok=True)
    df_adapt.to_csv("Results/ablation_adaptive_degree_circle.csv", index=False)
    df_fixed.to_csv("Results/ablation_fixed_degree_results_full.csv", index=False)
    summary_fixed.to_csv("Results/ablation_fixed_degree_summary.csv", index=False)

    print("\nResults saved to Results/")

if __name__ == "__main__":
    main()