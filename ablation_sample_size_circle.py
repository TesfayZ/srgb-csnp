#!/usr/bin/env python3
"""
Ablation: Sample size N vs exact recovery rate for circle invariant.
Noise σ = 0.01, degree=2, dense SVD.
"""

import numpy as np
import pandas as pd
from sympy import parse_expr
from sr_gb import sr_gb, exact_recovery
from utils_stats import wilson_interval
import matplotlib.pyplot as plt
import time

def generate_circle(N, sigma=0.01, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta)
    y = np.sin(theta)
    if sigma > 0:
        x += np.random.normal(0, sigma, N)
        y += np.random.normal(0, sigma, N)
    return np.column_stack([x, y])

def run_ablation(N_list, sigma=0.01, seeds_per_N=30):
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    results = {}
    rows = []
    for N in N_list:
        exact_count = 0
        for seed in range(seeds_per_N):
            data = generate_circle(N, sigma, seed)
            try:
                gb = sr_gb(data, var_names, degree=2, max_denom=32, eps=1e-3, sigma_estimate=sigma)
            except Exception as e:
                print(f"  N={N} seed={seed}: sr_gb error: {e}")
                gb = []
            exact = exact_recovery(gb, true_inv)
            if exact:
                exact_count += 1
        rate = exact_count / seeds_per_N
        ci = wilson_interval(exact_count, seeds_per_N)
        results[N] = rate
        rows.append({"N": N, "sigma": sigma, "n_seeds": seeds_per_N,
                     "n_exact": exact_count, "rate": rate,
                     "ci_low": ci[0], "ci_high": ci[1]})
        print(f"N={N:5d}, exact recovery rate: {rate:.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
    # CSV alongside the PNG so the paper's N>=100 claim has a data artifact,
    # not only the plot.
    pd.DataFrame(rows).to_csv("Results/sample_size_ablation.csv", index=False)
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same full N sweep as full run")
    args = parser.parse_args()

    N_list = [100, 500, 1000, 2000, 5000, 10000]
    if args.quick:
        print("Sample size ablation for circle (quick: 3 seeds, σ=0.01)")
        results = run_ablation(N_list, sigma=0.01, seeds_per_N=3)
    else:
        print("Sample size ablation for circle (σ=0.01)")
        results = run_ablation(N_list, sigma=0.01, seeds_per_N=30)

    # Plot
    plt.figure()
    plt.plot(list(results.keys()), list(results.values()), 'o-')
    plt.xscale('log')
    plt.xlabel('Number of samples N')
    plt.ylabel('Exact recovery rate')
    plt.title('Circle invariant (x²+y²=1), σ=0.01')
    plt.grid(True)
    plt.savefig('Results/sample_size_ablation.png', dpi=150)
    print("\nPlot saved as sample_size_ablation.png")