#!/usr/bin/env python3
"""
Correlated-noise (AR(1)) ablation on the circle invariant.

Every noise model elsewhere in this repo is iid Gaussian, which a referee
can fairly call out: real measurement noise is often serially correlated
(sensor drift, low-pass filtering, frame-to-frame tracking error). This
ablation keeps the circle benchmark identical to
ablation_noise_circle_snap_vs_original.py in every respect except the noise
process: points are sampled as a sweep around the circle (theta sorted, so
"consecutive sample" has the physical meaning of acquisition order along
the trajectory) and the additive noise in each coordinate follows a
stationary AR(1) process over that acquisition order,

    e_i = phi * e_{i-1} + sqrt(1 - phi^2) * sigma * w_i,   w_i ~ N(0,1),

whose MARGINAL standard deviation is exactly sigma for every phi, so
rows with the same sigma are directly comparable across phi and phi = 0
reproduces the iid baseline. sigma_estimate is passed the true marginal
sigma: the method is told the noise SCALE, never its correlation
structure, which is precisely the robustness being probed. Correlation
shrinks the effective sample size by (1-phi)/(1+phi) (~19x fewer
effective samples at phi = 0.9), degrading the SVD's noise averaging
without changing any pointwise statistic.

Saves: Results/ablation_noise_ar1_circle_results.csv (per-trial)
       Results/ablation_noise_ar1_circle_rates.csv   (rates + Wilson CIs)
"""

import time
import numpy as np
import pandas as pd
from sympy import parse_expr

from sr_gb import sr_gb, exact_recovery
from utils_stats import wilson_interval


def ar1_noise(N, sigma, phi, rng):
    """Stationary AR(1) sequence with marginal std exactly sigma."""
    w = rng.normal(0.0, 1.0, N)
    if phi == 0.0:
        return sigma * w
    e = np.empty(N)
    e[0] = w[0]                      # stationary start: e_0 ~ N(0, 1)
    innov = np.sqrt(1.0 - phi ** 2)
    for i in range(1, N):
        e[i] = phi * e[i - 1] + innov * w[i]
    return sigma * e


def generate_circle_sweep_ar1(N, sigma, phi, seed):
    """Circle sweep (sorted theta) with AR(1) noise in acquisition order."""
    rng = np.random.RandomState(seed)
    theta = np.sort(rng.uniform(0, 2 * np.pi, N))
    x = np.cos(theta)
    y = np.sin(theta)
    if sigma > 0:
        x = x + ar1_noise(N, sigma, phi, rng)
        y = y + ar1_noise(N, sigma, phi, rng)
    return np.column_stack([x, y])


def run_ablation(sigmas, phis, N=5000, seeds=30):
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    rows = []
    for phi in phis:
        for sigma in sigmas:
            for seed in range(seeds):
                data = generate_circle_sweep_ar1(N, sigma, phi, seed)
                t0 = time.time()
                try:
                    gb = sr_gb(data, var_names, degree=2, sigma_estimate=sigma)
                except Exception as e:
                    print(f"  phi={phi} sigma={sigma} seed={seed}: sr_gb error: {e}")
                    gb = []
                elapsed = time.time() - t0
                exact = exact_recovery(gb, true_inv)
                rows.append({"phi": phi, "sigma": sigma, "seed": seed,
                             "exact": exact, "n_generators": len(gb),
                             "time": elapsed})
            done = sum(r["exact"] for r in rows
                       if r["phi"] == phi and r["sigma"] == sigma)
            print(f"phi={phi:.1f} sigma={sigma:.2f}: {done}/{seeds} exact")
    df = pd.DataFrame(rows)

    rates = df.groupby(["phi", "sigma"]).agg(
        exact_rate=("exact", "mean"),
        mean_generators=("n_generators", "mean"),
        mean_time=("time", "mean"),
    ).reset_index()
    for idx, row in rates.iterrows():
        sub = df[(df["phi"] == row["phi"]) & (df["sigma"] == row["sigma"])]
        lo, hi = wilson_interval(sub["exact"].sum(), len(sub))
        rates.loc[idx, "ci_low"] = lo
        rates.loc[idx, "ci_high"] = hi
    return df, rates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only; same phi/sigma grid and N")
    args = parser.parse_args()

    # phi = 0 is the iid control (must reproduce the published circle rates);
    # 0.5 is moderate correlation, 0.9 is strong (effective N ~ 260 of 5000).
    phis = [0.0, 0.5, 0.9]
    sigmas = [0.02, 0.05, 0.10]
    seeds = 3 if args.quick else 30

    print(f"AR(1) correlated-noise circle ablation "
          f"(N=5000, {seeds} seeds, phi in {phis}, sigma in {sigmas})")
    df, rates = run_ablation(sigmas, phis, N=5000, seeds=seeds)

    print("\n=== Exact recovery rates ===")
    print(rates.to_string(index=False))
    df.to_csv("Results/ablation_noise_ar1_circle_results.csv", index=False)
    rates.to_csv("Results/ablation_noise_ar1_circle_rates.csv", index=False)
    print("\nSaved to Results/ablation_noise_ar1_circle_*.csv")
