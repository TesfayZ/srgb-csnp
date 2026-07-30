#!/usr/bin/env python3
"""
benchmark_linear_holonomic.py – Linear holonomic equality constraint q1 - q2 = 0.
SINDy-AD is omitted because this is a static constraint (no time dimension).
"""

import numpy as np
from sympy import parse_expr, groebner, symbols, Poly, cancel
import pandas as pd
from sr_gb import sr_gb, build_monomial_library
from sindy_baselines import sindy_nullspace, sindy_st_ensemble
from utils_stats import wilson_interval
import warnings
warnings.filterwarnings("ignore")


def generate_synthetic_constraint_data(n_samples=5000, sigma=0.0, seed=42):
    np.random.seed(seed)
    q1 = np.random.uniform(-np.pi/2, np.pi/2, n_samples)
    q2 = q1.copy()
    if sigma > 0:
        q1 += np.random.normal(0, sigma, n_samples)
        q2 += np.random.normal(0, sigma, n_samples)
    return np.column_stack([q1, q2])


def exact_recovery(gb_result, true_poly_expr):
    if not gb_result:
        return False
    for p in gb_result:
        p_expr = p.as_expr() if hasattr(p, 'as_expr') else p
        ratio = cancel(p_expr / true_poly_expr)
        if ratio.is_number or ratio.is_constant():
            return True
    return False


def benchmark_synthetic(noise_levels=[0.0, 0.01, 0.02, 0.05],
                         n_samples=5000, n_seeds=30,
                         output_csv="Results/linear_holonomic_equality_results.csv"):
    var_names = ["q1", "q2"]
    true_invariant = parse_expr("q1 - q2")
    results = []

    print("Running synthetic linear holonomic equality constraint benchmark...")
    for sigma in noise_levels:
        for seed in range(n_seeds):
            data = generate_synthetic_constraint_data(n_samples, sigma=sigma, seed=seed)

            # SR-GB+CSNP – adaptive (degree=None, D_max=1)
            try:
                gb_srgb = sr_gb(data, var_names, degree=None, D_max=1, sigma_estimate=sigma)
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: sr_gb error: {e}")
                gb_srgb = []
            exact_srgb = exact_recovery(gb_srgb, true_invariant)
            red_srgb = len(gb_srgb) if gb_srgb else 0

            # SINDy-null
            try:
                null_cands = sindy_nullspace(data, var_names, 1, sigma_estimate=sigma)
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: sindy_nullspace error: {e}")
                null_cands = []
            exact_null = False
            if null_cands:
                for p in null_cands:
                    ratio = cancel(p / true_invariant)
                    if ratio.is_number or ratio.is_constant():
                        exact_null = True
                        break
            red_null = len(null_cands)

            # SINDy-ST (ensemble)
            try:
                st_cands = sindy_st_ensemble(data, var_names, 1, sigma_estimate=sigma)
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: sindy_st_ensemble error: {e}")
                st_cands = []
            exact_st = False
            if st_cands:
                for p in st_cands:
                    ratio = cancel(p / true_invariant)
                    if ratio.is_number or ratio.is_constant():
                        exact_st = True
                        break
            red_st = len(st_cands)

            results.append({
                "sigma": sigma, "seed": seed,
                "srgb_exact": exact_srgb, "srgb_red": red_srgb,
                "null_exact": exact_null, "null_red": red_null,
                "st_exact": exact_st, "st_red": red_st,
            })
            print(f"σ={sigma:.3f} seed={seed}: SRGB={exact_srgb} | null={exact_null} | st={exact_st}")

    df = pd.DataFrame(results)
    summary = df.groupby("sigma").agg(
        srgb_rate=("srgb_exact", "mean"), srgb_red=("srgb_red", "mean"),
        null_rate=("null_exact", "mean"), null_red=("null_red", "mean"),
        st_rate=("st_exact", "mean"), st_red=("st_red", "mean"),
    ).reset_index()

    for idx, row in summary.iterrows():
        sigma_val = row['sigma']
        k_srgb = df[df['sigma'] == sigma_val]['srgb_exact'].sum()
        k_null = df[df['sigma'] == sigma_val]['null_exact'].sum()
        k_st = df[df['sigma'] == sigma_val]['st_exact'].sum()
        summary.loc[idx, 'srgb_ci_low'] = wilson_interval(k_srgb, n_seeds)[0]
        summary.loc[idx, 'srgb_ci_high'] = wilson_interval(k_srgb, n_seeds)[1]
        summary.loc[idx, 'null_ci_low'] = wilson_interval(k_null, n_seeds)[0]
        summary.loc[idx, 'null_ci_high'] = wilson_interval(k_null, n_seeds)[1]
        summary.loc[idx, 'st_ci_low'] = wilson_interval(k_st, n_seeds)[0]
        summary.loc[idx, 'st_ci_high'] = wilson_interval(k_st, n_seeds)[1]

    print("\n" + "=" * 60)
    print("Linear holonomic constraint (SINDy-AD is N/A for static data)")
    print(summary.round(3))
    print("=" * 60)
    df.to_csv(output_csv, index=False)
    summary.to_csv(output_csv.replace(".csv", "_summary.csv"), index=False)
    return summary, df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same noise levels/N as full run")
    args = parser.parse_args()
    if args.quick:
        benchmark_synthetic(n_seeds=2)
    else:
        benchmark_synthetic()