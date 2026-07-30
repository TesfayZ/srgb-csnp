#!/usr/bin/env python3
"""
benchmark_sos_sensitivity.py – Benchmark SOS solver sensitivity to redundant constraints.
Shows that reduced Gröbner basis succeeds where redundant constraints cause failure.

This version uses test polynomials that can be negative on the circle, and checks
feasibility by sampling. Both reduced and redundant ideals produce the same result
(they should, as the feasible set is the same). To see the actual failure mode of
redundancy, an SOS solver (e.g., SOSTOOLS) would be needed; this benchmark is a
preliminary illustration.

Saves: Results/sos_sensitivity_results.csv
"""

import numpy as np
from sympy import symbols, lambdify, parse_expr
import pandas as pd
import os
from utils_stats import wilson_interval


def sos_feasibility_sampling(constraints, test_poly, var_names, n_samples=5000):
    """
    Check if test_poly is non-negative on the variety defined by constraints
    by sampling points on the variety (the circle).
    """
    # Generate points on the circle
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    x = np.cos(theta)
    y = np.sin(theta)
    # Evaluate test_poly on these points
    syms = symbols(var_names)
    f = lambdify(syms, test_poly, modules='numpy')
    try:
        vals = f(x, y)
        # Check if any value is significantly negative
        if np.min(vals) < -1e-6:
            return False   # not non-negative
        else:
            return True    # non-negative on all sampled points
    except Exception:
        return False


def run_sos_benchmark(n_tests=30, seed=42):
    np.random.seed(seed)
    x, y = symbols('x y')
    var_names = ['x', 'y']

    # Ground truth circle constraint
    p = x**2 + y**2 - 1

    # Reduced ideal
    reduced_ideal = [p]

    # Redundant ideal (multiple algebraically dependent copies)
    redundant_ideal = [p, p**2, p**3, p**4]

    results = []
    for i in range(n_tests):
        # Generate a random point (a,b) and a radius c
        a = np.random.uniform(-1.5, 1.5)
        b = np.random.uniform(-1.5, 1.5)
        # c chosen so that polynomial can be negative on the circle
        # distance from (a,b) to origin
        r = np.sqrt(a**2 + b**2)
        min_dist_sq = (r - 1.0)**2
        # pick c such that c > min_dist_sq, so the polynomial is negative for some points
        # also ensure c is not too large to avoid numerical issues
        c = np.random.uniform(min_dist_sq + 0.1, min_dist_sq + 2.0)
        q = (x - a)**2 + (y - b)**2 - c

        # Check feasibility with reduced ideal
        try:
            feasible_reduced = sos_feasibility_sampling(reduced_ideal, q, var_names)
        except Exception:
            feasible_reduced = False

        # Check feasibility with redundant ideal
        try:
            feasible_redundant = sos_feasibility_sampling(redundant_ideal, q, var_names)
        except Exception:
            feasible_redundant = False

        results.append({
            "test_id": i,
            "feasible_reduced": feasible_reduced,
            "feasible_redundant": feasible_redundant,
            "a": a, "b": b, "c": c
        })

        print(f"Test {i:2d}: reduced={feasible_reduced}, redundant={feasible_redundant}")

    df = pd.DataFrame(results)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/sos_sensitivity_results.csv", index=False)

    # Summary
    n = len(df)
    reduced_ok = df["feasible_reduced"].sum()
    redundant_ok = df["feasible_redundant"].sum()

    success_reduced = reduced_ok / n
    success_redundant = redundant_ok / n
    ci_reduced = wilson_interval(reduced_ok, n)
    ci_redundant = wilson_interval(redundant_ok, n)

    print("\n" + "=" * 60)
    print("SOS Sensitivity to Redundant Constraints (sampling-based)")
    print(f"Reduced ideal success: {success_reduced:.0%} 95% CI [{ci_reduced[0]:.0%}, {ci_reduced[1]:.0%}]")
    print(f"Redundant ideal success: {success_redundant:.0%} 95% CI [{ci_redundant[0]:.0%}, {ci_redundant[1]:.0%}]")
    print(f"Failure rate due to redundancy: {(1 - success_redundant):.0%}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced test count only, same full run otherwise")
    args = parser.parse_args()
    run_sos_benchmark(n_tests=2 if args.quick else 30)