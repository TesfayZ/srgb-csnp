"""
benchmark_smt_verification.py – Demonstrate SMT-based verification of discovered invariants.
Uses Z3 if available; otherwise skips.
"""

import numpy as np
import pandas as pd
from sympy import parse_expr
from sr_gb import sr_gb, exact_recovery
from verification import verify_invariant_smt
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

def benchmark_smt(seeds=30, sigma=0.0):
    var_names = ["x", "y"]
    true_inv = parse_expr("x**2 + y**2 - 1")
    results = []
    for seed in range(seeds):
        data = generate_circle(N=5000, sigma=sigma, seed=seed)
        try:
            gb = sr_gb(data, var_names, degree=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  seed {seed}: sr_gb error: {e}")
            gb = []
        exact = exact_recovery(gb, true_inv)
        # SMT verification
        if gb:
            poly = gb[0].as_expr() if hasattr(gb[0], 'as_expr') else gb[0]
            smt_ok, counterexample, z3_available = verify_invariant_smt(poly, true_inv, var_names)
        else:
            smt_ok, z3_available = False, True
        # smt_ok is None when the check is inconclusive (Z3 answered "unknown"
        # for both sign checks, the sympy-to-z3 conversion raised, or Z3 isn't
        # installed): count it as not-verified for the rate, but record it
        # separately, and record z3_available so a reader without Z3 installed
        # can tell a clean 100% apart from one where SMT never actually ran.
        results.append({"seed": seed, "exact": exact,
                        "smt_verified": smt_ok is True,
                        "smt_inconclusive": smt_ok is None,
                        "z3_available": z3_available})
        print(f"Seed {seed}: exact={exact}, smt={smt_ok}, z3_available={z3_available}")

    df = pd.DataFrame(results)
    exact_rate = df["exact"].mean()
    smt_rate = df["smt_verified"].mean()
    k_exact = df["exact"].sum()
    k_smt = df["smt_verified"].sum()
    ci_exact = wilson_interval(k_exact, seeds)
    ci_smt = wilson_interval(k_smt, seeds)
    print(f"\nExact recovery: {exact_rate:.0%} 95% CI [{ci_exact[0]:.0%}, {ci_exact[1]:.0%}]")
    print(f"SMT verified:  {smt_rate:.0%} 95% CI [{ci_smt[0]:.0%}, {ci_smt[1]:.0%}]")
    df.to_csv("Results/benchmark_smt_verification.csv", index=False)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fewer seeds for fast end-to-end verification")
    args = parser.parse_args()
    benchmark_smt(seeds=3 if args.quick else 30)