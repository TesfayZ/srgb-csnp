#!/usr/bin/env python3
"""
benchmark_pysr_gb_implicit_vs_explicit.py – PySR-GB baseline comparing implicit (circle) vs explicit (F=ma).

Runs two experiments and compares them:

  1. Circle  (x² + y² - 1 = 0)  — multi-valued implicit variety
     PySR receives x as input, y as target.
     Expected: FAIL — y=f(x) is multi-valued, PySR sees contradictory training pairs.

  2. F = ma  (F - m*a = 0)      — single-valued linear relation
     PySR receives m, a as inputs, F as target.
     Expected: PASS — F=ma is explicit and unambiguous, PySR's natural task.

The contrast between the two results supports the paper's claim:
  "Naive adaptation of explicit SR tools to implicit invariant discovery
   fails on multi-valued varieties — the class SR-GB + CSNP is designed for."

Usage:
    python benchmark_pysr_gb_implicit_vs_explicit.py                        # both experiments, defaults
    python benchmark_pysr_gb_implicit_vs_explicit.py --iters 40             # more PySR iterations
    python benchmark_pysr_gb_implicit_vs_explicit.py --experiment circle    # circle only
    python benchmark_pysr_gb_implicit_vs_explicit.py --experiment fma       # F=ma only
"""

import numpy as np
import pandas as pd
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sympy import parse_expr, groebner, symbols, Poly, Symbol, cancel, expand
from sr_gb import exact_recovery
from utils_stats import wilson_interval
from data_generator import generate_variety_data

try:
    from pysr import PySRRegressor
except ImportError:
    raise ImportError(
        "PySR is required. Install with:  pip install pysr\n"
        "Also requires Julia: https://julialang.org/downloads/"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core PySR-GB runner
# ─────────────────────────────────────────────────────────────────────────────

def pysr_gb(X_feat, y, feature_names, target_name, true_expr, niterations=20):
    """
    Run PySR for explicit regression y = f(X_feat), then convert to implicit
    polynomial p(features, target) = 0 and compute its reduced Gröbner basis.

    Parameters
    ----------
    X_feat       : np.ndarray (N, n_features)
    y            : np.ndarray (N,)
    feature_names: list[str]   — column names of X_feat
    target_name  : str         — name of the target variable
    true_expr    : sympy expr  — ground-truth implicit polynomial in all variables
    niterations  : int

    Returns
    -------
    gb_list  : list of SymPy expressions (Gröbner basis generators)
    exact    : bool
    best_eq  : SymPy expression returned by PySR
    note     : str  — human-readable diagnosis
    """
    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        parallelism="serial",
        tempdir="pysr_temp",
        delete_tempfiles=True,
        verbosity=0,
    )
    model.fit(X_feat, y, variable_names=feature_names)

    try:
        best_eq = model.sympy()
    except (AttributeError, TypeError):
        try:
            best_eq = model.sympy_expr
        except AttributeError:
            best_eq = model.equations_.iloc[0]["sympy_format"]

    # Build implicit form:  target - best_eq = 0
    all_names = feature_names + [target_name]
    sym_vars  = symbols(" ".join(all_names))
    target_sym = sym_vars[-1]

    try:
        invariant = expand(target_sym - best_eq)
        poly      = Poly(invariant, *sym_vars)
        gb        = groebner([poly], *sym_vars, order="grevlex")
        gb_list   = list(gb)
        exact     = exact_recovery(gb_list, true_expr)
        note      = "ok"
    except Exception as e:
        gb_list = []
        exact   = False
        note    = f"groebner_failed: {str(e)[:60]}"

    return gb_list, exact, best_eq, note


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Circle  (expected: FAIL)
# ─────────────────────────────────────────────────────────────────────────────

def run_circle_baseline(N=5000, n_seeds=30, niterations=20):
    """
    Circle: x² + y² - 1 = 0

    Why PySR-GB is expected to fail:
      The variety is multi-valued — for each x there are two y values (+√(1-x²) and
      -√(1-x²)).  PySR sees contradictory (x, y) pairs and cannot fit a single
      function y = f(x).  Even if it finds a partial fit, converting to implicit
      form and computing a Gröbner basis will not recover x² + y² - 1.
    """
    print("=" * 60)
    print("Experiment 1: Circle  (x² + y² - 1 = 0)")
    print("Expected result: FAIL  (multi-valued implicit variety)")
    print("=" * 60)
    print(f"N={N}, seeds={n_seeds}, niterations={niterations}\n")

    # Generate all data at once, split by seed
    X_all = generate_variety_data(
        "x**2 + y**2 - 1", ["x", "y"],
        {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
        N=N * n_seeds, sigma=0.0, seed=0
    )

    syms      = {v: Symbol(v) for v in ["x", "y"]}
    true_expr = parse_expr("x**2 + y**2 - 1", local_dict=syms)

    results = []
    for seed in range(n_seeds):
        X      = X_all[seed * N : (seed + 1) * N]
        X_feat = X[:, 0:1]   # x → feature
        y_vals = X[:, 1]     # y → target

        gb, exact, eq, note = pysr_gb(
            X_feat, y_vals, ["x"], "y", true_expr, niterations=niterations
        )
        results.append({
            "experiment": "circle",
            "seed": seed, "exact": exact,
            "expression": str(eq), "note": note
        })
        print(f"  seed={seed}: exact={exact}  expression={eq}")

    df   = pd.DataFrame(results)
    rate = df["exact"].mean() * 100
    k = int(df['exact'].sum())
    ci = wilson_interval(k, n_seeds)
    print(f"\nCircle exact recovery: {rate:.0f}% 95% CI [{ci[0]*100:.0f}%, {ci[1]*100:.0f}%]")
    df.to_csv("Results/pysr_gb_circle_results.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: F = ma  (expected: PASS)
# ─────────────────────────────────────────────────────────────────────────────

def run_fma_baseline(N=5000, n_seeds=30, niterations=20):
    """
    Newton's second law: F - m*a = 0

    Why PySR-GB is expected to succeed:
      F = m*a is single-valued and explicit — given m and a there is exactly one F.
      PySR can learn this perfectly.  This is PySR's natural task.

    This experiment serves as a control: it confirms PySR-GB works when the
    implicit variety is single-valued, isolating the failure mode on the circle
    to multi-valuedness rather than any other factor.
    """
    print("=" * 60)
    print("Experiment 2: F = ma  (F - m*a = 0)")
    print("Expected result: PASS  (single-valued, explicit relation)")
    print("=" * 60)
    print(f"N={N}, seeds={n_seeds}, niterations={niterations}\n")

    X_all = generate_variety_data(
        "F - m*a", ["F", "m", "a"],
        {"F": (-10, 10), "m": (0.5, 5), "a": (-5, 5)},
        N=N * n_seeds, sigma=0.0, seed=0
    )

    syms      = {v: Symbol(v) for v in ["F", "m", "a"]}
    true_expr = parse_expr("F - m*a", local_dict=syms)

    results = []
    for seed in range(n_seeds):
        X      = X_all[seed * N : (seed + 1) * N]
        X_feat = X[:, 1:]    # m, a → features
        y_vals = X[:, 0]     # F   → target

        gb, exact, eq, note = pysr_gb(
            X_feat, y_vals, ["m", "a"], "F", true_expr, niterations=niterations
        )
        results.append({
            "experiment": "fma",
            "seed": seed, "exact": exact,
            "expression": str(eq), "note": note
        })
        print(f"  seed={seed}: exact={exact}  expression={eq}")

    df   = pd.DataFrame(results)
    rate = df["exact"].mean() * 100
    k = int(df['exact'].sum())
    ci = wilson_interval(k, n_seeds)
    print(f"\nF=ma exact recovery: {rate:.0f}% 95% CI [{ci[0]*100:.0f}%, {ci[1]*100:.0f}%]")
    df.to_csv("Results/pysr_gb_fma_results.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df_circle, df_fma):
    print()
    print("=" * 60)
    print("COMBINED SUMMARY")
    print("=" * 60)
    combined = pd.concat([df_circle, df_fma], ignore_index=True)
    summary  = (combined.groupby("experiment")["exact"]
                        .agg(exact_rate="mean", n_seeds="count")
                        .reset_index())
    summary["exact_rate"] = (summary["exact_rate"] * 100).round(1).astype(str) + "%"
    summary.columns       = ["Experiment", "Exact recovery", "Seeds"]
    print(summary.to_string(index=False))
    print()
    print("Interpretation:")
    print("  F=ma  succeeds → PySR-GB works for single-valued explicit relations.")
    print("  Circle fails   → PySR-GB cannot handle multi-valued implicit varieties.")
    print("  SR-GB handles both by working directly in the polynomial nullspace,")
    print("  treating all variables symmetrically without designating a target.")
    combined.to_csv("Results/pysr_gb_combined_results.csv", index=False)
    print("\nAll results saved to Results/pysr_gb_*_results.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PySR-GB baseline: compare circle (fail) vs F=ma (pass)"
    )
    parser.add_argument("--N",           type=int, default=5000,
                        help="samples per experiment (default 5000)")
    parser.add_argument("--seeds",       type=int, default=30,
                        help="random seeds per experiment (default 30)")
    parser.add_argument("--iters",       type=int, default=20,
                        help="PySR niterations (default 20, increase for better fit)")
    parser.add_argument("--experiment",  type=str, default="both",
                        choices=["both", "circle", "fma"],
                        help="which experiment to run (default: both)")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N/iters as full run")
    args = parser.parse_args()
    if args.quick:
        args.seeds = 2

    df_circle = df_fma = None

    if args.experiment in ("both", "circle"):
        df_circle = run_circle_baseline(N=args.N, n_seeds=args.seeds,
                                        niterations=args.iters)
        print()

    if args.experiment in ("both", "fma"):
        df_fma = run_fma_baseline(N=args.N, n_seeds=args.seeds,
                                   niterations=args.iters)

    if df_circle is not None and df_fma is not None:
        print_summary(df_circle, df_fma)