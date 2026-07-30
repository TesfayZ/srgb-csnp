#!/usr/bin/env python3
"""
ablation_sindy_snap_gb_hybrid.py runs the "SINDy + snap-rounding + GB"
hybrid that Appendix app:baseline-rationale of Adaptive-CSNP.tex argues
against but never actually executes (TODO.md item 14).

The appendix's argument: passing several redundant, only-approximately-
proportional inexact polynomials into one exact Groebner-basis call is not
a clean canonicalization-only ablation (SINDy-null uses Lasso/thresholded
regression, not SVD nullspace estimation) and is not numerically well-posed
(exact GB over inexact coefficients need not collapse to the true generator).

sindy_nullspace() (sindy_baselines.py) already snap-rounds each candidate
individually and only deduplicates by exact equality, so on an ambiguous
nullspace (d>1) it returns several distinct-but-near-proportional exact
rational polynomials, exactly the input the appendix describes. This
script feeds that whole candidate list into ONE groebner(...) call (rather
than reducing a single dominant vector, as the paper's actual "Dense
SVD+GB" ablation does in benchmark_redundancy_elimination.py's
dense_svd_gb) and reports what happens on the over-lifted circle already
used for exactly this ambiguous-nullspace scenario in
benchmark_overlifting_circle.py.
"""
import os
import numpy as np
import pandas as pd
from sympy import parse_expr, groebner, Poly, symbols

from sr_gb import exact_recovery
from sindy_baselines import sindy_nullspace
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


def sindy_snap_gb_hybrid(data, var_names, degree, sigma_estimate=0.0):
    """The naive hybrid: SINDy-null's candidates (each already snap-rounded
    internally), passed jointly to one exact Groebner-basis call. Returns
    (gb, n_candidates, degenerate, raised_exception) where degenerate means
    the basis collapsed to the unit ideal [1]."""
    cands = sindy_nullspace(data, var_names, degree, sigma_estimate=sigma_estimate)
    n_candidates = len(cands)
    if n_candidates == 0:
        return [], 0, False, False
    sym_vars = symbols(var_names)
    try:
        gb = list(groebner([Poly(p, *sym_vars) for p in cands], *sym_vars, order='grevlex'))
    except Exception:
        return [], n_candidates, False, True
    degenerate = len(gb) == 1 and gb[0].as_expr() == 1
    return gb, n_candidates, degenerate, False


def main(seeds=30, N=5000):
    var_names = ["x", "y"]
    true_expr = parse_expr("x**2 + y**2 - 1")
    degree = 3  # over-lifted: true relation is degree 2, library goes to 3
    sigmas = [0.0, 0.05]
    results = []

    print(f"Running SINDy+snap+GB hybrid ablation (circle, D={degree}, N={N}, seeds={seeds})")
    for sigma in sigmas:
        for seed in range(seeds):
            data = generate_circle(N, sigma, seed)
            try:
                gb, n_cands, degenerate, raised = sindy_snap_gb_hybrid(
                    data, var_names, degree, sigma_estimate=sigma)
                exact = exact_recovery(gb, true_expr) if gb else False
            except Exception as e:
                print(f"  sigma={sigma} seed={seed}: unexpected error: {e}")
                gb, n_cands, degenerate, raised, exact = [], 0, False, True, False

            results.append({
                "sigma": sigma,
                "seed": seed,
                "n_candidates": n_cands,
                "gb_size": len(gb),
                "degenerate": degenerate,
                "raised_exception": raised,
                "exact": exact,
            })
            print(f"  sigma={sigma} seed={seed:2d}: n_candidates={n_cands} "
                  f"gb_size={len(gb)} degenerate={degenerate} "
                  f"exception={raised} exact={exact}")

    df = pd.DataFrame(results)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/sindy_snap_gb_hybrid_results.csv", index=False)

    summary_rows = []
    for sigma in sigmas:
        sub = df[df["sigma"] == sigma]
        k = int(sub["exact"].sum())
        n = len(sub)
        lo, hi = wilson_interval(k, n)
        summary_rows.append({
            "sigma": sigma,
            "n_seeds": n,
            "exact_rate": k / n,
            "ci_low": lo,
            "ci_high": hi,
            "degenerate_rate": sub["degenerate"].mean(),
            "exception_rate": sub["raised_exception"].mean(),
            "mean_n_candidates": sub["n_candidates"].mean(),
        })
        print(f"\nsigma={sigma}: exact {k}/{n} = {k/n:.0%} 95% CI [{lo:.0%}, {hi:.0%}], "
              f"degenerate {sub['degenerate'].mean():.0%}, "
              f"exception {sub['raised_exception'].mean():.0%}, "
              f"mean candidates {sub['n_candidates'].mean():.1f}")

    pd.DataFrame(summary_rows).to_csv("Results/sindy_snap_gb_hybrid_summary.csv", index=False)
    print("\nResults saved to Results/sindy_snap_gb_hybrid_*.csv")


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
