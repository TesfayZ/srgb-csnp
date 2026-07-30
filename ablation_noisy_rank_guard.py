#!/usr/bin/env python3
"""
ablation_noisy_rank_guard.py - Sensitivity of the noisy-rank dominance-pruning
guard (sr_gb._conservative_noisy_rank, sr_gb._NOISY_RANK_GUARD_MULTIPLIER) to
its multiplier k in "rel_smallest <= k*sigma_estimate".

A sensitivity ablation, not a search for the best-scoring k: per
Adaptive-CSNP.tex Table tab:thresholds, hand-set tolerances in this pipeline
are characterized across a range rather than tuned to whichever value
scores best, following the same procedure as ablation_qmax_tolerance.py.

Two arms, both on the noisy 2D-oscillator deflation system the guard was
written for (see benchmark_oracle_misclassification.py):

  A. Rescue rate: does the guard still stop a true generator's support from
     being wrongly dominance-pruned as HARD_INFEASIBLE, across a sigma
     sweep? Reuses benchmark_oracle_misclassification.py's harness
     (systems, TRUE_GENERATORS, estimate_rank_robust), varying only k and
     sigma.

  B. Over-admission cost: does raising k let the full end-to-end pipeline
     (sr_gb() with full_nullspace=True) invent a spurious extra invariant
     out of noise, or lose a genuine one? Two systems reused from
     validate_sr_gb.py:
       B1. test_single_invariant_no_spurious_growth's system: a single
           invariant in an over-lifted degree-2 library. Pass condition
           is exact recovery AND exactly 1 generator.
       B2. test_entangled_multi_invariant_with_noise's system: two
           independent invariants x0-x1, x2-x3 in the same over-lifted
           library. Pass condition is both recovered AND exactly 2
           generators.

Both arms require d>=2 (_conservative_noisy_rank is a no-op below that): the
oracle diagnostic's 2D oscillator has d=4, and B1/B2's over-lifted degree-2
library has d* in the low teens, so both exercise the guard.

k=0.0 is included as a reference point reproducing pre-guard behavior: the
guard can never fire there (rel_smallest <= 0*sigma is false for any
nonzero singular value).

Saves:
  - Results/ablation_noisy_rank_guard_rescue.csv        (Arm A, per k x sigma x seed x generator)
  - Results/ablation_noisy_rank_guard_rescue_summary.csv
  - Results/ablation_noisy_rank_guard_admission.csv     (Arm B, per k x sigma x seed x sub-system)
  - Results/ablation_noisy_rank_guard_admission_summary.csv

Accepts --quick (2 seeds, same grids).
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.linalg import svd

import sr_gb
from sr_gb import evaluate_support, sr_gb as run_sr_gb, exact_recovery
from sympy import parse_expr
from benchmark_oracle_misclassification import (
    generate_2d_harmonic_pairs, estimate_rank_robust, support_of,
    TRUE_GENERATORS, evaluate,
)
from utils_stats import wilson_interval

MULTIPLIER_GRID = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
SIGMA_GRID = [0.01, 0.02, 0.05, 0.1]
N = 5000
DT = 0.1

# Arm B sigma grid stops at 0.02, matching the range validate_sr_gb.py's
# own tests establish as known-good for these two systems; pushing further
# would be measuring an unvalidated noise regime, not this guard.
ADMISSION_SIGMA_GRID = [0.01, 0.02]


def run_rescue(seeds):
    """Arm A: fraction of true generators NOT wrongly HARD_INFEASIBLE."""
    rows = []
    for sigma in SIGMA_GRID:
        for seed in range(seeds):
            data = generate_2d_harmonic_pairs(N=N, dt=DT, sigma=sigma, seed=seed)
            Phi_old, _, _ = evaluate(data[:, :4])
            Phi_new, _, _ = evaluate(data[:, 4:])
            Phi_diff = (Phi_old - Phi_new)[:, 1:]
            _, s, Vt = svd(Phi_diff, full_matrices=False)
            r = estimate_rank_robust(s, sigma_estimate=sigma, N=N)
            d = Phi_diff.shape[1] - r
            if d < 1:
                continue
            V_null = Vt[-d:, :].T
            eps = max(1e-4, 3.0 * sigma)
            for k in MULTIPLIER_GRID:
                sr_gb._NOISY_RANK_GUARD_MULTIPLIER = k
                for name, expr in TRUE_GENERATORS.items():
                    result = evaluate_support(
                        support_of(expr), V_null, d, max_denom=16, eps=eps,
                        sigma_estimate=sigma, Phi=Phi_diff, N=Phi_diff.shape[0])
                    rows.append({
                        "k": k, "sigma": sigma, "seed": seed,
                        "generator": name,
                        "hard_infeasible": result["status"] == "HARD_INFEASIBLE",
                    })
        print(f"[A] sigma={sigma} done")
    df = pd.DataFrame(rows)
    summ = (df.groupby(["k", "sigma"])
              .agg(misclassified=("hard_infeasible", "sum"),
                   n=("hard_infeasible", "size"))
              .reset_index())
    summ["rescue_rate"] = 1.0 - summ["misclassified"] / summ["n"]
    ci = summ.apply(lambda r: wilson_interval(
        int(r["n"] - r["misclassified"]), int(r["n"])), axis=1)
    summ["ci_low"] = [c[0] for c in ci]
    summ["ci_high"] = [c[1] for c in ci]
    return df, summ


def _single_invariant_case(sigma, seed):
    """Same construction as validate_sr_gb.py's
    test_single_invariant_no_spurious_growth: one genuine invariant x0-x1
    in an over-lifted degree-2 library over 6 variables."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (4000, 6))
    if sigma > 0:
        X[:, 1] = X[:, 0] + rng.normal(0, sigma, 4000)
    else:
        X[:, 1] = X[:, 0]
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    gb = run_sr_gb(X, var_names, degree=2, sigma_estimate=sigma, full_nullspace=True)
    ok = exact_recovery(gb, parse_expr("x0 - x1")) and len(gb) == 1
    return ok, len(gb)


def _entangled_two_invariant_case(sigma, seed):
    """Same construction as validate_sr_gb.py's
    test_entangled_multi_invariant_with_noise: two independent invariants
    x0-x1, x2-x3 in the same over-lifted degree-2 library, full_nullspace
    mode (the ungated, must-recover-both path)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, (4000, 6))
    X[:, 1] = X[:, 0] + (rng.normal(0, sigma, 4000) if sigma > 0 else 0.0)
    X[:, 3] = X[:, 2] + (rng.normal(0, sigma, 4000) if sigma > 0 else 0.0)
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    gb = run_sr_gb(X, var_names, degree=2, sigma_estimate=sigma, full_nullspace=True)
    polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]
    has_01 = exact_recovery(gb, parse_expr("x0 - x1"))
    has_23 = exact_recovery(gb, parse_expr("x2 - x3"))
    ok = has_01 and has_23 and len(polys) == 2
    return ok, len(polys)


def run_admission(seeds):
    """Arm B: does k break either of two known-good noisy multi-invariant
    cases (clean recovery, no spurious extras)?"""
    rows = []
    for sub_name, case_fn in [("single_invariant", _single_invariant_case),
                              ("entangled_two_invariant", _entangled_two_invariant_case)]:
        for sigma in ADMISSION_SIGMA_GRID:
            for k in MULTIPLIER_GRID:
                sr_gb._NOISY_RANK_GUARD_MULTIPLIER = k
                for seed in range(seeds):
                    ok, n_gen = case_fn(sigma, seed)
                    rows.append({"sub_system": sub_name, "k": k, "sigma": sigma,
                                 "seed": seed, "clean_recovery": ok,
                                 "n_generators": n_gen})
            print(f"[B] {sub_name} sigma={sigma} done")
    df = pd.DataFrame(rows)
    summ = (df.groupby(["sub_system", "k", "sigma"])
              .agg(clean_rate=("clean_recovery", "mean"),
                   k_clean=("clean_recovery", "sum"),
                   n=("clean_recovery", "size"),
                   mean_n_generators=("n_generators", "mean"),
                   max_n_generators=("n_generators", "max"))
              .reset_index())
    ci = summ.apply(lambda r: wilson_interval(int(r["k_clean"]), int(r["n"])), axis=1)
    summ["ci_low"] = [c[0] for c in ci]
    summ["ci_high"] = [c[1] for c in ci]
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--quick", action="store_true", help="2 seeds only; same grids")
    args = ap.parse_args()
    seeds = 2 if args.quick else args.seeds

    default_multiplier = sr_gb._NOISY_RANK_GUARD_MULTIPLIER
    print("=" * 72)
    print("ABLATION: noisy-rank dominance-pruning guard multiplier sensitivity")
    print(f"seeds={seeds}  multiplier grid={MULTIPLIER_GRID}  sigma grid={SIGMA_GRID}")
    print(f"shipped default: {default_multiplier}")
    print("=" * 72)

    os.makedirs("Results", exist_ok=True)
    try:
        print("\n-- Arm A: rescue rate (true generator NOT Hard-pruned) vs k --")
        a_full, a_summ = run_rescue(seeds)
        a_full.to_csv("Results/ablation_noisy_rank_guard_rescue.csv", index=False)
        a_summ.to_csv("Results/ablation_noisy_rank_guard_rescue_summary.csv", index=False)

        print("\n-- Arm B: known-good noisy case, clean-recovery / spurious-generator cost vs k --")
        b_full, b_summ = run_admission(seeds)
        b_full.to_csv("Results/ablation_noisy_rank_guard_admission.csv", index=False)
        b_summ.to_csv("Results/ablation_noisy_rank_guard_admission_summary.csv", index=False)
    finally:
        sr_gb._NOISY_RANK_GUARD_MULTIPLIER = default_multiplier

    print("\n=== Arm A: rescue rate by sigma x k ===")
    piv = a_summ.pivot(index="sigma", columns="k", values="rescue_rate")
    print(piv.to_string(float_format=lambda v: f"{v:.0%}"))
    for sub_name in sorted(b_summ["sub_system"].unique()):
        sub = b_summ[b_summ["sub_system"] == sub_name]
        print(f"\n=== Arm B ({sub_name}): clean-recovery rate by sigma x k ===")
        pivb = sub.pivot(index="sigma", columns="k", values="clean_rate")
        print(pivb.to_string(float_format=lambda v: f"{v:.0%}"))
        print(f"=== Arm B ({sub_name}): mean generator count by sigma x k ===")
        pivc = sub.pivot(index="sigma", columns="k", values="mean_n_generators")
        print(pivc.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nShipped default multiplier is {default_multiplier} (unchanged by this "
          f"ablation; see module docstring).")
    print("Results written to Results/ablation_noisy_rank_guard_*.csv")


if __name__ == "__main__":
    main()
