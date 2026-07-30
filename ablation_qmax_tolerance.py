#!/usr/bin/env python3
"""
Ablation: sensitivity to the rational-denominator cap Qmax (max_denom) and to
the snap/rational-window tolerance eps.

This answers a reviewer question that the rest of the suite did not isolate:
how do exact-recovery rate, false-positive rate, and runtime depend on Qmax and
on eps? Three arms:

  A. Qmax recovery / ceiling.
     Systems with rational-coefficient invariants of increasing denominator
     demand, sampled exactly on the variety (sigma=0, so representability is
     the ONLY thing that varies). Each system's coefficient vector is a set of
     small coprime integers; the "min_denom" annotation is the smallest Qmax
     at which the invariant is representable at all UNDER snap_round's OWN
     normalisation rule (divide by whichever coefficient has the largest
     magnitude), not the best denominator achievable over every possible
     normalisation. min_denom is a necessary AND sufficient floor here:
     recovery is 0% at every Qmax below it and 100% at every Qmax at or
     above it, with no observed margin. The two systems whose min_denom
     exceeds 16 document the ceiling of the default Qmax=16 candidly.

  B. Qmax false positive.
     Negative systems with NO low-denominator rational invariant: a generic
     point cloud (no polynomial relation at all) and a variety with an
     irrational coefficient (x^2 + pi*y^2 - 1). The method should abstain. We
     count a false positive whenever a nonzero generator is returned, and track
     whether raising Qmax inflates that rate (the worry that a bigger cap snaps
     noise to a spurious complicated fraction).

  C. eps / tolerance sensitivity.
     Circle recovery under noise and irrational-coefficient abstention as eps
     sweeps, to show the snap tolerance is not finely tuned.

Saves (read by generate_result_tables.py):
  - Results/ablation_qmax_recovery.csv        (per system x Qmax x seed)
  - Results/ablation_qmax_recovery_summary.csv
  - Results/ablation_qmax_falsepos.csv
  - Results/ablation_qmax_falsepos_summary.csv
  - Results/ablation_eps_sensitivity.csv
  - Results/ablation_eps_sensitivity_summary.csv

Every entry point accepts --quick (2 seeds).
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from math import gcd
from functools import reduce
from sympy import parse_expr

from sr_gb import sr_gb, exact_recovery
from data_generator import generate_variety_data
from utils_stats import wilson_interval

VAR_NAMES = ["x", "y"]
QMAX_GRID = [2, 4, 8, 16, 32, 64]
EPS_GRID = [1e-2, 1e-3, 1e-4, 1e-5]
RANGES = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


def _min_denom(coeffs):
    """Smallest Qmax at which the invariant is exactly representable under
    sr_gb.snap_round's OWN normalisation rule, not the best achievable over
    every possible normalisation. snap_round always divides a candidate
    vector by whichever entry has the largest magnitude (its `dom_idx`,
    sr_gb.py's snap_round: `dom_idx = max(range(len(v)), key=lambda i:
    abs(v[i]))`); it does not search over which entry to normalise by. So the
    denominator that actually has to fit inside Qmax is the one produced by
    THAT fixed choice, which for these constructed systems is always the
    constant term (built as the largest-magnitude coefficient in every row).
    An earlier version of this function minimised over every possible
    normalising entry instead of using snap_round's actual one, which
    understated the true denominator floor (e.g. reporting 13 instead of 19
    for 13x^2+17y^2-19) and manufactured an apparent "turns on past q*, not
    right at it" lag that was really just the true floor being higher than
    reported; see TODO.md item 13.
    """
    coeffs = [int(c) for c in coeffs]
    g = reduce(gcd, (abs(c) for c in coeffs if c != 0))
    prim = [c // g for c in coeffs]
    dom = max(prim, key=abs)
    return max(abs(dom) // gcd(abs(c), abs(dom)) for c in prim if c != 0)


# Arm A systems: (name, expr, integer coefficient vector for the min_denom
# annotation). Coefficients are small coprime integers so the ceiling ladder is
# clean. Constant terms are chosen so the real variety is nonempty over the
# sampling box (leading + cross coeff >= constant at |y|<=1).
RECOVERY_SYSTEMS = [
    ("circle",     "x**2 + y**2 - 1",       (1, 1, -1)),
    ("ellipse_357", "3*x**2 + 5*y**2 - 7",  (3, 5, -7)),
    ("ellipse_5711", "5*x**2 + 7*y**2 - 11", (5, 7, -11)),
    ("ellipse_71113", "7*x**2 + 11*y**2 - 13", (7, 11, -13)),
    ("ellipse_131719", "13*x**2 + 17*y**2 - 19", (13, 17, -19)),
    ("ellipse_171923", "17*x**2 + 19*y**2 - 23", (17, 19, -23)),
]


def generate_cloud(N, seed):
    """Generic point cloud in the box: no polynomial relation up to degree 2."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(N, 2))


def run_recovery(seeds, N):
    """Arm A: recovery vs Qmax at sigma=0."""
    rows = []
    for name, expr, coeffs in RECOVERY_SYSTEMS:
        true_inv = parse_expr(expr)
        md = _min_denom(coeffs)
        for qmax in QMAX_GRID:
            for seed in range(seeds):
                data = generate_variety_data(expr, VAR_NAMES, RANGES,
                                             N=N, sigma=0.0, seed=seed)
                t0 = time.perf_counter()
                gb = sr_gb(data, VAR_NAMES, degree=2, max_denom=qmax,
                           sigma_estimate=0.0)
                dt = time.perf_counter() - t0
                ok = exact_recovery(gb, true_inv)
                rows.append({"system": name, "min_denom": md, "qmax": qmax,
                             "seed": seed, "exact": bool(ok), "runtime_s": dt})
        print(f"[A] {name} (min_denom={md}) done")
    df = pd.DataFrame(rows)
    summ = (df.groupby(["system", "min_denom", "qmax"])
              .agg(rate=("exact", "mean"), k=("exact", "sum"),
                   n=("exact", "size"), med_runtime_s=("runtime_s", "median"))
              .reset_index())
    ci = summ.apply(lambda r: wilson_interval(int(r["k"]), int(r["n"])), axis=1)
    summ["ci_low"] = [c[0] for c in ci]
    summ["ci_high"] = [c[1] for c in ci]
    return df, summ


def run_falsepos(seeds, N):
    """Arm B: false-positive (non-abstention) rate vs Qmax on negatives."""
    rows = []
    negatives = [
        ("generic_cloud", None),
        ("irrational_pi", "x**2 + pi*y**2 - 1"),
    ]
    for name, expr in negatives:
        for qmax in QMAX_GRID:
            for seed in range(seeds):
                if expr is None:
                    data = generate_cloud(N, seed)
                else:
                    data = generate_variety_data(expr, VAR_NAMES, RANGES,
                                                 N=N, sigma=0.0, seed=seed)
                t0 = time.perf_counter()
                gb = sr_gb(data, VAR_NAMES, degree=2, max_denom=qmax,
                           sigma_estimate=0.0)
                dt = time.perf_counter() - t0
                # False positive: any nonzero generator returned. Correct
                # behaviour on these systems is abstention (empty basis).
                fp = _returned_invariant(gb)
                rows.append({"system": name, "qmax": qmax, "seed": seed,
                             "false_positive": bool(fp), "runtime_s": dt})
        print(f"[B] {name} done")
    df = pd.DataFrame(rows)
    summ = (df.groupby(["system", "qmax"])
              .agg(fp_rate=("false_positive", "mean"),
                   k=("false_positive", "sum"), n=("false_positive", "size"),
                   med_runtime_s=("runtime_s", "median"))
              .reset_index())
    ci = summ.apply(lambda r: wilson_interval(int(r["k"]), int(r["n"])), axis=1)
    summ["ci_low"] = [c[0] for c in ci]
    summ["ci_high"] = [c[1] for c in ci]
    return df, summ


def run_eps(seeds, N, sigma=0.01):
    """Arm C: recovery + abstention vs eps (snap/rational-window tolerance)."""
    circle_expr = "x**2 + y**2 - 1"
    circle_inv = parse_expr(circle_expr)
    irr_expr = "x**2 + pi*y**2 - 1"
    rows = []
    for eps in EPS_GRID:
        for seed in range(seeds):
            # recovery under noise
            data = generate_variety_data(circle_expr, VAR_NAMES, RANGES,
                                         N=N, sigma=sigma, seed=seed)
            gb = sr_gb(data, VAR_NAMES, degree=2, eps=eps, sigma_estimate=sigma)
            rec = exact_recovery(gb, circle_inv)
            # abstention on irrational coefficient (sigma=0)
            datai = generate_variety_data(irr_expr, VAR_NAMES, RANGES,
                                          N=N, sigma=0.0, seed=seed)
            gbi = sr_gb(datai, VAR_NAMES, degree=2, eps=eps, sigma_estimate=0.0)
            fp = _returned_invariant(gbi)
            rows.append({"eps": eps, "seed": seed, "recovery": bool(rec),
                         "false_positive": bool(fp)})
        print(f"[C] eps={eps:g} done")
    df = pd.DataFrame(rows)
    summ = (df.groupby("eps")
              .agg(recovery_rate=("recovery", "mean"),
                   k_rec=("recovery", "sum"),
                   fp_rate=("false_positive", "mean"),
                   k_fp=("false_positive", "sum"),
                   n=("recovery", "size"))
              .reset_index())
    return df, summ


def _returned_invariant(gb):
    """True iff sr_gb returned at least one nonzero generator."""
    if not gb:
        return False
    for g in gb:
        expr = g.as_expr() if hasattr(g, "as_expr") else g
        if expr != 0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--quick", action="store_true",
                    help="2 seeds only; same grids/N otherwise")
    args = ap.parse_args()
    if args.quick:
        args.seeds = 2

    print("=" * 72)
    print("ABLATION: Qmax (max_denom) and eps (snap tolerance) sensitivity")
    print(f"seeds={args.seeds}  N={args.N}  Qmax grid={QMAX_GRID}  eps grid={EPS_GRID}")
    print("=" * 72)

    os.makedirs("Results", exist_ok=True)

    print("\n-- Arm A: recovery vs Qmax (sigma=0) --")
    a_full, a_summ = run_recovery(args.seeds, args.N)
    a_full.to_csv("Results/ablation_qmax_recovery.csv", index=False)
    a_summ.to_csv("Results/ablation_qmax_recovery_summary.csv", index=False)

    print("\n-- Arm B: false-positive rate vs Qmax --")
    b_full, b_summ = run_falsepos(args.seeds, args.N)
    b_full.to_csv("Results/ablation_qmax_falsepos.csv", index=False)
    b_summ.to_csv("Results/ablation_qmax_falsepos_summary.csv", index=False)

    print("\n-- Arm C: eps / tolerance sensitivity --")
    c_full, c_summ = run_eps(args.seeds, args.N)
    c_full.to_csv("Results/ablation_eps_sensitivity.csv", index=False)
    c_summ.to_csv("Results/ablation_eps_sensitivity_summary.csv", index=False)

    print("\n=== Arm A: recovery rate by system x Qmax ===")
    piv = a_summ.pivot(index="system", columns="qmax", values="rate")
    print(piv.to_string(float_format=lambda v: f"{v:.0%}"))
    print("\n=== Arm B: false-positive rate by system x Qmax ===")
    pivb = b_summ.pivot(index="system", columns="qmax", values="fp_rate")
    print(pivb.to_string(float_format=lambda v: f"{v:.0%}"))
    print("\n=== Arm C: recovery / false-positive vs eps ===")
    print(c_summ.to_string(index=False))

    print("\nResults written to Results/ablation_qmax_*.csv and ablation_eps_*.csv")


if __name__ == "__main__":
    main()
