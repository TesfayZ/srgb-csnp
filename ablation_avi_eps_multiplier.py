#!/usr/bin/env python3
"""
ablation_avi_eps_multiplier.py - Sensitivity of the AVI competitor
baseline's (avi_baselines.py) relative-residual tolerance multiplier k in
eps = max(1e-6, k*sigma_estimate) to k.

A sensitivity ablation, not a search for the best-scoring k: per
Adaptive-CSNP.tex Table tab:thresholds, hand-set tolerances in this project
are characterized across a range rather than tuned to whichever value
scores best, and tuning a competitor baseline's own tolerance to the
comparison outcome would additionally be a fairness problem. This script
characterizes AVI's shipped default (`avi_baselines._AVI_EPS_MULTIPLIER`)
across a range without changing it.

For each system x sigma x k, reports:
  - avi_cardinality: size of the returned border basis (too-small k:
    under-merging, nothing looks dependent, cardinality inflates; too-large
    k: over-merging, unrelated monomials look dependent, cardinality and
    correctness both collapse).
  - avi_contains_true: whether the true invariant is (up to scalar) still a
    member of the border basis -- the direct correctness signal.

sigma=0 is excluded from the sweep: eps=max(1e-6, k*0)=1e-6 regardless of
k, so the multiplier has no effect there.

Systems: a representative subset of benchmark_avi_baseline.py's own system
set (reusing its build_systems(), not reinvented) -- circle and sphere as
clean single-invariant varieties, and harmonic_oscillator_2d because
avi_baselines.py's own docstring already flags it as the case expected to
show degenerate (bloated) border-basis behaviour, making it the most
informative stress case for this sweep.

Saves:
  - Results/ablation_avi_eps_multiplier.csv
  - Results/ablation_avi_eps_multiplier_summary.csv

Accepts --quick (2 seeds, same grids).
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sympy import parse_expr

from sr_gb import exact_recovery
import avi_baselines
from avi_baselines import avi_border_basis
from benchmark_avi_baseline import build_systems
from utils_stats import wilson_interval

MULTIPLIER_GRID = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0]
SIGMA_GRID = [0.01, 0.02, 0.05]
SYSTEM_NAMES = {"circle", "sphere", "harmonic_oscillator_2d"}
N = 1000


def run_sweep(seeds):
    systems = [s for s in build_systems() if s["name"] in SYSTEM_NAMES]
    rows = []
    for system in systems:
        name = system["name"]
        var_names = system["var_names"]
        deg = system["degree"]
        true_expr = parse_expr(system["expr_str"])
        for sigma in SIGMA_GRID:
            for k in MULTIPLIER_GRID:
                eps = max(1e-6, k * sigma)
                for seed in range(seeds):
                    try:
                        data = system["gen"](N, sigma, seed)
                        G = avi_border_basis(data, var_names, max_degree=deg,
                                              eps=eps, sigma_estimate=sigma)
                    except Exception:
                        G = []
                    try:
                        contains_true = bool(exact_recovery(G, true_expr))
                    except Exception:
                        contains_true = False
                    rows.append({"system": name, "sigma": sigma, "k": k,
                                 "seed": seed, "avi_cardinality": len(G),
                                 "avi_contains_true": contains_true})
            print(f"[AVI eps ablation] {name} sigma={sigma} done")
    df = pd.DataFrame(rows)
    summ = (df.groupby(["system", "sigma", "k"])
              .agg(mean_cardinality=("avi_cardinality", "mean"),
                   contains_true_k=("avi_contains_true", "sum"),
                   n=("avi_contains_true", "size"))
              .reset_index())
    summ["contains_true_rate"] = summ["contains_true_k"] / summ["n"]
    ci = summ.apply(lambda r: wilson_interval(
        int(r["contains_true_k"]), int(r["n"])), axis=1)
    summ["ci_low"] = [c[0] for c in ci]
    summ["ci_high"] = [c[1] for c in ci]
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="2 seeds only; same grids")
    args = ap.parse_args()
    seeds = 2 if args.quick else args.seeds

    print("=" * 72)
    print("ABLATION: AVI baseline eps multiplier (eps = k*sigma) sensitivity")
    print(f"seeds={seeds}  multiplier grid={MULTIPLIER_GRID}  sigma grid={SIGMA_GRID}")
    print(f"shipped default: k={avi_baselines._AVI_EPS_MULTIPLIER} (avi_baselines.py)")
    print("=" * 72)

    os.makedirs("Results", exist_ok=True)
    full, summ = run_sweep(seeds)
    full.to_csv("Results/ablation_avi_eps_multiplier.csv", index=False)
    summ.to_csv("Results/ablation_avi_eps_multiplier_summary.csv", index=False)

    for name in sorted(summ["system"].unique()):
        print(f"\n=== {name}: contains_true_rate by sigma x k ===")
        piv = summ[summ["system"] == name].pivot(
            index="sigma", columns="k", values="contains_true_rate")
        print(piv.to_string(float_format=lambda v: f"{v:.0%}"))
        print(f"=== {name}: mean cardinality by sigma x k ===")
        pivc = summ[summ["system"] == name].pivot(
            index="sigma", columns="k", values="mean_cardinality")
        print(pivc.to_string(float_format=lambda v: f"{v:.2f}"))

    print(f"\nShipped default k={avi_baselines._AVI_EPS_MULTIPLIER} is unchanged by this ablation.")
    print("Results written to Results/ablation_avi_eps_multiplier*.csv")


if __name__ == "__main__":
    main()
