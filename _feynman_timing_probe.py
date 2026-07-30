#!/usr/bin/env python3
"""
Timing probe: times SR-GB+CSNP, SINDy-null, and SINDy-ST separately for
every (equation, sigma) cell, averaged over 30 seeds.

Two uses, which is why it runs at the full 30-seed scale rather than a
single seed:

  1. It backs the baseline-cost ratios quoted in Adaptive-CSNP.tex (the
     SINDy-ST and KRONIC wall-time relative to the whole SR-GB+CSNP
     pipeline). Per-cell timing varies noticeably with the seed, and the
     "worst cell" figure is a max over cells, so a single seed gives an
     unstable number; the paper's ratios are meant to be reproducible
     from Results/feynman_timing_probe.csv at the same 30-seed scale as
     every other headline number.
  2. It anchors benchmark_feyman.py's timeout in run_all_experiments.py.
     A single seed would already be enough for that budgeting use on its
     own.

Outputs:
  - Results/feynman_timing_probe.csv       per-cell means over the seeds
  - Results/feynman_timing_probe_full.csv  one row per (equation, sigma, seed)

The printed summary reports the exact ratios the paper cites: aggregate
SINDy-ST / SR-GB+CSNP and KRONIC / SR-GB+CSNP wall time, and the per-cell
median and worst-case SINDy-ST / SR-GB+CSNP ratio, plus the std-dev/IQR of
each method's per-trial wall time (from the full per-seed CSV, not the
per-cell means) that the paper's runtime paragraph was missing.

Pass --from-csv to skip re-timing entirely and recompute this summary from
the already-committed Results/feynman_timing_probe_full.csv, useful for
regenerating the std/IQR numbers without a multi-hour rerun.
"""
import argparse
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sympy import parse_expr, Symbol, Poly
from sr_gb import sr_gb, exact_recovery
from data_generator import generate_variety_data
from feynman_polynomials import feynman_polynomials
from sindy_baselines import sindy_nullspace, sindy_st_ensemble

SEEDS = list(range(30))
N = 5000
NOISE = [0.00, 0.02, 0.05]


def get_degree(expr_str, var_names):
    syms = {v: Symbol(v) for v in var_names}
    try:
        return max(2, Poly(parse_expr(expr_str, local_dict=syms),
                            *[syms[v] for v in var_names]).total_degree())
    except Exception:
        return 2


parser = argparse.ArgumentParser()
parser.add_argument("--from-csv", action="store_true",
                    help="Skip timing; recompute the summary from the "
                         "already-committed Results/feynman_timing_probe_full.csv")
args = parser.parse_args()

if args.from_csv:
    full = pd.read_csv("Results/feynman_timing_probe_full.csv")
else:
    rows = []
    t_start = time.time()
    for name, expr_str, var_names, ranges in feynman_polynomials:
        deg = get_degree(expr_str, var_names)
        for sigma in NOISE:
            syms = {v: Symbol(v) for v in var_names}
            truth = parse_expr(expr_str, local_dict=syms)
            for seed in SEEDS:
                X = generate_variety_data(expr_str, var_names, ranges, N=N,
                                          sigma=sigma, seed=seed)

                t0 = time.time()
                try:
                    gb = sr_gb(X, var_names, degree=deg, sigma_estimate=sigma)
                    srgb_exact = exact_recovery(gb, truth)
                except Exception:
                    srgb_exact = False
                t_srgb = time.time() - t0

                t0 = time.time()
                try:
                    sindy_nullspace(X, var_names, deg, sigma_estimate=sigma,
                                    n_bootstrap=15, bootstrap_frac=0.8)
                except Exception:
                    pass
                t_null = time.time() - t0

                t0 = time.time()
                try:
                    sindy_st_ensemble(X, var_names, deg, sigma_estimate=sigma)
                except Exception:
                    pass
                t_st = time.time() - t0

                rows.append({"name": name, "sigma": sigma, "degree": deg,
                             "seed": seed,
                             "t_srgb": round(t_srgb, 3),
                             "t_sindy_null": round(t_null, 3),
                             "t_sindy_st": round(t_st, 3),
                             "t_total": round(t_srgb + t_null + t_st, 3),
                             "srgb_exact": srgb_exact})
            # progress line per (equation, sigma) cell, averaged over seeds
            cell = [r for r in rows if r["name"] == name and r["sigma"] == sigma]
            m_srgb = np.mean([r["t_srgb"] for r in cell])
            m_null = np.mean([r["t_sindy_null"] for r in cell])
            m_st = np.mean([r["t_sindy_st"] for r in cell])
            print(f"{name:10s} sigma={sigma:.2f} deg={deg} "
                  f"srgb={m_srgb:6.2f}s null={m_null:6.2f}s st={m_st:6.2f}s "
                  f"(mean over {len(SEEDS)} seeds, elapsed {time.time()-t_start:.0f}s)")

    full = pd.DataFrame(rows)
    full.to_csv("Results/feynman_timing_probe_full.csv", index=False)

# Per-cell means over the seeds: the canonical probe CSV.
summary = (full.groupby(["name", "sigma", "degree"], as_index=False)
                .agg(n_seeds=("seed", "count"),
                     t_srgb=("t_srgb", "mean"),
                     t_sindy_null=("t_sindy_null", "mean"),
                     t_sindy_st=("t_sindy_st", "mean"),
                     t_total=("t_total", "mean"),
                     srgb_exact_rate=("srgb_exact", "mean")))
for c in ["t_srgb", "t_sindy_null", "t_sindy_st", "t_total"]:
    summary[c] = summary[c].round(3)
summary["srgb_exact_rate"] = summary["srgb_exact_rate"].round(3)
if not args.from_csv:
    summary.to_csv("Results/feynman_timing_probe.csv", index=False)

# Ratios exactly as the paper cites them.
agg_st = summary["t_sindy_st"].sum() / summary["t_srgb"].sum()
agg_null = summary["t_sindy_null"].sum() / summary["t_srgb"].sum()
cell_ratio = summary["t_sindy_st"] / summary["t_srgb"]

print("\n" + "=" * 70)
print(f"{len(SEEDS)}-seed per-cell-mean total wall time: "
      f"{summary['t_total'].sum():.1f}s ({summary['t_total'].sum()/60:.1f} min)")
print(f"Full {len(SEEDS)}-seed run wall time: {full['t_total'].sum():.1f}s "
      f"({full['t_total'].sum()/3600:.2f} h)")
print("\nBaseline-cost ratios (relative to the whole SR-GB+CSNP pipeline):")
print(f"  aggregate SINDy-ST / SR-GB+CSNP : {agg_st:.2f}x")
print(f"  aggregate KRONIC   / SR-GB+CSNP : {agg_null:.2f}x")
print(f"  per-cell median SINDy-ST / SR-GB+CSNP : {cell_ratio.median():.2f}x")
print(f"  per-cell worst  SINDy-ST / SR-GB+CSNP : {cell_ratio.max():.1f}x "
      f"({summary.loc[cell_ratio.idxmax(), 'name']} "
      f"sigma={summary.loc[cell_ratio.idxmax(), 'sigma']})")

# Per-trial (not per-cell-mean) spread of wall time, from the full CSV.
# The runtime paragraph previously reported only aggregate/median/worst
# ratios and no variance.
print("\nPer-trial wall-time spread (seconds, over all equation x sigma x seed rows):")
for col, label in [("t_srgb", "SR-GB+CSNP"), ("t_sindy_null", "KRONIC"), ("t_sindy_st", "SINDy-ST")]:
    q1, q3 = full[col].quantile([0.25, 0.75])
    print(f"  {label:12s}: mean={full[col].mean():.2f} std={full[col].std():.2f} "
          f"IQR=[{q1:.2f}, {q3:.2f}]")

print("\nSaved Results/feynman_timing_probe.csv (per-cell means) and "
      "Results/feynman_timing_probe_full.csv (per-seed).")
