#!/usr/bin/env python3
"""
benchmark_feyman.py – Full Feynman polynomial benchmark.
Add --only-srgb to skip baselines.

With the all-targets implicit STLSQ SINDy-ST baseline, the 1-seed probe
measures ~242s/seed locally (Results/feynman_timing_probe.csv), so the full
30-seed run is ~2h local and ~2.3-3.3h on a slower hosted CPU, and runs in a
single session alongside the rest of run_all_experiments.py. Every trial
reseeds numpy's global RNG from just `seed` (data_generator.generate_variety_data),
not from (equation, sigma, seed) jointly, so any rerun reproduces exactly the
same trials. But for a fixed seed, the free-variable draws and the
pre-scaling noise draws are identical across sigma (a deliberate common-
random-numbers pattern, not independent noise per sigma level).
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sympy import parse_expr, Symbol, Poly, cancel
from sr_gb import sr_gb, exact_recovery
from data_generator import generate_variety_data
from feynman_polynomials import feynman_polynomials
from utils_stats import wilson_interval
from sindy_baselines import sindy_nullspace, sindy_st_ensemble

ALL_METHODS = ("srgb", "null", "st")


def get_degree(expr_str, var_names):
    syms = {v: Symbol(v) for v in var_names}
    try:
        return max(2, Poly(parse_expr(expr_str, local_dict=syms),
                           *[syms[v] for v in var_names]).total_degree())
    except Exception:
        return 2


def _empty_row(name, sigma, seed, deg, note=""):
    # Method columns start as NaN (rendered as empty CSV cells); a method
    # this run was asked to skip (--only-srgb) stays NaN so downstream
    # readers can distinguish "not computed" from "failed".
    return {"name": name, "sigma": sigma, "seed": seed, "degree": deg,
            "srgb_exact": np.nan, "srgb_red": np.nan,
            "null_exact": np.nan, "null_red": np.nan,
            "sindy_st_exact": np.nan, "sindy_st_red": np.nan,
            "runtime": 0.0, "note": note}


def run_single(name, expr_str, var_names, ranges, sigma, seed, N,
               methods=ALL_METHODS):
    t0 = time.time()
    syms = {v: Symbol(v) for v in var_names}
    truth = parse_expr(expr_str, local_dict=syms)
    deg = get_degree(expr_str, var_names)

    X = generate_variety_data(expr_str, var_names, ranges, N=N, sigma=sigma, seed=seed)
    if len(X) < N // 2:
        row = _empty_row(name, sigma, seed, deg, note="insufficient_data")
        for m, exact_col, red_col in (("srgb", "srgb_exact", "srgb_red"),
                                      ("null", "null_exact", "null_red"),
                                      ("st", "sindy_st_exact", "sindy_st_red")):
            if m in methods:
                row[exact_col], row[red_col] = False, 0
        return row

    row = _empty_row(name, sigma, seed, deg)

    # ---- SR-GB+CSNP ----
    if "srgb" in methods:
        try:
            gb = sr_gb(X, var_names, degree=deg, sigma_estimate=sigma)
            row["srgb_exact"] = exact_recovery(gb, truth)
            row["srgb_red"] = len(gb)
        except Exception:
            row["srgb_exact"], row["srgb_red"] = False, 0

    # Baselines run at the library-default settings, the same configuration
    # every other benchmark in this repo uses: KRONIC at n_bootstrap=15
    # (_verify_bootstrap_reduction.py measures that reducing this to 5 could
    # only ever under-credit the baseline, never inflate it, so the full 15 is
    # kept), and SINDy-ST as the all-targets implicit STLSQ ensemble (see
    # sindy_st_ensemble's docstring: a single-target Lasso-alpha ensemble is
    # structurally unable to recover 23/26 of these equations regardless of
    # regularization, and Lasso is not the STLSQ the paper cites).
    # ---- SINDy-null (KRONIC) ----
    if "null" in methods:
        null_exact = False
        try:
            null_cands = sindy_nullspace(X, var_names, deg, sigma_estimate=sigma,
                                         n_bootstrap=15, bootstrap_frac=0.8)
        except Exception:
            null_cands = []
        row["null_red"] = len(null_cands)
        if null_cands:
            for p in null_cands:
                try:
                    ratio = cancel(p / truth)
                    if ratio.is_number or ratio.is_constant():
                        null_exact = True
                        break
                except:
                    pass
        row["null_exact"] = null_exact

    # ---- SINDy-ST (all-targets implicit STLSQ ensemble) ----
    if "st" in methods:
        st_exact, st_red = False, 0
        try:
            st_cands = sindy_st_ensemble(X, var_names, deg, sigma_estimate=sigma)
            st_red = len(st_cands)
            if st_cands:
                for p in st_cands:
                    try:
                        ratio = cancel(p / truth)
                        if ratio.is_number or ratio.is_constant():
                            st_exact = True
                            break
                    except:
                        pass
        except Exception:
            st_exact, st_red = False, 0
        row["sindy_st_exact"], row["sindy_st_red"] = st_exact, st_red

    row["runtime"] = round(time.time() - t0, 3)
    return row


def summarize(df):
    """Per-(equation, sigma) summary and per-sigma overall table with Wilson
    CIs. NaN-aware so that method-restricted runs (--only-srgb) get rates and
    CIs over the trials that actually ran each method; on a complete run this
    reduces to the previous behavior exactly."""
    summary = df.groupby(["name", "sigma"]).agg(
        srgb_rate=("srgb_exact", "mean"),
        srgb_red=("srgb_red", "mean"),
        null_rate=("null_exact", "mean"),
        null_red=("null_red", "mean"),
        sindy_st_rate=("sindy_st_exact", "mean"),
        sindy_st_red=("sindy_st_red", "mean"),
        runtime=("runtime", "mean")
    ).reset_index().round(3)

    overall = df.groupby("sigma").agg(
        srgb_rate=("srgb_exact", "mean"),
        null_rate=("null_exact", "mean"),
        sindy_st_rate=("sindy_st_exact", "mean"),
        n_eqs=("name", "nunique")
    ).reset_index().round(3)

    for idx, row in overall.iterrows():
        mask = df['sigma'] == row['sigma']
        for col, prefix in (("srgb_exact", "srgb"), ("null_exact", "null"),
                            ("sindy_st_exact", "sindy_st")):
            vals = df.loc[mask, col].dropna()
            if len(vals) == 0:
                lo = hi = np.nan
            else:
                lo, hi = wilson_interval(vals.sum(), len(vals))
            overall.loc[idx, f'{prefix}_ci_low'] = lo
            overall.loc[idx, f'{prefix}_ci_high'] = hi
    return summary, overall


def write_results(df, summary, overall, output_dir, tag=""):
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    df.to_csv(os.path.join(output_dir, f"feynman_results_full{suffix}.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"feynman_results_summary{suffix}.csv"), index=False)
    overall.to_csv(os.path.join(output_dir, f"feynman_results_overall{suffix}.csv"), index=False)
    print(f"Saved: feynman_results_full{suffix}.csv (+summary/overall) in {output_dir}")


def run_benchmark(equations, noise_levels, N, seed_range, output_dir,
                  methods=ALL_METHODS, tag=""):
    seeds = list(seed_range)
    rows = []
    total = len(equations) * len(noise_levels) * len(seeds)
    done = 0

    def mark(row, key, method):
        if method not in methods:
            return "N/A"
        return "✓" if row[key] else "✗"

    for name, expr_str, var_names, ranges in equations:
        for sigma in noise_levels:
            for seed in seeds:
                done += 1
                row = run_single(name, expr_str, var_names, ranges, sigma, seed, N, methods)
                print(f"[{done:>3}/{total}] {name:10s} σ={sigma:.2f} s={seed} "
                      f"SRGB={mark(row, 'srgb_exact', 'srgb')} "
                      f"NULL={mark(row, 'null_exact', 'null')} "
                      f"ST={mark(row, 'sindy_st_exact', 'st')} t={row['runtime']:.3f}s")
                rows.append(row)

    df = pd.DataFrame(rows)
    summary, overall = summarize(df)

    method_names = {"srgb": "SR-GB+CSNP", "null": "KRONIC", "st": "SINDy-ST"}
    print("\n" + "=" * 60)
    print("Feynman Benchmark Results")
    print(f"(Methods: {', '.join(method_names[m] for m in methods)}; "
          f"seeds {seeds[0]}..{seeds[-1]})")
    print("=" * 60)
    print(overall[['sigma', 'srgb_rate', 'null_rate', 'sindy_st_rate']].to_string(index=False))

    write_results(df, summary, overall, output_dir, tag)
    return df, summary, overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="Results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only-srgb", action="store_true",
                        help="Skip KRONIC and SINDy-ST baselines; run only SR-GB+CSNP "
                             "(writes tagged CSVs so it never clobbers a full run)")
    args = parser.parse_args()

    if args.only_srgb:
        methods, tag = ("srgb",), "srgb-only"
    else:
        methods, tag = ALL_METHODS, ""

    n_seeds = 1 if args.quick else args.seeds

    N = args.N
    noise = [0.00, 0.02, 0.05]

    if args.quick:
        print(f"=== QUICK MODE ({len(feynman_polynomials)} equations, 1 seed, N={N}) ===\n")

    run_benchmark(feynman_polynomials, noise, N, range(n_seeds),
                  args.outdir, methods=methods, tag=tag)
