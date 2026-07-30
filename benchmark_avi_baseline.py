#!/usr/bin/env python3
"""
benchmark_avi_baseline.py - Run the Approximate Vanishing Ideal (AVI) /
Approximate Border Basis (ABM) baseline (avi_baselines.py) on the same
systems used elsewhere in this repo's synthetic / Feynman / Kepler / 2D
harmonic oscillator benchmarks, and compare against SR-GB+CSNP.

For each system and noise level this reports:
  - AVI border basis cardinality (|G|),
  - whether the border basis contains the true invariant up to scalar,
  - whether SR-GB+CSNP's recovered generator is (up to scalar) a member
    of AVI's border basis,
  - runtime.

Systems: circle (x^2+y^2=1), sphere, the cubic variety `algebraic_cubic_toy`
from feynman_polynomials.py's benchmark list, Kepler angular momentum (reusing
benchmark_kepler_angular_momentum.py's exact data generator), the 2D
harmonic oscillator transition invariant (reusing
benchmark_harmonic_oscillator_vs_sindy.py's generator), and a
representative subset of the polynomial Feynman equations
(feynman_polynomials.py), generated via data_generator.generate_variety_data
exactly as benchmark_feyman.py does.

See avi_baselines.py's module docstring for the eps/tolerance discussion;
in particular the harmonic oscillator and noisy cubic cases below are
*expected* to look degenerate (bloated / non-matching border basis) --
this is a genuine property of the fixed-order AVI construction (no
MDL-based rationality selection), not a bug in this baseline, and is
discussed in the accompanying report.
"""

import sys
import os
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sympy import parse_expr, Symbol, Poly

from sr_gb import sr_gb, exact_recovery
from avi_baselines import avi_border_basis
from data_generator import generate_variety_data
from feynman_polynomials import feynman_polynomials
from benchmark_kepler_angular_momentum import generate_kepler_pairs
from benchmark_harmonic_oscillator_vs_sindy import generate_harmonic_trajectory
from utils_stats import wilson_interval


def get_degree(expr_str, var_names):
    """Same convention as benchmark_feyman.py: true total degree, floor 2."""
    syms = {v: Symbol(v) for v in var_names}
    try:
        return max(2, Poly(parse_expr(expr_str, local_dict=syms),
                            *[syms[v] for v in var_names]).total_degree())
    except Exception:
        return 2


# ============================================================================
# System definitions
# ============================================================================
# Each entry: (name, expr_str, var_names, ranges, degree, data_fn)
# data_fn(var_names, ranges, N, sigma, seed) -> (data, expr_str_for_truth)
# For the variety-based systems we reuse generate_variety_data directly; for
# Kepler / harmonic oscillator we reuse their own dedicated generators
# (analytic Kepler solution / linear rotation map), matching the pattern
# their own benchmark scripts use.

def _variety_system(name, expr_str, var_names, ranges, degree=None):
    deg = degree if degree is not None else get_degree(expr_str, var_names)

    def gen(N, sigma, seed):
        return generate_variety_data(expr_str, var_names, ranges,
                                      N=N, sigma=sigma, seed=seed)
    return {"name": name, "expr_str": expr_str, "var_names": var_names,
            "degree": deg, "gen": gen}


def _kepler_system():
    var_names = ["x_t", "y_t", "vx_t", "vy_t",
                 "x_next", "y_next", "vx_next", "vy_next"]
    expr_str = "x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next"

    def gen(N, sigma, seed):
        # Kepler's own generator produces exact (sigma=0) transition pairs;
        # noise is added on top, matching how omp_nullspace's reuse of it
        # is documented (sigma applied post-hoc is not part of the
        # dedicated generator's signature, so we add it here directly).
        data = generate_kepler_pairs(N_pairs=N, dt=0.1, seed=seed)
        if sigma > 0:
            rng = np.random.RandomState(seed)
            data = data + rng.normal(0, sigma, data.shape)
        return data
    return {"name": "kepler_angular_momentum", "expr_str": expr_str,
            "var_names": var_names, "degree": 2, "gen": gen}


def _harmonic_system():
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    expr_str = "x_t**2 + v_t**2 - x_next**2 - v_next**2"

    def gen(N, sigma, seed):
        pairs, _ = generate_harmonic_trajectory(N=N, dt=0.1, sigma=sigma,
                                                  seed=seed)
        return pairs
    return {"name": "harmonic_oscillator_2d", "expr_str": expr_str,
            "var_names": var_names, "degree": 2, "gen": gen}


def build_systems():
    systems = [
        _variety_system("circle", "x**2 + y**2 - 1", ["x", "y"],
                         {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}),
        _variety_system("sphere", "x**2 + y**2 + z**2 - 1", ["x", "y", "z"],
                         {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-1.5, 1.5)}),
        # algebraic_cubic_toy from feynman_polynomials.py: cubic variety in
        # 2 variables (no verified physical content; kept as a polynomial
        # stress case).
        _variety_system("cubic_algebraic_toy", "x**3 + 2*x*y + y**2", ["x", "y"],
                         {"x": (-2, 2), "y": (-2, 2)}),
        _kepler_system(),
        _harmonic_system(),
    ]

    # Representative subset: 3 quadratic + 2 cubic equations, with varying
    # variable counts, drawn directly from feynman_polynomials.py (no
    # reinvented equations/ranges). "I.12.2" is a verified official Feynman
    # ID (Coulomb's law); the other four are honest non-Feynman labels for
    # entries that don't correspond to any official database equation.
    feynman_ids = {"circle_locus", "newtons_second_law", "angular_momentum_2d",
                   "I.12.2", "kinematics_position"}
    by_id = {fid: (expr_str, var_names, ranges)
             for fid, expr_str, var_names, ranges in feynman_polynomials}
    for fid in feynman_ids:
        expr_str, var_names, ranges = by_id[fid]
        systems.append(_variety_system(f"feynman_{fid}", expr_str, var_names,
                                        ranges))
    return systems


# ============================================================================
# Benchmark runner
# ============================================================================
def run_system(system, sigma, seed, N):
    name = system["name"]
    var_names = system["var_names"]
    deg = system["degree"]
    true_expr = parse_expr(system["expr_str"])

    row = {"system": name, "n_vars": len(var_names), "true_degree": deg,
           "sigma": sigma, "seed": seed, "avi_cardinality": 0,
           "avi_runtime_sec": 0.0, "srgb_exact": False,
           "avi_contains_true": False, "avi_contains_srgb_generator": None,
           "note": ""}

    try:
        data = system["gen"](N, sigma, seed)
    except Exception as e:
        row["note"] = f"data_gen_failed: {e}"
        return row
    if len(data) < N // 2:
        row["note"] = "insufficient_data"
        return row

    # ---- AVI border basis ----
    t0 = time.time()
    try:
        G = avi_border_basis(data, var_names, max_degree=deg,
                              sigma_estimate=sigma)
    except Exception as e:
        row["note"] = f"avi_failed: {e}"
        G = []
    row["avi_runtime_sec"] = time.time() - t0
    row["avi_cardinality"] = len(G)
    try:
        row["avi_contains_true"] = bool(exact_recovery(G, true_expr))
    except Exception:
        row["avi_contains_true"] = False

    # ---- SR-GB+CSNP recovered generator ----
    try:
        gb = sr_gb(data, var_names, degree=deg, sigma_estimate=sigma)
    except Exception as e:
        gb = []
        row["note"] = (row["note"] + f"; srgb_failed: {e}").strip("; ")
    srgb_exact = False
    try:
        srgb_exact = bool(exact_recovery(gb, true_expr))
    except Exception:
        pass
    row["srgb_exact"] = srgb_exact

    if gb:
        srgb_gen = gb[0].as_expr() if hasattr(gb[0], "as_expr") else gb[0]
        try:
            row["avi_contains_srgb_generator"] = bool(
                exact_recovery(G, srgb_gen))
        except Exception:
            row["avi_contains_srgb_generator"] = False
    else:
        # SR-GB+CSNP produced nothing to compare against; fall back to the
        # ground-truth invariant itself and flag this explicitly.
        row["avi_contains_srgb_generator"] = row["avi_contains_true"]
        row["note"] = (row["note"] + "; srgb_empty_fallback_to_truth").strip("; ")

    return row


def run_benchmark(n_seeds=5, N=1000, sigmas=(0.0, 0.01, 0.02)):
    systems = build_systems()
    rows = []
    for system in systems:
        for sigma in sigmas:
            for seed in range(n_seeds):
                row = run_system(system, sigma, seed, N)
                rows.append(row)
                print(f"{row['system']:20s} sigma={sigma:<5} seed={seed} "
                      f"|G|={row['avi_cardinality']:3d} "
                      f"avi_contains_true={row['avi_contains_true']} "
                      f"srgb_exact={row['srgb_exact']} "
                      f"t={row['avi_runtime_sec']:.3f}s"
                      + (f"  [{row['note']}]" if row["note"] else ""))

    df = pd.DataFrame(rows)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/avi_baseline_results.csv", index=False)

    summary_rows = []
    for (name, sigma), grp in df.groupby(["system", "sigma"]):
        n = len(grp)
        k_true = int(grp["avi_contains_true"].sum())
        ci = wilson_interval(k_true, n)
        summary_rows.append({
            "system": name, "sigma": sigma, "n_seeds": n,
            "mean_avi_cardinality": grp["avi_cardinality"].mean(),
            "avi_contains_true_rate": k_true / n,
            "avi_contains_true_ci_low": ci[0],
            "avi_contains_true_ci_high": ci[1],
            "srgb_exact_rate": grp["srgb_exact"].mean(),
            "mean_avi_runtime_sec": grp["avi_runtime_sec"].mean(),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("Results/avi_baseline_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("AVI Border Basis Baseline Summary")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("\nResults saved to Results/avi_baseline_results.csv and "
          "Results/avi_baseline_summary.csv")
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N/sigma grid as full run")
    args = parser.parse_args()
    if args.quick:
        run_benchmark(n_seeds=2, N=1000, sigmas=(0.0, 0.01, 0.02))
    else:
        # Matches the 30-seed/N=5000 headline transition-invariant config.
        # NOTE: sec:avi-baseline in Adaptive-CSNP.tex currently reports
        # numbers (cardinality means, per-seed anecdotes) computed from the
        # prior 5-seed/N=1000 run; that prose needs updating from a fresh
        # run at this config before submission.
        run_benchmark(n_seeds=30, N=5000, sigmas=(0.0, 0.01, 0.02))
