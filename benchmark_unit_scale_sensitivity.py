#!/usr/bin/env python3
"""
benchmark_unit_scale_sensitivity.py - unit-invariance stress test.

The rationality cost R(c) is not invariant to variable rescaling: relabel
v as c*v for some arbitrary (unit-conversion-like) constant c and a
coefficient that was 1 becomes 1/c^2, which need not be a simple rational
any more. Every benchmark elsewhere in this repo is generated in "nice"
units by construction, so this has never been stress-tested directly.

For four benchmarks (circle, the fixed-dt harmonic oscillator transition
invariant, Kepler angular momentum, and feynman_polynomials.py's
`momentum_conservation_1d` entry, whose five variables span three distinct
physical dimensions: mass, velocity, momentum), each physically distinct
quantity is rescaled by an independent
random factor, and the pipeline is rerun under three conditions:

  1. baseline       - no rescaling (the same units already used elsewhere).
  2. rescaled        - after the random per-quantity rescaling, no
                        preprocessing.
  3. standardized    - after the same rescaling, each column divided by
                        its own empirical standard deviation before the
                        monomial library is built.

"Exact recovery" is checked against the true invariant expressed in
whatever coordinate system the pipeline actually saw, not the original
one, via a tolerant coefficient-vector proportionality check
(sympy's exact `cancel`-based check is unsuitable here since the rescaled
target's coefficients are themselves floats, not clean rationals).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sympy import symbols, sympify, expand, Poly, Symbol
import warnings
warnings.filterwarnings('ignore')

from sr_gb import sr_gb
from data_generator import generate_variety_data
from utils_stats import wilson_interval
from feynman_polynomials import feynman_polynomials
from benchmark_harmonic_oscillator_vs_sindy import generate_harmonic_trajectory
from benchmark_kepler_angular_momentum import generate_kepler_pairs

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def numeric_proportional(expr1, expr2, var_syms, rtol=1e-2):
    """Tolerant replacement for exact_recovery: checks whether the
    coefficient vectors of expr1 and expr2 (over the union of monomials
    appearing in either) are proportional up to rtol, allowing for an
    overall sign flip. Needed because the rescaled/standardized "true"
    target has float, not clean-rational, coefficients."""
    try:
        p1 = Poly(expand(expr1), *var_syms)
        p2 = Poly(expand(expr2), *var_syms)
    except Exception:
        return False
    d1, d2 = p1.as_dict(), p2.as_dict()
    keys = set(d1) | set(d2)
    v1 = np.array([float(d1.get(k, 0)) for k in keys])
    v2 = np.array([float(d2.get(k, 0)) for k in keys])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return False
    v1n, v2n = v1 / n1, v2 / n2
    diff = min(np.linalg.norm(v1n - v2n), np.linalg.norm(v1n + v2n))
    return diff < rtol


def check_recovery_scaled(candidates, true_expr_transformed, var_syms):
    if not candidates:
        return False
    for p in candidates:
        e = p.as_expr() if hasattr(p, "as_expr") else p
        if e == 0:
            continue
        if numeric_proportional(e, true_expr_transformed, var_syms):
            return True
    return False


def apply_scale(data, var_names, groups, rng, lo=0.15, hi=8.0):
    """Draws one random scale factor per group (log-uniform in [lo,hi])
    and returns (scaled_data, k) where k[i] is the factor applied to
    column i (var_names[i])."""
    group_keys = sorted(set(groups.values()))
    group_scale = {g: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
                   for g in group_keys}
    k = np.array([group_scale[groups[v]] for v in var_names])
    return data * k[np.newaxis, :], k


def transform_true_expr(true_expr, var_names, k):
    var_syms = symbols(var_names)
    subs = {var_syms[i]: var_syms[i] / k[i] for i in range(len(var_names))}
    return true_expr.subs(subs)


# ---------------------------------------------------------------------
# Benchmark definitions: (name, generator(N,seed) -> (data, var_names),
#                         true_expr_str, groups, D_max)
# ---------------------------------------------------------------------

def gen_circle(N, seed):
    var_names = ["x", "y"]
    data = generate_variety_data("x**2+y**2-1", var_names,
                                  {"x": (-2, 2), "y": (-2, 2)}, N=N, sigma=0.0, seed=seed)
    return data, var_names


def gen_oscillator(N, seed):
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    pairs, _ = generate_harmonic_trajectory(N=N, dt=0.1, sigma=0.0, seed=seed)
    return pairs, var_names


def gen_kepler(N, seed):
    var_names = ["x_t", "y_t", "vx_t", "vy_t", "x_next", "y_next", "vx_next", "vy_next"]
    data = generate_kepler_pairs(N_pairs=N, dt=0.1, seed=seed)
    return data, var_names


def gen_feynman_momentum_conservation(N, seed):
    entry = [e for e in feynman_polynomials if e[0] == "momentum_conservation_1d"][0]
    _, expr_str, var_names, ranges = entry
    data = generate_variety_data(expr_str, var_names, ranges, N=N, sigma=0.0, seed=seed)
    return data, var_names


BENCHMARKS = [
    {
        "name": "circle",
        "gen": gen_circle,
        "true_expr_str": "x**2+y**2-1",
        "groups": {"x": "gx", "y": "gy"},
        "D_max": 2,
    },
    {
        "name": "oscillator_fixed_dt",
        "gen": gen_oscillator,
        "true_expr_str": "x_t**2+v_t**2-x_next**2-v_next**2",
        "groups": {"x_t": "pos", "x_next": "pos", "v_t": "vel", "v_next": "vel"},
        "D_max": 2,
    },
    {
        "name": "kepler_angular_momentum",
        "gen": gen_kepler,
        "true_expr_str": "x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next",
        "groups": {"x_t": "pos", "y_t": "pos", "x_next": "pos", "y_next": "pos",
                   "vx_t": "vel", "vy_t": "vel", "vx_next": "vel", "vy_next": "vel"},
        "D_max": 2,
    },
    {
        "name": "feynman_momentum_conservation_1d",
        "gen": gen_feynman_momentum_conservation,
        "true_expr_str": "m1*v1 + m2*v2 - P",
        "groups": {"m1": "mass", "m2": "mass", "v1": "vel", "v2": "vel", "P": "momentum"},
        "D_max": 2,
    },
]


def run_one(bench, condition, seed, N=5000, rng_seed_offset=0):
    var_names = None
    data, var_names = bench["gen"](N, seed)
    true_expr = sympify(bench["true_expr_str"], locals={v: Symbol(v) for v in var_names})
    var_syms = symbols(var_names)

    rng = np.random.RandomState(seed + rng_seed_offset)

    if condition == "baseline":
        k = np.ones(len(var_names))
        run_data = data
    else:
        run_data, k = apply_scale(data, var_names, bench["groups"], rng)
        if condition == "standardized":
            std = run_data.std(axis=0)
            std[std == 0] = 1.0
            run_data = run_data / std[np.newaxis, :]
            k = k / std

    target_expr = transform_true_expr(true_expr, var_names, k)

    t0 = time.time()
    try:
        gb = sr_gb(run_data, var_names, degree=None, D_max=bench["D_max"], sigma_estimate=0.0)
    except Exception as e:
        print(f"  {bench['name']} / {condition} seed {seed}: sr_gb error: {e}")
        gb = []
    elapsed = time.time() - t0
    ok = check_recovery_scaled(gb, target_expr, var_syms)
    return ok, elapsed


def run(n_seeds=30, N=5000):
    conditions = ["baseline", "rescaled", "standardized"]
    rows = []
    for bench in BENCHMARKS:
        for condition in conditions:
            print(f"\n=== {bench['name']} / {condition} ===")
            t0 = time.time()
            for seed in range(n_seeds):
                cond_key = "baseline" if condition == "baseline" else condition
                ok, elapsed = run_one(bench, cond_key, seed, N=N)
                rows.append({"benchmark": bench["name"], "condition": condition,
                             "seed": seed, "exact": ok, "elapsed_s": elapsed})
            print(f"  {n_seeds} seeds in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby(["benchmark", "condition"]).agg(
        rate=("exact", "mean"), n=("exact", "count")
    ).reset_index()
    summary["ci_low"] = summary.apply(lambda r: wilson_interval(int(r["rate"] * r["n"]), int(r["n"]))[0], axis=1)
    summary["ci_high"] = summary.apply(lambda r: wilson_interval(int(r["rate"] * r["n"]), int(r["n"]))[1], axis=1)

    print("\n" + "=" * 70)
    print("Unit scale sensitivity summary")
    print("=" * 70)
    print(summary.to_string(index=False))

    df.to_csv(os.path.join(RESULTS_DIR, "unit_scale_sensitivity_results.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "unit_scale_sensitivity_summary.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/unit_scale_sensitivity_*.csv")
    return df, summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        n_seeds = 2
    elif len(sys.argv) > 1:
        n_seeds = int(sys.argv[1])
    else:
        n_seeds = 30
    run(n_seeds=n_seeds)
