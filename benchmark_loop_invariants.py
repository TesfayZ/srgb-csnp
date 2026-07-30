#!/usr/bin/env python3
"""
benchmark_loop_invariants.py - Loop invariant discovery with exact-arithmetic baselines.

Tests SR-GB+CSNP against exact-integer versions of SINDy-null, OMP, and
Dense SVD+GB on classic loop-invariant benchmarks (sum, ps2, ps3, egcd, hash),
plus SINDy-ST in its standard all-targets implicit STLSQ configuration
(sindy_baselines.sindy_st_ensemble, the same code path the Feynman benchmark
runs).

The NULLSPACE baselines all share the same exact sympy integer nullspace
computed from the traces (their most favourable setting). SR-GB+CSNP, by
contrast, runs the
real shipped pipeline end-to-end: `sr_gb(..., full_nullspace=True)` on the raw
float trace matrix, i.e. float SVD with column preconditioning (which is what
makes ps3's ~23-orders-of-magnitude library tractable in floating point) and
full-nullspace deflation for the multi-invariant case (egcd's two Bezout
identities). This is deliberately HARDER for SR-GB+CSNP than the baselines'
exact-arithmetic setting, and routes through sr_gb() rather than a private
bb_search/deflation reimplementation.

Usage:
    python benchmark_loop_invariants.py            # 30 seeds, all benchmarks
    python benchmark_loop_invariants.py --seeds 10 # fewer seeds
    python benchmark_loop_invariants.py sum        # single benchmark
"""

import os
import sys
import numpy as np
import pandas as pd
import sympy as sp
import warnings
from itertools import combinations, combinations_with_replacement, product
from math import comb
from scipy.linalg import svd
from sklearn.linear_model import OrthogonalMatchingPursuit
import heapq
import time

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Import shared modules
# ----------------------------------------------------------------------
from sr_gb import (
    build_monomial_library, snap_round, estimate_rank,
    rationality_cost, _l1_nullspace_fallback,
    exact_recovery, reduce_to_minimal_generator, sr_gb
)
from sindy_baselines import sindy_st_ensemble
from utils_stats import wilson_interval

# ----------------------------------------------------------------------
# Exact nullspace computation (shared by all methods)
# ----------------------------------------------------------------------
def exact_monomial_matrix(var_names, max_degree, data_rows):
    """Build monomial matrix with exact integers (no float64)."""
    n_vars = len(var_names)
    sym_vars = sp.symbols(var_names)
    exponents = []
    for d in range(0, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            exp = [0] * n_vars
            for idx in combo:
                exp[idx] += 1
            exponents.append(tuple(exp))
    monomials = []
    for exp in exponents:
        mon = 1
        for i, e in enumerate(exp):
            if e > 0:
                mon *= sym_vars[i] ** e
        monomials.append(mon)
    rows = []
    for row in data_rows:
        vals = []
        for exp in exponents:
            v = 1
            for i, e in enumerate(exp):
                if e > 0:
                    v *= row[i] ** e
            vals.append(v)
        rows.append(vals)
    Phi = sp.Matrix(rows)
    return Phi, monomials, sym_vars, exponents

def exact_nullspace(Phi):
    """Return nullspace basis as list of column vectors (sympy matrices)."""
    return Phi.nullspace()

# ----------------------------------------------------------------------
# Shared sparse recovery helpers (operate on exact nullspace basis)
# ----------------------------------------------------------------------
def l1_fallback_on_nullspace(V_null_np, Phi_np, tau_resid):
    """L1 fallback to find sparse null vector (uses float, but input is exact)."""
    c = _l1_nullspace_fallback(V_null_np, Phi_np, tau_resid)
    return c

def omp_on_nullspace(V_null_np, Phi_np, tau_resid, n_nonzero=4):
    """OMP to find sparse null vector."""
    M, d = V_null_np.shape
    if d <= 1:
        return V_null_np[:, 0] if d == 1 else None
    target = V_null_np[:, 0]
    X = V_null_np[:, 1:]
    if X.shape[1] == 0:
        return target
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=min(n_nonzero, X.shape[1]), fit_intercept=False)
    omp.fit(X, target)
    alpha = np.zeros(d)
    alpha[0] = 1.0
    alpha[1:] = omp.coef_
    c = V_null_np @ alpha
    norm = np.linalg.norm(c)
    if norm < 1e-10:
        return None
    c = c / norm
    resid = np.linalg.norm(Phi_np @ c)
    if resid > tau_resid:
        return None
    return c

def run_sindy_st_ensemble(X_float, var_names, max_deg):
    """SINDy-ST in its standard configuration: the all-targets implicit
    STLSQ ensemble from sindy_baselines (true lstsq + hard thresholding,
    every library monomial cycled once as the regression target), identical
    to the code path the Feynman and holonomic benchmarks run. Returns the
    candidate list (possibly empty). A single-target Lasso variant is avoided
    here: it is structurally unable to recover any relation not containing the
    strongest-signal monomial."""
    try:
        return sindy_st_ensemble(X_float, var_names, max_deg, sigma_estimate=0.0)
    except Exception:
        return []

def dense_svd_exact(V_null_np, Phi_np, tau_resid):
    """Dense SVD: take trailing vector if d=1, else L1 fallback."""
    M, d = V_null_np.shape
    if d == 1:
        return V_null_np[:, 0]
    else:
        return l1_fallback_on_nullspace(V_null_np, Phi_np, tau_resid)

# ----------------------------------------------------------------------
# Deflation: recover multiple independent generators from a nullspace whose
# dimension d > 1 by repeatedly solving, projecting the found direction out,
# and re-solving in the residual subspace (egcd has d=2: the Bezout identity
# holds separately for the old_r/old_s/old_t triple and the r/s/t triple, so
# a single solve can only ever return one of the two, or an arbitrary linear
# mixture of both -- exactly the KRONIC/SINDy ambiguity the paper's method
# is meant to resolve). Used by the BASELINES only; SR-GB+CSNP's own
# multi-invariant deflation runs inside sr_gb(..., full_nullspace=True).
# ----------------------------------------------------------------------
def deflate_search(solve_fn, V_null_np, max_invariants, monomials, sym_vars, sigma_estimate=0.0):
    V_curr = V_null_np.copy()
    generators = []
    gb_lists = []
    for iteration in range(max_invariants):
        if V_curr.shape[1] == 0:
            break
        c = solve_fn(V_curr, iteration)
        if c is None:
            break
        gb_list, poly_expr = build_polynomial_from_coeff(c, monomials, sym_vars, sigma_estimate)
        if gb_list is None:
            break
        generators.append(poly_expr)
        gb_lists.append(gb_list)

        norm = np.linalg.norm(c)
        if norm < 1e-12:
            break
        c_unit = np.asarray(c, dtype=float) / norm
        V_curr = V_curr - c_unit.reshape(-1, 1) @ (c_unit.reshape(1, -1) @ V_curr)
        if V_curr.shape[1] == 0:
            break
        U2, s2, Vt2 = svd(V_curr, full_matrices=False)
        tol = max(1e-8, 1e-6 * s2[0]) if len(s2) > 0 else 1e-8
        r2 = int(np.sum(s2 > tol))
        V_curr = U2[:, :r2]
    return generators, gb_lists

# ----------------------------------------------------------------------
# Post-processing: build polynomial and canonicalize
# ----------------------------------------------------------------------
def build_polynomial_from_coeff(c, monomials, sym_vars, sigma_estimate=0.0):
    """Snap round, build poly, reduce to minimal generator, compute GB."""
    if c is None:
        return None, None
    c = np.asarray(c).flatten()
    c[np.abs(c) < 1e-3] = 0.0
    rounded = snap_round(c, sigma_estimate)
    poly_expr = sum(sp.Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
    if poly_expr == 0:
        return None, None
    # Reduce to minimal generator (algebraic minimality)
    poly_expr = reduce_to_minimal_generator(poly_expr, None, sigma_estimate, sym_vars=sym_vars)
    if poly_expr == 0:
        return None, None
    # Normalize leading coefficient
    poly_expr = poly_expr.expand()
    terms = poly_expr.as_ordered_terms()
    lead_term = None
    max_deg = -1
    for term in terms:
        if term == 0:
            continue
        tdeg = 0 if term.is_Number else int(sum(term.as_powers_dict().values()))
        if tdeg > max_deg:
            max_deg = tdeg
            lead_term = term
    if lead_term is not None:
        lead_coef = lead_term.as_coeff_Mul()[0] if not lead_term.is_Number else lead_term
        if lead_coef != 0:
            poly_expr = sp.simplify(poly_expr / lead_coef)
    # Compute Gröbner basis
    try:
        gb = sp.groebner([sp.Poly(poly_expr, *sym_vars)], *sym_vars, order='grevlex')
        gb_list = list(gb)
    except Exception:
        gb_list = [poly_expr]
    return gb_list, poly_expr

# ----------------------------------------------------------------------
# Loop generators
# ----------------------------------------------------------------------
def trace_sum(rng):
    n = int(rng.integers(3, 80))
    i, s = 0, 0
    rows = [(i, s)]
    while i < n:
        s = s + i
        i = i + 1
        rows.append((i, s))
    return rows

def trace_ps2(rng):
    n = int(rng.integers(3, 40))
    i, s = 0, 0
    rows = [(i, s)]
    while i < n:
        s = s + i * i
        i = i + 1
        rows.append((i, s))
    return rows

def trace_ps3(rng):
    n = int(rng.integers(3, 25))
    i, s = 0, 0
    rows = [(i, s)]
    while i < n:
        s = s + i ** 3
        i = i + 1
        rows.append((i, s))
    return rows

def trace_egcd(rng):
    a0 = int(rng.integers(2, 500))
    b0 = int(rng.integers(2, 500))
    old_r, r = a0, b0
    old_s, s = 1, 0
    old_t, t = 0, 1
    rows = [(old_r, r, old_s, s, old_t, t, a0, b0)]
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
        rows.append((old_r, r, old_s, s, old_t, t, a0, b0))
    return rows

def trace_hash(rng):
    n = int(rng.integers(5, 40))
    M = 1_000_003
    i, h = 0, 0
    rows = [(i, h)]
    while i < n:
        h = (h * 31 + i) % M
        i = i + 1
        rows.append((i, h))
    return rows

BENCHMARKS = {
    "sum": {
        "vars": ["i", "s"],
        "trace_fn": trace_sum,
        "ground_truth": "i**2 - i - 2*s",
        "D_max": 2,
        "k_max": 6,
        "multi": False,
    },
    "ps2": {
        "vars": ["i", "s"],
        "trace_fn": trace_ps2,
        "ground_truth": "2*i**3 - 3*i**2 + i - 6*s",
        "D_max": 3,
        "k_max": 6,
        "multi": False,
    },
    "ps3": {
        "vars": ["i", "s"],
        "trace_fn": trace_ps3,
        "ground_truth": "i**4 - 2*i**3 + i**2 - 4*s",
        "D_max": 4,
        "k_max": 6,
        "multi": False,
    },
    "egcd": {
        "vars": ["old_r", "r", "old_s", "s", "old_t", "t", "a", "b"],
        "trace_fn": trace_egcd,
        "ground_truth": "old_r - old_s*a - old_t*b",
        "ground_truth_2": "r - s*a - t*b",
        "D_max": 2,
        "k_max": 4,
        "multi": True,
    },
    "hash": {
        "vars": ["i", "h"],
        "trace_fn": trace_hash,
        "ground_truth": None,
        "D_max": 4,
        "k_max": 6,
        "multi": False,
    },
}

# ----------------------------------------------------------------------
# Single run for a given benchmark and seed
# ----------------------------------------------------------------------
def run_single_benchmark(name, cfg, seed, N_TRAJECTORIES=60):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(N_TRAJECTORIES):
        rows.extend(cfg["trace_fn"](rng))
    int_rows = rows
    var_names = cfg["vars"]
    max_deg = cfg["D_max"]
    true_expr = sp.sympify(cfg["ground_truth"]) if cfg["ground_truth"] else None
    true_expr2 = sp.sympify(cfg.get("ground_truth_2")) if cfg.get("ground_truth_2") else None

    Phi, monomials, sym_vars, exponents = exact_monomial_matrix(var_names, max_deg, int_rows)
    ns = exact_nullspace(Phi)
    d = len(ns)
    M = Phi.cols
    N = Phi.rows

    Phi_np = np.array(Phi).astype(float)

    result = {"seed": seed, "benchmark": name, "null_dim": d, "M": M, "N": N}

    if d == 0:
        for meth in ["SR-GB+CSNP", "SINDy-null (exact)", "SINDy-ST", "OMP (exact)", "Dense SVD+GB (exact)"]:
            result[f"{meth}_exact"] = False
            result[f"{meth}_redundancy"] = 0
        return result

    # Convert nullspace to float array (M x d) for the baselines
    V_null_np = np.column_stack([np.array(col).flatten().astype(float) for col in ns])

    tau_resid = 1e-4

    # SR-GB+CSNP runs the real shipped pipeline end-to-end on the raw float
    # trace matrix: float SVD + column preconditioning + full-nullspace
    # deflation (multi-invariant) inside sr_gb(..., full_nullspace=True).
    # The baselines below keep their exact sympy integer nullspace, which is
    # the setting most favourable to them.
    X_float = np.array(int_rows, dtype=float)
    try:
        gb_srgb = sr_gb(X_float, var_names, degree=max_deg, sigma_estimate=0.0,
                        k_max=cfg["k_max"], full_nullspace=True)
    except Exception:
        gb_srgb = []

    if cfg.get("multi"):
        # egcd's nullspace is genuinely d=2 (the Bezout identity holds
        # separately for the old_r/old_s/old_t and r/s/t triples): a single
        # invariant can only cover one target, so "exact recovery" means the
        # returned generating set covers BOTH ground-truth relations. sr_gb's
        # full-nullspace deflation produces that set directly; each baseline
        # still gets its own deflate-and-resolve loop below.
        def sindy_null_solve(V_curr, iteration):
            return l1_fallback_on_nullspace(V_curr, Phi_np, tau_resid)

        def omp_solve(V_curr, iteration):
            return omp_on_nullspace(V_curr, Phi_np, tau_resid)

        def dense_solve(V_curr, iteration):
            return dense_svd_exact(V_curr, Phi_np, tau_resid)

        method_solvers = {
            "SINDy-null (exact)": sindy_null_solve,
            "OMP (exact)": omp_solve,
            "Dense SVD+GB (exact)": dense_solve,
        }
        targets = [t for t in (true_expr, true_expr2) if t is not None]

        matched_srgb = [exact_recovery(gb_srgb, t) for t in targets]
        result["SR-GB+CSNP_exact"] = all(matched_srgb) if targets else False
        result["SR-GB+CSNP_redundancy"] = len(gb_srgb)

        for method_name, solve_fn in method_solvers.items():
            try:
                _, gb_lists = deflate_search(solve_fn, V_null_np, d, monomials, sym_vars, sigma_estimate=0.0)
            except Exception:
                gb_lists = []
            matched = [False] * len(targets)
            for gb_list in gb_lists:
                for i, t in enumerate(targets):
                    if not matched[i] and exact_recovery(gb_list, t):
                        matched[i] = True
            result[f"{method_name}_exact"] = all(matched) if targets else False
            result[f"{method_name}_redundancy"] = sum(len(g) for g in gb_lists)

        # SINDy-ST regresses against implicit target columns of Phi rather
        # than reading the nullspace, so it has no deflation loop; but the
        # all-targets ensemble returns one candidate per target monomial,
        # so both Bezout identities can be covered by different targets and
        # the set is scored any-of per target like every other method.
        try:
            st_cands = run_sindy_st_ensemble(X_float, var_names, max_deg)
        except Exception:
            st_cands = []
        result["SINDy-ST_exact"] = (all(exact_recovery(st_cands, t) for t in targets)
                                    if targets else False)
        result["SINDy-ST_redundancy"] = len(st_cands)

        return result

    methods = {}

    # SR-GB+CSNP: the real pipeline's result, computed above via
    # sr_gb(..., full_nullspace=True) on the float trace matrix. An empty
    # basis still counts as a failure over the same denominator, not a
    # silent abstention.
    result["SR-GB+CSNP_exact"] = (exact_recovery(gb_srgb, true_expr)
                                  if true_expr is not None else False)
    result["SR-GB+CSNP_redundancy"] = len(gb_srgb)

    # ---- Baselines ----
    # A candidate of None is deliberately still recorded (not skipped): every
    # method's rate must be reported over the same fixed denominator (all
    # seeds), so a baseline that fails to produce any candidate counts as a
    # miss for that seed instead of being silently excluded from its own
    # average, which would otherwise selectively inflate methods that abstain
    # more often on the hard seeds.
    try:
        methods["SINDy-null (exact)"] = l1_fallback_on_nullspace(V_null_np, Phi_np, tau_resid)
    except Exception:
        methods["SINDy-null (exact)"] = None

    # OMP
    try:
        methods["OMP (exact)"] = omp_on_nullspace(V_null_np, Phi_np, tau_resid)
    except Exception:
        methods["OMP (exact)"] = None

    # Dense SVD
    try:
        methods["Dense SVD+GB (exact)"] = dense_svd_exact(V_null_np, Phi_np, tau_resid)
    except Exception:
        methods["Dense SVD+GB (exact)"] = None

    # For each method, build polynomial and check exact recovery
    for method_name, c in methods.items():
        gb_list, poly_expr = build_polynomial_from_coeff(c, monomials, sym_vars, sigma_estimate=0.0)
        if gb_list is None:
            result[f"{method_name}_exact"] = False
            result[f"{method_name}_redundancy"] = 0
            continue
        exact = False
        if true_expr is not None:
            exact = exact_recovery(gb_list, true_expr)
        result[f"{method_name}_exact"] = exact
        result[f"{method_name}_redundancy"] = len(gb_list)

    # SINDy-ST: the standard all-targets implicit STLSQ ensemble on the
    # float trace matrix (it regresses implicit target columns of Phi and
    # never reads the shared nullspace); scored any-of across candidates,
    # the same criterion every other benchmark applies to it.
    try:
        st_cands = run_sindy_st_ensemble(X_float, var_names, max_deg)
    except Exception:
        st_cands = []
    result["SINDy-ST_exact"] = (exact_recovery(st_cands, true_expr)
                                if true_expr is not None else False)
    result["SINDy-ST_redundancy"] = len(st_cands)

    return result

# ----------------------------------------------------------------------
# Main benchmark runner
# ----------------------------------------------------------------------
def run_benchmark(seeds=30, benchmarks=None, outdir="Results"):
    if benchmarks is None:
        benchmarks = list(BENCHMARKS.keys())
    os.makedirs(outdir, exist_ok=True)
    all_results = []
    for name in benchmarks:
        cfg = BENCHMARKS[name]
        print(f"\nRunning {name} ({seeds} seeds)...")
        for seed in range(seeds):
            try:
                res = run_single_benchmark(name, cfg, seed)
            except Exception as e:
                print(f"  seed {seed}: run_single_benchmark error: {e}")
                res = {"seed": seed, "benchmark": name, "null_dim": None, "M": None, "N": None}
                for meth in ["SR-GB+CSNP", "SINDy-null (exact)", "SINDy-ST", "OMP (exact)", "Dense SVD+GB (exact)"]:
                    res[f"{meth}_exact"] = False
                    res[f"{meth}_redundancy"] = 0
            all_results.append(res)
            if (seed+1) % 10 == 0:
                print(f"  seed {seed+1}/{seeds} done")
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(outdir, "loop_invariants_full.csv"), index=False)

    methods = ["SR-GB+CSNP", "SINDy-null (exact)", "SINDy-ST", "OMP (exact)", "Dense SVD+GB (exact)"]
    summary_rows = []
    for name in benchmarks:
        subset = df[df["benchmark"] == name]
        row = {"benchmark": name}
        for meth in methods:
            exact_col = f"{meth}_exact"
            if exact_col in subset.columns:
                rate = subset[exact_col].mean()
                k = subset[exact_col].sum()
                n = len(subset)
                ci = wilson_interval(k, n)
                row[f"{meth}_rate"] = rate
                row[f"{meth}_ci_low"] = ci[0]
                row[f"{meth}_ci_high"] = ci[1]
                red_col = f"{meth}_redundancy"
                if red_col in subset.columns:
                    row[f"{meth}_redundancy"] = subset[red_col].mean()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir, "loop_invariants_summary.csv"), index=False)

    print("\n" + "="*70)
    print("Loop Invariant Benchmark Summary (30 seeds)")
    print("="*70)
    print(summary_df[["benchmark"] + [f"{m}_rate" for m in methods]].to_string(index=False))
    return df, summary_df

# ----------------------------------------------------------------------
# Command line entry
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--outdir", type=str, default="Results")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer seeds for fast end-to-end verification")
    parser.add_argument("benchmarks", nargs="*", default=None,
                        help="Specific benchmarks to run (default: all)")
    args = parser.parse_args()
    seeds = 3 if args.quick else args.seeds
    run_benchmark(seeds=seeds, benchmarks=args.benchmarks if args.benchmarks else None,
                  outdir=args.outdir)