#!/usr/bin/env python3
"""
benchmark_runtime_breakdown.py — Measure wall-clock time per pipeline stage.

"""

import time
import numpy as np
import pandas as pd
from scipy.linalg import svd
from itertools import combinations_with_replacement, combinations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sr_gb import build_monomial_library, snap_round, estimate_rank, _l1_nullspace_fallback
from sympy import groebner, Poly, symbols, Rational
import warnings
warnings.filterwarnings('ignore')


# ── Data generators ───────────────────────────────────────────────────────────

def generate_circle(N=5000, sigma=0.01, seed=0):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    x = np.cos(theta) + np.random.normal(0, sigma, N)
    y = np.sin(theta) + np.random.normal(0, sigma, N)
    return np.column_stack([x, y])


def generate_mujoco(N=5000, sigma=0.01, seed=0):
    """Double pendulum q1=q2 constraint."""
    np.random.seed(seed)
    q = np.random.uniform(-np.pi, np.pi, N)
    data = np.column_stack([q + np.random.normal(0, sigma, N),
                            q + np.random.normal(0, sigma, N)])
    return data


def generate_4sphere(N=5000, sigma=0.0, seed=0):
    """4-sphere data for L1 fallback timing demo."""
    np.random.seed(seed)
    phi_a = np.random.uniform(0, np.pi, N)
    phi_b = np.random.uniform(0, np.pi, N)
    theta  = np.random.uniform(0, 2*np.pi, N)
    x = np.sin(phi_a)*np.sin(phi_b)*np.cos(theta)
    y = np.sin(phi_a)*np.sin(phi_b)*np.sin(theta)
    z = np.sin(phi_a)*np.cos(phi_b)
    w = np.cos(phi_a)
    data = np.column_stack([x, y, z, w])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data


# ── Timing function ───────────────────────────────────────────────────────────

def time_pipeline(data, var_names, degree, min_degree=0, sigma=0.01,
                  label="", use_l1=False):
    N = data.shape[0]
    timings = {}

    # Stage 1: library build
    t0 = time.perf_counter()
    sym_vars, monomials, evaluate = build_monomial_library(
        var_names, degree, min_degree, scale=False)
    timings["library_build"] = time.perf_counter() - t0

    # Stage 2: design matrix
    t0 = time.perf_counter()
    Phi, _, _ = evaluate(data)
    timings["phi_eval"] = time.perf_counter() - t0
    M = Phi.shape[1]

    # Stage 3: SVD
    t0 = time.perf_counter()
    U, s, Vt = svd(Phi, full_matrices=False)
    timings["svd"] = time.perf_counter() - t0

    # Stage 4: rank + CSNP or L1
    r = estimate_rank(s, sigma_estimate=sigma, N=N)
    d = M - r

    if d <= 1 or (not use_l1 and d <= 1):
        timings["csnp_or_l1"] = 0.0
        c = Vt[-1, :]
    elif use_l1:
        V_null = Vt[r:, :].T
        tau_resid = max(1e-3, sigma * np.sqrt(N) * 3.0)
        t0 = time.perf_counter()
        c = _l1_nullspace_fallback(V_null, Phi, tau_resid)
        timings["csnp_or_l1"] = time.perf_counter() - t0
        if c is None:
            c = Vt[-1, :]
    else:
        # d=2 fast mode CSNP
        V_null = Vt[r:, :].T
        tau_resid = max(1e-3, sigma * np.sqrt(N) * 3.0)
        t0 = time.perf_counter()
        from math import comb as _comb
        best_c = None; best_k = np.inf; best_resid = np.inf
        for k in range(1, min(7, M+1)):
            if _comb(M, k) * d**2 > 1_000_000:
                break
            for S in combinations(range(M), k):
                notS = [i for i in range(M) if i not in S]
                if not notS:
                    continue
                A = V_null[notS, :]
                Ua, sa, Vta = svd(A, full_matrices=False)
                null_dim_A = int(np.sum(sa < 1e-8 * (sa[0]+1e-30)))
                if null_dim_A == 1:
                    alpha = Vta[-1, :]
                    c_cand = V_null @ alpha
                    norm = np.linalg.norm(c_cand)
                    if norm < 1e-10: continue
                    c_cand /= norm
                    resid = np.linalg.norm(Phi @ c_cand)
                    if resid < tau_resid and k < best_k:
                        best_k = k; best_c = c_cand; best_resid = resid
            if best_c is not None:
                break
        timings["csnp_or_l1"] = time.perf_counter() - t0
        c = best_c if best_c is not None else Vt[-1, :]

    # Stage 5: snap-round
    t0 = time.perf_counter()
    c[np.abs(c) < 1e-3] = 0.0
    rounded = snap_round(c, sigma_estimate=sigma)
    timings["snap_round"] = time.perf_counter() - t0

    # Stage 6: Gröbner
    t0 = time.perf_counter()
    c_rat = [Rational(v) for v in rounded]
    poly = sum(coef * mon for coef, mon in zip(c_rat, monomials) if coef != 0)
    if poly != 0:
        gb = groebner([Poly(poly, *sym_vars)], *sym_vars, order='grevlex')
    timings["groebner"] = time.perf_counter() - t0

    timings["total"] = sum(v for k, v in timings.items())
    timings["problem"] = label
    timings["M"] = M
    timings["d"] = d
    timings["N"] = N
    return timings


def main(n_repeats=3):
    all_results = []

    for rep in range(n_repeats):
        # ── Circle (d=1) ──────────────────────────────────────────────────
        data = generate_circle(N=5000, sigma=0.01, seed=rep)
        t = time_pipeline(data, ["x", "y"], degree=2, min_degree=0,
                          sigma=0.01, label="Circle (d=1, no CSNP)")
        all_results.append(t)

        # ── MuJoCo q1=q2 (d=1) ───────────────────────────────────────────
        data2 = generate_mujoco(N=5000, sigma=0.01, seed=rep)
        t2 = time_pipeline(data2, ["q1", "q2"], degree=1, min_degree=0,
                           sigma=0.01, label="MuJoCo q1=q2 (d=1, no CSNP)")
        all_results.append(t2)

        # ── 4-sphere degree-3 (L1 fallback, M=35) ────────────────────────
        data3 = generate_4sphere(N=5000, sigma=0.0, seed=rep)
        t3 = time_pipeline(data3, ["x", "y", "z", "w"], degree=3, min_degree=0,
                           sigma=0.0, label="4-sphere degree-3 (L1 fallback)",
                           use_l1=True)
        all_results.append(t3)

    df = pd.DataFrame(all_results)
    summary = df.groupby("problem").agg(
        M=("M", "first"),
        d=("d", "first"),
        N=("N", "first"),
        library_build=("library_build", "mean"),
        phi_eval=("phi_eval", "mean"),
        svd=("svd", "mean"),
        csnp_or_l1=("csnp_or_l1", "mean"),
        snap_round=("snap_round", "mean"),
        groebner=("groebner", "mean"),
        total=("total", "mean"),
    ).reset_index()

    print(f"\nRuntime breakdown (seconds, mean over {n_repeats} runs)")
    print(summary.round(5).to_string(index=False))
    print()

    # Stage-by-stage % for each problem
    for _, row in summary.iterrows():
        total = row["total"]
        if total < 1e-9: continue
        print(f"\n── {row['problem']} (M={int(row['M'])}, d={int(row['d'])}) ──")
        for stage in ["library_build","phi_eval","svd","csnp_or_l1","snap_round","groebner"]:
            pct = 100*row[stage]/total
            print(f"  {stage:18s}: {row[stage]:.5f}s  ({pct:.1f}%)")
        print(f"  {'TOTAL':18s}: {total:.5f}s")

    df.to_csv("Results/runtime_breakdown.csv", index=False)
    summary.to_csv("Results/runtime_breakdown_summary.csv", index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fewer repeats for fast end-to-end verification")
    args = parser.parse_args()
    main(n_repeats=1 if args.quick else 3)
