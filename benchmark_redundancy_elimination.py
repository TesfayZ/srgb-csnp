#!/usr/bin/env python3
"""
benchmark_redundancy_elimination.py – Reproduces Table 3.
SINDy-AD is omitted for static varieties because it requires time-series data.
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from sympy import parse_expr, groebner, Poly, symbols, Rational, simplify, cancel, lambdify
import warnings
warnings.filterwarnings('ignore')

from sr_gb import sr_gb, build_monomial_library, estimate_rank, snap_round, exact_recovery, reduce_to_minimal_generator
from sindy_baselines import sindy_nullspace, sindy_st, sindy_st_ensemble
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


def generate_sphere(N=5000, sigma=0.0, seed=42):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2*np.pi, N)
    phi = np.arccos(2*np.random.uniform(0, 1, N) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    if sigma > 0:
        x += np.random.normal(0, sigma, N)
        y += np.random.normal(0, sigma, N)
        z += np.random.normal(0, sigma, N)
    return np.column_stack([x, y, z])


def generate_cubic(N=5000, sigma=0.0, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-2, 1, N)
    x = np.minimum(x, 0.99)
    sqrt_term = np.sqrt(x**2 * (1 - x) + 1e-12)
    y1 = -x + sqrt_term
    y2 = -x - sqrt_term
    mask = np.random.choice([0, 1], N)
    y = np.where(mask, y1, y2)
    data = np.column_stack([x, y])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data


def dense_svd_gb(data, var_names, degree, sigma_estimate=0.0):
    """Naive ablation: take the single dominant (smallest-singular-value)
    SVD null vector, snap-round it, factor out any redundant/divisible
    components (the "algebraic minimality" step -- this is what lets a
    degree-lifted multiple like y*(x^2+y^2-1) reduce to the true generator
    x^2+y^2-1, per Adaptive-CSNP.tex's own description of this baseline),
    then take the Groebner basis. No CSNP combinatorial search over
    supports: this ablation isolates what factorization alone buys you."""
    sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data)
    U, s, Vt = svd(Phi, full_matrices=False)
    c = Vt[-1, :]
    c[np.abs(c) < 1e-3] = 0.0
    rounded = snap_round(c, sigma_estimate)
    poly_expr = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
    if poly_expr == 0:
        return [], None
    poly_expr = simplify(poly_expr.expand())
    sym_vars = symbols(var_names)
    poly_expr = reduce_to_minimal_generator(poly_expr, data, sigma_estimate, sym_vars=sym_vars)
    if poly_expr == 0:
        return [], None
    gb = groebner([Poly(poly_expr, *sym_vars)], *sym_vars, order='grevlex')
    return list(gb), poly_expr


def reconstruction_error(poly_expr, data, var_names):
    if poly_expr is None or poly_expr == 0:
        return 1.0
    sym_vars = symbols(var_names)
    f = lambdify(sym_vars, poly_expr, modules='numpy')
    try:
        vals = f(*[data[:, i] for i in range(len(var_names))])
        return float(np.mean(np.abs(vals)))
    except Exception:
        return 1.0


def compute_fdr(candidates, true_expr):
    if not candidates:
        return 1.0
    redundant = 0
    for p in candidates:
        try:
            ratio = cancel(p / true_expr)
            if ratio.is_number or ratio.is_constant():
                continue
        except:
            pass
        redundant += 1
    return redundant / len(candidates)


def run_benchmarks(n_seeds=30, N=5000, sigma=0.0):
    problems = [
        ("Circle", ["x", "y"], 2, "x**2 + y**2 - 1", generate_circle),
        ("Sphere", ["x", "y", "z"], 2, "x**2 + y**2 + z**2 - 1", generate_sphere),
        ("Cubic", ["x", "y"], 3, "x**3 + 2*x*y + y**2", generate_cubic),
    ]

    rows = []
    for name, var_names, degree, expr_str, gen_func in problems:
        true_expr = parse_expr(expr_str)
        for seed in range(n_seeds):
            data = gen_func(N=N, sigma=sigma, seed=seed)

            # ---- SINDy-null ----
            try:
                null_cands = sindy_nullspace(data, var_names, degree, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {name} seed={seed}: sindy_nullspace error: {e}")
                null_cands = []
            null_red = len(null_cands)
            null_poly = null_cands[0] if null_cands else None
            null_rec = reconstruction_error(null_poly, data, var_names) if null_poly else 1.0
            null_fdr = compute_fdr(null_cands, true_expr) if null_cands else 1.0

            # ---- SINDy-ST (ensemble) ----
            try:
                st_cands = sindy_st_ensemble(data, var_names, degree, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {name} seed={seed}: sindy_st_ensemble error: {e}")
                st_cands = []
            st_red = len(st_cands)
            st_poly = st_cands[0] if st_cands else None
            st_rec = reconstruction_error(st_poly, data, var_names) if st_poly else 1.0
            st_fdr = compute_fdr(st_cands, true_expr) if st_cands else 1.0

            # ---- SINDy-AD: NOT APPLICABLE (static data) ----
            ad_red = float('nan')
            ad_rec = float('nan')
            ad_fdr = float('nan')

            # ---- Dense SVD+GB ----
            try:
                gb_dense, poly_dense = dense_svd_gb(data, var_names, degree, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {name} seed={seed}: dense_svd_gb error: {e}")
                gb_dense, poly_dense = [], None
            dense_red = len(gb_dense)
            dense_rec = reconstruction_error(poly_dense, data, var_names) if poly_dense else 1.0
            dense_fdr = compute_fdr(gb_dense, true_expr) if gb_dense else 1.0

            # ---- SR-GB+CSNP (ours) – now adaptive ----
            # We call sr_gb with degree=None, D_max=degree (so it searches up to the true degree)
            try:
                gb_srgb = sr_gb(data, var_names, degree=None, D_max=degree, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {name} seed={seed}: sr_gb error: {e}")
                gb_srgb = []
            srgb_red = len(gb_srgb)
            if gb_srgb:
                poly_srgb = gb_srgb[0].as_expr() if hasattr(gb_srgb[0], 'as_expr') else gb_srgb[0]
                srgb_rec = reconstruction_error(poly_srgb, data, var_names)
                srgb_fdr = compute_fdr(gb_srgb, true_expr)
            else:
                srgb_rec = 1.0
                srgb_fdr = 1.0

            rows.append({
                "problem": name, "seed": seed,
                "null_red": null_red, "null_rec": null_rec, "null_fdr": null_fdr,
                "st_red": st_red, "st_rec": st_rec, "st_fdr": st_fdr,
                "ad_red": ad_red, "ad_rec": ad_rec, "ad_fdr": ad_fdr,
                "dense_red": dense_red, "dense_rec": dense_rec, "dense_fdr": dense_fdr,
                "srgb_red": srgb_red, "srgb_rec": srgb_rec, "srgb_fdr": srgb_fdr,
            })
            if (seed + 1) % 10 == 0:
                print(f"{name}: seed {seed+1}/{n_seeds} done")

    df = pd.DataFrame(rows)
    summary = df.groupby("problem").agg(
        null_red_mean=("null_red", "mean"), null_red_std=("null_red", "std"),
        st_red_mean=("st_red", "mean"), st_red_std=("st_red", "std"),
        ad_red_mean=("ad_red", "mean"), ad_red_std=("ad_red", "std"),
        dense_red_mean=("dense_red", "mean"), dense_red_std=("dense_red", "std"),
        srgb_red_mean=("srgb_red", "mean"), srgb_red_std=("srgb_red", "std"),
        null_rec_mean=("null_rec", "mean"),
        st_rec_mean=("st_rec", "mean"),
        ad_rec_mean=("ad_rec", "mean"),
        dense_rec_mean=("dense_rec", "mean"),
        srgb_rec_mean=("srgb_rec", "mean"),
        null_fdr_mean=("null_fdr", "mean"),
        st_fdr_mean=("st_fdr", "mean"),
        ad_fdr_mean=("ad_fdr", "mean"),
        dense_fdr_mean=("dense_fdr", "mean"),
        srgb_fdr_mean=("srgb_fdr", "mean"),
    ).reset_index()

    overall_rec = {
        "null": summary["null_rec_mean"].mean(),
        "st": summary["st_rec_mean"].mean(),
        "ad": summary["ad_rec_mean"].mean(),
        "dense": summary["dense_rec_mean"].mean(),
        "srgb": summary["srgb_rec_mean"].mean(),
    }
    overall_fdr = {
        "null": summary["null_fdr_mean"].mean(),
        "st": summary["st_fdr_mean"].mean(),
        "ad": summary["ad_fdr_mean"].mean(),
        "dense": summary["dense_fdr_mean"].mean(),
        "srgb": summary["srgb_fdr_mean"].mean(),
    }

    print("\n" + "=" * 80)
    print("Table 3: Redundancy Elimination (30 seeds, σ=0.0)")
    print("Note: SINDy-AD is omitted (N/A) because these are static varieties.")
    print("=" * 80)
    print(f"{'Method':<18} {'Red. Circle':<15} {'Red. Sphere':<15} {'Red. Cubic':<15} {'Rec. error':<12} {'FDR':<10}")
    print("-" * 80)

    def red_str(problem, col_mean, col_std):
        mean = summary.loc[summary["problem"] == problem, col_mean].values[0]
        std = summary.loc[summary["problem"] == problem, col_std].values[0]
        return f"{mean:.1f} ± {std:.1f}"

    for label, prefix in [("SINDy-null", "null"), ("SINDy-ST", "st"),
                          ("Dense SVD+GB", "dense"), ("SR-GB+CSNP (ours)", "srgb")]:
        c_red = red_str("Circle", f"{prefix}_red_mean", f"{prefix}_red_std")
        s_red = red_str("Sphere", f"{prefix}_red_mean", f"{prefix}_red_std")
        cub_red = red_str("Cubic", f"{prefix}_red_mean", f"{prefix}_red_std")
        rec = overall_rec[prefix]
        fdr = overall_fdr[prefix]
        print(f"{label:<18} {c_red:<15} {s_red:<15} {cub_red:<15} {rec:<12.2f} {fdr:<10.2f}")

    print("=" * 80)
    df.to_csv("Results/redundancy_elimination_full.csv", index=False)
    summary.to_csv("Results/redundancy_elimination_summary.csv", index=False)
    return summary, df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N as full run")
    args = parser.parse_args()
    if args.quick:
        run_benchmarks(n_seeds=2)
    else:
        run_benchmarks()