#!/usr/bin/env python3
"""
ablation_sparsity_vs_rationality_harmonic.py - sparsity-first vs
rationality-first selection on the fixed-dt harmonic oscillator's
transition-pair dictionary.

Isolates the selection criterion (sparsity vs. rationality) as the only
variable in SR-GB+CSNP's own search. Both arms search the IDENTICAL candidate pool: the same monomial
library, the same column-preconditioned SVD nullspace front end the rest of
the pipeline uses (`_column_preconditioned_svd` / `estimate_rank` /
`_orthonormal_null_basis`), the same direct support enumeration over that
nullspace, and the same `snap_round` post-processing. The two arms differ
only in the final selection key:

  - sparsity-first: smallest active support size k_act, tie-broken by
    degree-sum then coefficient-sum of the rounded coefficients.
  - rationality-first: smallest `rationality_cost` (sr_gb.py's bit-cost
    metric), tie-broken by k_act -- the same (R, k) lexicographic order
    `progressive_nullspace_search` uses in the main pipeline.

On the exact rotation flow at fixed dt, the transition-pair dictionary
(x_t, v_t, x_next, v_next) contains a dt-specific spurious relation that is
sparser than the true energy invariant x_t^2+v_t^2-x_next^2-v_next^2 but has
no simple rational coefficients. Sparsity-first selection prefers the
spurious relation; rationality-first selection rejects it. This does not
compare against other methods (e.g. SINDy-FD, which searches a different,
restricted dictionary and is reported by benchmark_dt_discriminator.py
instead) -- it holds the search algorithm fixed and varies only what "best
candidate" means.
"""

import argparse
import os
import time
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
from sympy import Rational, simplify, parse_expr

from sr_gb import (build_monomial_library, estimate_rank, rationality_cost,
                    snap_round, exact_recovery,
                    _column_preconditioned_svd, _orthonormal_null_basis)
from utils_stats import wilson_interval

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VAR_NAMES = ["x_t", "v_t", "x_next", "v_next"]
TRUE_INVARIANT = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")

DT_VALUES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
K_MAX = 6
ENUM_BUDGET = 1_000_000


def generate_harmonic_pairs(N, dt, sigma, seed):
    """Exact rotation flow (same construction as
    benchmark_harmonic_oscillator_vs_sindy.generate_harmonic_trajectory's
    pairs, and test_rationality_resolves_dt_ambiguity.generate_harmonic_pairs)."""
    rng = np.random.RandomState(seed)
    x_t = rng.uniform(-1, 1, N)
    v_t = rng.uniform(-1, 1, N)
    c, s = np.cos(dt), np.sin(dt)
    x_n = c * x_t + s * v_t
    v_n = -s * x_t + c * v_t
    data = np.column_stack([x_t, v_t, x_n, v_n])
    if sigma > 0:
        data = data + rng.normal(0, sigma, data.shape)
    return data


def _support_info(rounded, monomials, eps=1e-3):
    """(k_act, deg_sum, coef_sum) of the rounded coefficient vector: active
    support size, sum of active monomials' total degree, sum of |coefficient|."""
    k = 0
    deg_sum = 0
    coef_sum = 0.0
    for r, mon in zip(rounded, monomials):
        if abs(r) > eps:
            k += 1
            deg_sum += 0 if mon == 1 else sum(mon.as_powers_dict().values())
            coef_sum += abs(float(r))
    return k, deg_sum, coef_sum


def _enumerate_candidates(Phi, V_null, tau_resid, geometric_tolerance, k_max=K_MAX):
    """Direct support enumeration over the nullspace basis: for each support
    size k (smallest first), test every size-k complement subspace for
    rank-deficiency-by-one against V_null, extract the resulting circuit
    direction, and keep it if its data residual clears tau_resid. Sweeping k
    from 1 upward, rather than fixing it in advance, is what lets the
    sparsity-first arm find its own minimal feasible support rather than
    having it handed to it."""
    M = Phi.shape[1]
    d = V_null.shape[1]
    candidates = []
    for k in range(1, min(k_max, M) + 1):
        if comb(M, k) * d ** 2 > ENUM_BUDGET:
            break
        for S in combinations(range(M), k):
            notS = [i for i in range(M) if i not in S]
            if len(notS) < d:
                continue
            A = V_null[notS, :]
            Ua, sa, Vta = np.linalg.svd(A, full_matrices=False)
            if len(sa) < d or sa[0] <= 0:
                continue
            if sa[-1] / sa[0] > geometric_tolerance:
                continue
            alpha = Vta[-1, :]
            c = V_null @ alpha
            norm = np.linalg.norm(c)
            if norm < 1e-30:
                continue
            c = c / norm
            resid = np.linalg.norm(Phi @ c)
            if resid < tau_resid:
                candidates.append(c)
    return candidates


def _select(candidates, monomials, sigma_estimate, max_denom=16):
    """Score every candidate once, then return (sparsity_winner_poly,
    rationality_winner_poly) as sympy expressions, or (None, None) for an
    arm with no feasible candidate."""
    scored = []
    for c in candidates:
        r_score = rationality_cost(c, max_denom=max_denom, eps=1e-3)
        rounded = snap_round(c, sigma_estimate=sigma_estimate, max_denom=max_denom)
        k_act, deg_act, coef_act = _support_info(rounded, monomials)
        poly = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
        scored.append({
            "sparsity_key": (k_act, deg_act, coef_act),
            "rationality_key": (r_score, k_act, deg_act),
            "poly": poly,
        })
    if not scored:
        return None, None
    sparsity_winner = min(scored, key=lambda s: s["sparsity_key"])["poly"]
    rationality_winner = min(scored, key=lambda s: s["rationality_key"])["poly"]
    return sparsity_winner, rationality_winner


def run_trial(dt, seed, N=5000, sigma=0.0):
    data = generate_harmonic_pairs(N, dt, sigma, seed)
    sym_vars, monomials, evaluate = build_monomial_library(
        VAR_NAMES, max_degree=2, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data)
    Nrows, M = Phi.shape

    s, Vt = _column_preconditioned_svd(Phi)
    r = estimate_rank(s, sigma_estimate=sigma, N=Nrows)
    d = M - r
    if d < 1:
        return False, False
    V_null = _orthonormal_null_basis(Vt, M, d)

    tau_resid = max(1e-4, sigma * np.sqrt(Nrows) * 3.0)
    geometric_tolerance = max(1e-8, 2.0 * sigma / np.sqrt(Nrows))

    candidates = _enumerate_candidates(Phi, V_null, tau_resid, geometric_tolerance)
    sparsity_poly, rationality_poly = _select(candidates, monomials, sigma)

    sparsity_ok = exact_recovery([sparsity_poly] if sparsity_poly is not None else [], TRUE_INVARIANT)
    rationality_ok = exact_recovery([rationality_poly] if rationality_poly is not None else [], TRUE_INVARIANT)
    return sparsity_ok, rationality_ok


def run_experiment(dt_values, n_seeds=30, N=5000, sigma=0.0):
    rows = []
    print(f"{'dt':>6} | {'seeds':>5} | {'sparsity_rate':>13} | {'rationality_rate':>16}")
    print("-" * 55)
    for dt in dt_values:
        t0 = time.time()
        sp_hits, ra_hits = 0, 0
        for seed in range(n_seeds):
            sp_ok, ra_ok = run_trial(dt, seed, N=N, sigma=sigma)
            sp_hits += int(sp_ok)
            ra_hits += int(ra_ok)
            rows.append({"dt": dt, "seed": seed, "sparsity_exact": sp_ok,
                         "rationality_exact": ra_ok})
        print(f"{dt:>6.2f} | {n_seeds:>5} | {sp_hits/n_seeds:>13.1%} | "
              f"{ra_hits/n_seeds:>16.1%}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    summary = df.groupby("dt").agg(
        sparsity_rate=("sparsity_exact", "mean"),
        rationality_rate=("rationality_exact", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()
    for idx, row in summary.iterrows():
        k_sp = int(row["sparsity_rate"] * row["n_seeds"])
        k_ra = int(row["rationality_rate"] * row["n_seeds"])
        n = int(row["n_seeds"])
        summary.loc[idx, "sparsity_ci_low"], summary.loc[idx, "sparsity_ci_high"] = wilson_interval(k_sp, n)
        summary.loc[idx, "rationality_ci_low"], summary.loc[idx, "rationality_ci_high"] = wilson_interval(k_ra, n)

    df.to_csv(f"{RESULTS_DIR}/ablation_sparsity_vs_rationality_harmonic_full.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/ablation_sparsity_vs_rationality_harmonic_summary.csv", index=False)

    overall_sp = df["sparsity_exact"].mean()
    overall_ra = df["rationality_exact"].mean()
    k_sp_all, n_all = int(df["sparsity_exact"].sum()), len(df)
    k_ra_all = int(df["rationality_exact"].sum())
    ci_sp = wilson_interval(k_sp_all, n_all)
    ci_ra = wilson_interval(k_ra_all, n_all)
    print("\n" + "=" * 55)
    print(f"Overall (pooled across dt, n={n_all}):")
    print(f"  sparsity-first:     {overall_sp:.1%}  95% CI [{ci_sp[0]:.1%}, {ci_sp[1]:.1%}]")
    print(f"  rationality-first:  {overall_ra:.1%}  95% CI [{ci_ra[0]:.1%}, {ci_ra[1]:.1%}]")
    print("=" * 55)
    return df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count and dt grid for a fast smoke check")
    args = parser.parse_args()
    if args.quick:
        run_experiment(DT_VALUES[:3], n_seeds=3, N=5000, sigma=0.0)
    else:
        run_experiment(DT_VALUES, n_seeds=30, N=5000, sigma=0.0)
