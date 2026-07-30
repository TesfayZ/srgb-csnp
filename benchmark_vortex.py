#!/usr/bin/env python3
"""
benchmark_vortex.py - Multi-invariant recovery on planar point-vortex flows.

N identical unit-circulation point vortices evolve under the Kirchhoff
equations (see vortex_data.py). The motion is Hamiltonian with a
logarithmic (non-polynomial) energy, but it carries three polynomial first
integrals from translational and rotational symmetry:

    P = sum_j x_j                       (linear impulse, degree 1)
    Q = sum_j y_j                       (linear impulse, degree 1)
    I = sum_j (x_j^2 + y_j^2) - I0      (angular impulse, degree 2)

Each initial condition is projected onto the shared level set P=Q=0, I=I0
before integration, so pooling trajectories from different initial
conditions leaves exactly the ideal <P, Q, I> vanishing on the whole pool
(pooling independent-value trajectories would cancel the invariants). This
is the physical, dense multi-invariant test for the full-nullspace
deflation path: a degree-2 lift over 2n state variables carries all three
generators in one nullspace, so recovering them means separating three
low-degree invariants from a high-dimensional numerical nullspace and
returning the minimal generating set, not a redundant superset.

RECOVERY CHECK. The reduced Groebner basis rewrites I into its normal form
modulo P and Q (grevlex eliminates x_1, y_1), so a per-generator
scalar-multiple test against the raw target I = sum r^2 - I0 spuriously
fails even when the ideal is exactly right. Recovery is therefore scored by
ideal equality: the recovered basis and <P, Q, I> must reduce each other's
generators to zero. Per-invariant columns (P, Q, I recovered) report
one-directional membership (does the true generator reduce to zero modulo
the recovered basis) for a finer-grained view.

Saves: Results/vortex_results.csv, Results/vortex_summary.csv
"""

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sympy as sp

from sr_gb import sr_gb, build_monomial_library
from sindy_baselines import rref_nullspace
from vortex_data import load_or_simulate, true_invariants
from utils_stats import wilson_interval

# (cache name, number of vortices) in ascending difficulty.
SYSTEMS = [("vortex3", 3), ("vortex4", 4), ("vortex5", 5)]
DEGREE = 2
N = 3000


def _reduces_to_zero(basis_G, expr):
    try:
        return basis_G.reduce(sp.expand(expr))[1] == 0
    except Exception:
        return False


def _score(recovered, true_exprs, syms):
    """Return (ideal_match, per-true-generator membership list).

    ideal_match is True iff <recovered> == <true>: every true generator
    reduces to zero modulo the recovered basis AND vice versa.

    NOTE ON WHAT THIS DOES AND DOES NOT TEST: whenever `recovered` spans the
    full nullspace of Phi (any complete basis of it, not just a minimal
    generating set), this is close to tautologically True, since the
    nullspace IS exactly the degree<=2 graded piece of <P,Q,I> by
    construction of the data. It is the right check for CSNP's own output,
    which is a *reduced Groebner basis* (a minimal, canonical set) computed
    from that nullspace. It is the WRONG check for a baseline like RREF that
    returns the raw, non-minimal, un-Gr\"obner-reduced nullspace basis
    itself (one row per nullspace dimension): scoring that basis this way
    mostly re-confirms the shared SVD front end estimated the nullspace
    correctly, not that the individual disambiguated rows are recognisable
    as P, Q, or I. See `_direct_match`/`rref_*_direct` below for the
    per-generator test that actually exercises RREF's disambiguation claim.
    """
    if not recovered:
        return False, [False] * len(true_exprs)
    try:
        Grec = sp.groebner([sp.expand(p) for p in recovered], *syms,
                           order="grevlex", domain="QQ")
        Gtru = sp.groebner([sp.expand(e) for e in true_exprs], *syms,
                           order="grevlex", domain="QQ")
    except Exception:
        return False, [False] * len(true_exprs)
    per_true = [_reduces_to_zero(Grec, e) for e in true_exprs]
    rec_in_true = all(_reduces_to_zero(Gtru, g) for g in Grec.exprs)
    ideal_match = all(per_true) and rec_in_true
    return ideal_match, per_true


def _direct_match(poly, target_expr):
    """Raw scalar-multiple match: poly == c*target_expr for some nonzero
    constant c, with NO Groebner reduction on either side. This is the
    literal test of whether a single un-reduced candidate (e.g. one RREF
    row) already IS the named invariant, matching
    benchmark_deflation_multi_invariant.py's exact_match and matching what
    Oellerich & Emelianenko's row-echelon output actually delivers: d
    individually-presented rows, not a further-reduced minimal basis."""
    if poly is None or poly == 0:
        return False
    try:
        ratio = sp.cancel(poly / target_expr)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return False


def run_one(name, n, seed, N=N, degree=DEGREE, sigma=0.0):
    df = load_or_simulate(name, n)
    xs = [f"x{j + 1}" for j in range(n)]
    ys = [f"y{j + 1}" for j in range(n)]
    var_names = xs + ys
    M = df[var_names].to_numpy()

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(M), size=min(N, len(M)), replace=False)
    data = M[idx]
    if sigma > 0:
        data = data + rng.normal(0, sigma, data.shape)

    syms = sp.symbols(var_names)
    true_exprs = [e for _, e in true_invariants(n)]

    t0 = time.time()
    try:
        gb = sr_gb(data, var_names, degree=degree, sigma_estimate=sigma,
                   full_nullspace=True)
    except Exception:
        gb = []
    runtime = time.time() - t0
    polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]
    ideal_match, per_true = _score(polys, true_exprs, syms)

    # RREF disambiguation (Oellerich & Emelianenko 2024) on the IDENTICAL
    # degree-2 library over the pooled state, built the same way sr_gb()
    # builds it internally (build_monomial_library(..., min_degree=0,
    # scale=False), same noise_sigma-corrected evaluate()), so it sees the
    # same nullspace CSNP's full-nullspace deflation consumes. P and Q are
    # linear (Oellerich & Emelianenko's own tested case); I is a dense,
    # degree-2, (2n+1)-term generator sharing monomial support with P^2, Q^2,
    # and PQ -- the sharper test of whether RREF's canonicalisation survives
    # outside the linear case it was demonstrated on.
    try:
        sym_vars, monomials, evaluate = build_monomial_library(
            var_names, degree, min_degree=0, scale=False)
        Phi, _, _ = evaluate(data, noise_sigma=sigma)
        rref_polys = rref_nullspace(None, var_names, degree, sigma_estimate=sigma,
                                    Phi=Phi, monomials=monomials)
    except Exception:
        rref_polys = []
    # Secondary diagnostic only (see _score's docstring): near-tautological
    # once the nullspace estimate is numerically correct, since ANY complete
    # basis of it spans <P,Q,I>'s graded piece. Kept to distinguish "RREF's
    # individual rows aren't recognisable as P/Q/I" from "the shared SVD
    # front end got the nullspace itself wrong."
    rref_ideal_match, _ = _score(rref_polys, true_exprs, syms)
    # Primary signal: does any raw RREF row, taken on its own with no further
    # reduction, literally equal a scalar multiple of P, Q, or I.
    rref_P_direct = any(_direct_match(p, true_exprs[0]) for p in rref_polys)
    rref_Q_direct = any(_direct_match(p, true_exprs[1]) for p in rref_polys)
    rref_I_direct = any(_direct_match(p, true_exprs[2]) for p in rref_polys)
    rref_all_direct = rref_P_direct and rref_Q_direct and rref_I_direct

    return {
        "system": name, "n_vortices": n, "seed": seed, "sigma": sigma,
        "degree": degree, "n_generators": len(polys),
        "P_recovered": per_true[0], "Q_recovered": per_true[1],
        "I_recovered": per_true[2], "ideal_match": ideal_match,
        "runtime_s": runtime,
        "rref_n_generators": len(rref_polys),
        "rref_ideal_match": rref_ideal_match,
        "rref_P_direct": rref_P_direct, "rref_Q_direct": rref_Q_direct,
        "rref_I_direct": rref_I_direct, "rref_all_direct": rref_all_direct,
    }


def benchmark(n_seeds=10, degree=DEGREE, sigma=0.0, N=N):
    rows = []
    for name, n in SYSTEMS:
        print(f"\n{name} (n={n}, degree={degree}, sigma={sigma})")
        for seed in range(n_seeds):
            row = run_one(name, n, seed, N=N, degree=degree, sigma=sigma)
            rows.append(row)
            print(f"  seed={seed:2d}: ideal_match={row['ideal_match']} "
                  f"(P={row['P_recovered']} Q={row['Q_recovered']} "
                  f"I={row['I_recovered']}) ngen={row['n_generators']} "
                  f"t={row['runtime_s']:.1f}s | RREF direct-match="
                  f"{row['rref_all_direct']} (P={row['rref_P_direct']} "
                  f"Q={row['rref_Q_direct']} I={row['rref_I_direct']}) "
                  f"ngen={row['rref_n_generators']} "
                  f"[ideal_match={row['rref_ideal_match']}]")

    df = pd.DataFrame(rows)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/vortex_results.csv", index=False)

    summary_rows = []
    for name, n in SYSTEMS:
        sub = df[df["system"] == name]
        if sub.empty:
            continue
        k = int(sub["ideal_match"].sum())
        m = len(sub)
        lo, hi = wilson_interval(k, m)
        # Primary RREF metric: per-generator direct (unreduced) scalar match.
        rk = int(sub["rref_all_direct"].sum())
        rlo, rhi = wilson_interval(rk, m)
        # Secondary diagnostic only -- see _score's docstring.
        rik = int(sub["rref_ideal_match"].sum())
        rilo, rihi = wilson_interval(rik, m)
        summary_rows.append({
            "system": name, "n_vortices": n, "degree": degree, "sigma": sigma,
            "n_seeds": m, "ideal_match_rate": k / m,
            "ci_low": lo, "ci_high": hi,
            "P_rate": sub["P_recovered"].mean(),
            "Q_rate": sub["Q_recovered"].mean(),
            "I_rate": sub["I_recovered"].mean(),
            "mean_n_generators": sub["n_generators"].mean(),
            "mean_runtime_s": sub["runtime_s"].mean(),
            "rref_direct_match_rate": rk / m,
            "rref_ci_low": rlo, "rref_ci_high": rhi,
            "rref_P_direct_rate": sub["rref_P_direct"].mean(),
            "rref_Q_direct_rate": sub["rref_Q_direct"].mean(),
            "rref_I_direct_rate": sub["rref_I_direct"].mean(),
            "rref_mean_n_generators": sub["rref_n_generators"].mean(),
            "rref_ideal_match_rate_DIAGNOSTIC_ONLY": rik / m,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("Results/vortex_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Point-vortex multi-invariant recovery (full-nullspace deflation)")
    print("=" * 70)
    for r in summary_rows:
        print(f"{r['system']:9s} (n={r['n_vortices']}): ideal match "
              f"{r['ideal_match_rate']:.0%} 95% CI "
              f"[{r['ci_low']:.0%}, {r['ci_high']:.0%}]  "
              f"P/Q/I={r['P_rate']:.0%}/{r['Q_rate']:.0%}/{r['I_rate']:.0%}  "
              f"mean gens={r['mean_n_generators']:.1f}  |  "
              f"RREF direct match {r['rref_direct_match_rate']:.0%} 95% CI "
              f"[{r['rref_ci_low']:.0%}, {r['rref_ci_high']:.0%}]  "
              f"P/Q/I={r['rref_P_direct_rate']:.0%}/{r['rref_Q_direct_rate']:.0%}/"
              f"{r['rref_I_direct_rate']:.0%}  mean gens={r['rref_mean_n_generators']:.1f}  "
              f"(ideal_match diagnostic: "
              f"{r['rref_ideal_match_rate_DIAGNOSTIC_ONLY']:.0%})")
    print("=" * 70)
    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N/degree as full run")
    args = parser.parse_args()
    if args.quick:
        benchmark(n_seeds=2)
    else:
        # 30 seeds matches the paper's standard full-run seed grid; recovery at
        # sigma=0 is deterministic across seeds (the invariants are exact), so
        # the seed count only sharpens the Wilson interval, it does not change
        # which invariants are found.
        benchmark(n_seeds=30)
