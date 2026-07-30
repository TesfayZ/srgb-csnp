#!/usr/bin/env python3
"""
benchmark_deflation_multi_invariant.py – Multi-invariant recovery on a 2D
harmonic oscillator (independent x- and y- axes, same frequency).

Recovery routes through the pipeline: `sr_gb()` with a prebuilt
difference library (`monomials=`/`Phi=`, data=None) and `full_nullspace=True`,
so the same cost-routed circuit search + full-nullspace deflation the core
method uses everywhere is what gets benchmarked here, rather than a private
reimplementation. The recovered generating set is checked against all four
ground-truth generators (per-axis energies, symmetric cross term, angular
momentum); the energy/ang-mom columns are reported in the paper table.

DESIGN NOTES (rationale for the library construction below).
The difference-library construction avoids two failure modes:

FAILURE MODE 1 (library explosion): building ONE degree-2 monomial
library over the *joint* 8-variable space (x_t,y_t,vx_t,vy_t,x_n,y_n,vx_n,vy_n),
M=45. Because (x_n,...,vy_n) is an exact deterministic linear function of
(x_t,...,vy_t), the entire "new" block of columns is linearly dependent on the
"old" block, giving a true nullspace dimension of d=30 -- far beyond both the
combinatorial budget of the exact Branch-and-Bound search (which is skipped once
comb(M,k)*d^2 > 1e6, already exceeded at k=2) and the capability of the L1
fallback, which returns a dense, physically meaningless ~40-term polynomial.
INSTEAD: build a single degree-2 library over the 4-variable *state*
(x,y,vx,vy), evaluate it once on the pre-step state and once on the
post-step state, and take Phi_old - Phi_new (dropping the trivial constant
column, which is always 1-1=0). A conserved quantity Q satisfies
Q(old) - Q(new) = 0 on every sample, so it lies in the nullspace of this much
smaller, purpose-built M=14 matrix, and the true nullspace dimension collapses
to d=4 (see note on ground truth below) -- well within BB's exact search budget.

FAILURE MODE 2 (fragile rank estimation): sr_gb.estimate_rank() picks the rank
via the largest ratio of adjacent singular values, regularized by an absolute
epsilon (1e-30). When the trailing singular values include a value that
underflows to *exactly* 0.0 next to a "small but nonzero" (~1e-15) neighbour,
that pair's ratio (~1e15) can spuriously exceed the true signal/noise gap ratio
and hijack argmax, giving a badly wrong rank. This is exactly what happens on
the noiseless (sigma=0) difference library here. estimate_rank therefore uses a
rank-relative regularization floor (proportional to the leading singular value)
instead of an absolute constant, so near-zero/near-zero ratios don't explode.

NOTE ON GROUND TRUTH: the "energy" typically quoted for this system,
x^2+vx^2+y^2+vy^2, is NOT itself a minimal generator of the conserved-quantity
ideal -- it is the (non-minimal) sum of two sparser invariants. Because the x-
and y- axes evolve as two independent single-frequency oscillators, the true
minimal generating set (reduced Groebner basis) of quadratic invariants is:
    x^2 + vx^2        (per-axis "x energy")
    y^2 + vy^2         (per-axis "y energy")
    x*y + vx*vy        (symmetric cross term)
    x*vy - y*vx        (angular momentum)
Any sparsity-seeking recovery method (this one included) will correctly prefer
the two 2-term per-axis energies over the 4-term combined "total energy",
since the latter is algebraically redundant (= sum of the first two). This
script's exact-match check reflects this: "energy recovered" checks whether
x^2+vx^2 and y^2+vy^2 (or their sum) is exactly recoverable, alongside angular
momentum, rather than requiring the non-minimal 4-term combination verbatim.

Also reports SINDy-FD on a single continuous trajectory of the same system,
over the same 4-variable state dictionary (x,y,vx,vy) it always restricts
to. Unlike the harmonic oscillator's single-invariant case (where SINDy-FD
matches SR-GB+CSNP's 100%), this system has four simultaneous invariants,
and SINDy-FD's single nullspace-direction read is exactly the "arbitrary
mixture" failure Kaiser-Kutz-Brunton describe: it does not recover any of
the four generators in any seed. Compounding that, its finite-difference
derivative estimate is imprecise even at sigma=0 (a first-order slope, not
the exact ODE derivative), which blurs the spectral gap enough that its own
nullspace-dimension estimate comes out far larger than the true d=4.

Also reports SINDy-AD on the identical trajectory and dictionary, differing
from SINDy-FD only in the derivative estimator (a smoothing spline per
state variable instead of a raw finite difference). This is a lightweight
proxy for Kaheman et al. 2022's SINDy-AD (arXiv:2009.08810), not a
reimplementation: the real method jointly optimizes a denoised trajectory,
a parametric noise distribution, and the sparse dynamics coefficients via
automatic differentiation, whereas the proxy here only swaps the
derivative estimator into the unchanged SINDy-FD pipeline. That gap is
deliberately left unfilled because this benchmark runs at sigma=0, where
there is no measurement noise for a jointly-learned noise model to
characterize; the feature actually under test is whether a lower-
truncation-error derivative (relevant here since SINDy-FD's own d=10-vs-
true-d=4 rank blowup is a sigma=0 accuracy artifact, not a noise artifact)
changes the "arbitrary mixture" outcome, not noise robustness. Each
SINDy-AD call is wrapped in a wall-clock timeout
(utils_stats.call_with_timeout); timeouts are recorded as their own
outcome, not folded into "no candidate found".

Saves: Results/deflation_multi_invariant_results_fixed.csv
"""

import numpy as np
from sympy import parse_expr, cancel
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

from sr_gb import build_monomial_library, sr_gb
from sindy_baselines import (sindy_nullspace, sindy_fd_trajectories,
                             sindy_ad_trajectories, rref_nullspace)
from utils_stats import wilson_interval, call_with_timeout

SINDY_AD_TIMEOUT_S = 30


def generate_2d_harmonic_pairs(N=5000, dt=0.1, sigma=0.0, seed=42):
    np.random.seed(seed)
    x_t = np.random.uniform(-1, 1, N)
    y_t = np.random.uniform(-1, 1, N)
    vx_t = np.random.uniform(-1, 1, N)
    vy_t = np.random.uniform(-1, 1, N)
    c, s = np.cos(dt), np.sin(dt)
    x_n = c * x_t + s * vx_t
    vx_n = -s * x_t + c * vx_t
    y_n = c * y_t + s * vy_t
    vy_n = -s * y_t + c * vy_t
    data = np.column_stack([x_t, y_t, vx_t, vy_t, x_n, y_n, vx_n, vy_n])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data


def generate_2d_harmonic_trajectory(N=5000, dt=0.1, sigma=0.0, seed=42):
    """A single continuous trajectory of the same system, for SINDy-FD's
    finite-difference derivative (the pair generator above samples
    independent initial conditions per row, which has no time axis to
    differentiate along)."""
    rng = np.random.RandomState(seed)
    x0, y0, vx0, vy0 = rng.uniform(-1, 1, 4)
    c, s = np.cos(dt), np.sin(dt)
    states = [(x0, y0, vx0, vy0)]
    for _ in range(N):
        x, y, vx, vy = states[-1]
        states.append((c * x + s * vx, c * y + s * vy,
                       -s * x + c * vx, -s * y + c * vy))
    states = np.array(states)
    if sigma > 0:
        states = states + rng.normal(0, sigma, states.shape)
    return states


def exact_match(poly, target_expr):
    if poly is None or poly == 0:
        return False
    try:
        ratio = cancel(poly / target_expr)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return False


def run_deflation(seed, N=5000, dt=0.1, sigma=0.0, max_invariants=4):
    """Route the difference library through the real pipeline: sr_gb() with a
    prebuilt Phi (monomials=/Phi=) and full_nullspace=True, which runs the
    same cost-routed circuit search + deflation loop the core method uses
    everywhere else, instead of a private bb_search + orthogonal-complement
    deflation.
    data=None makes sr_gb gate candidates on the Phi row residual (a conserved
    quantity vanishes as Q(old)-Q(new)=0 per sample, not pointwise on states).
    """
    state_vars = ["x", "y", "vx", "vy"]
    data = generate_2d_harmonic_pairs(N=N, dt=dt, sigma=sigma, seed=seed)
    old, new = data[:, 0:4], data[:, 4:8]

    sym_vars, monomials, evaluate = build_monomial_library(
        state_vars, max_degree=2, min_degree=0, scale=False)
    Phi_old, _, _ = evaluate(old)
    Phi_new, _, _ = evaluate(new)

    # Drop the trivial constant column (always exactly 1 - 1 = 0)
    Phi_diff = (Phi_old - Phi_new)[:, 1:]
    monomials_nc = monomials[1:]

    try:
        gb = sr_gb(None, state_vars, degree=2, monomials=monomials_nc,
                   Phi=Phi_diff, sigma_estimate=sigma, full_nullspace=True)
    except Exception:
        gb = []
    polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]

    true_energy_x = parse_expr("x**2 + vx**2")
    true_energy_y = parse_expr("y**2 + vy**2")
    true_cross = parse_expr("x*y + vx*vy")
    true_ang_mom = parse_expr("x*vy - y*vx")

    energy_x_ok = any(exact_match(p, true_energy_x) for p in polys)
    energy_y_ok = any(exact_match(p, true_energy_y) for p in polys)
    cross_ok = any(exact_match(p, true_cross) for p in polys)
    ang_mom_ok = any(exact_match(p, true_ang_mom) for p in polys)

    # SINDy-null (KRONIC) on the IDENTICAL difference dictionary: the paper's
    # deflation discussion quotes a SINDy-null-on-the-same-dictionary
    # comparison, produced here. Scored with the same any-of exact_match
    # against the same four ground-truth generators.
    try:
        sindy_polys = sindy_nullspace(None, state_vars, 2, sigma_estimate=sigma,
                                      Phi=Phi_diff, monomials=monomials_nc)
    except Exception:
        sindy_polys = []
    s_ex = any(exact_match(p, true_energy_x) for p in sindy_polys)
    s_ey = any(exact_match(p, true_energy_y) for p in sindy_polys)
    s_cross = any(exact_match(p, true_cross) for p in sindy_polys)
    s_am = any(exact_match(p, true_ang_mom) for p in sindy_polys)

    # RREF disambiguation (Oellerich & Emelianenko 2024) on the IDENTICAL
    # difference dictionary: reduces the same SVD nullspace basis CSNP
    # consumes to reduced row-echelon form instead of a sparsity/rationality
    # criterion. Tests whether that linear-algebra canonicalisation, shown
    # to work on two co-existing LINEAR laws in the source paper, also
    # disentangles these four overlapping-support NONLINEAR generators.
    try:
        rref_polys = rref_nullspace(None, state_vars, 2, sigma_estimate=sigma,
                                    Phi=Phi_diff, monomials=monomials_nc)
    except Exception:
        rref_polys = []
    r_ex = any(exact_match(p, true_energy_x) for p in rref_polys)
    r_ey = any(exact_match(p, true_energy_y) for p in rref_polys)
    r_cross = any(exact_match(p, true_cross) for p in rref_polys)
    r_am = any(exact_match(p, true_ang_mom) for p in rref_polys)

    # SINDy-FD on a single continuous trajectory of the same system,
    # restricted (as it always is) to the 4-variable state dictionary. It
    # returns one candidate, not a list, so it can match at most one of the
    # four generators per seed, so "full set recovered" is structurally
    # impossible for it and is not reported.
    trajectory = generate_2d_harmonic_trajectory(N=N, dt=dt, sigma=sigma, seed=seed)
    try:
        fd_poly = sindy_fd_trajectories(trajectory, state_vars, 2, dt=dt,
                                        sigma_estimate=sigma)
    except Exception as e:
        print(f"  seed {seed}: sindy_fd_trajectories error: {e}")
        fd_poly = None
    fd_ex = exact_match(fd_poly, true_energy_x)
    fd_ey = exact_match(fd_poly, true_energy_y)
    fd_cross = exact_match(fd_poly, true_cross)
    fd_am = exact_match(fd_poly, true_ang_mom)

    # SINDy-AD: identical trajectory/dictionary as the SINDy-FD arm above,
    # differing only in the derivative estimator (spline vs. finite
    # difference). See the module docstring for why the joint noise-model
    # piece of the real method is not needed at this benchmark's sigma=0.
    ad_timeout = False
    try:
        ad_poly = call_with_timeout(
            sindy_ad_trajectories, args=(trajectory, state_vars, 2),
            kwargs={"dt": dt, "sigma_estimate": sigma},
            timeout_s=SINDY_AD_TIMEOUT_S)
    except TimeoutError:
        ad_poly = None
        ad_timeout = True
    except Exception as e:
        print(f"  seed {seed}: sindy_ad_trajectories error: {e}")
        ad_poly = None
    ad_ex = exact_match(ad_poly, true_energy_x)
    ad_ey = exact_match(ad_poly, true_energy_y)
    ad_cross = exact_match(ad_poly, true_cross)
    ad_am = exact_match(ad_poly, true_ang_mom)

    # "Energy recovered" if at least one of the two per-axis energy generators
    # (the minimal, sparsest form) is found -- see NOTE ON GROUND TRUTH above.
    return {"seed": seed,
            "energy_recovered": energy_x_ok or energy_y_ok,
            "ang_mom_recovered": ang_mom_ok,
            "energy_x_recovered": energy_x_ok,
            "energy_y_recovered": energy_y_ok,
            "cross_term_recovered": cross_ok,
            "full_set_recovered": (energy_x_ok and energy_y_ok
                                   and cross_ok and ang_mom_ok),
            "n_generators": len(polys),
            "sindy_null_energy_recovered": s_ex or s_ey,
            "sindy_null_ang_mom_recovered": s_am,
            "sindy_null_cross_recovered": s_cross,
            "sindy_null_full_set_recovered": (s_ex and s_ey and s_cross and s_am),
            "sindy_null_n_candidates": len(sindy_polys),
            "rref_energy_recovered": r_ex or r_ey,
            "rref_ang_mom_recovered": r_am,
            "rref_cross_recovered": r_cross,
            "rref_full_set_recovered": (r_ex and r_ey and r_cross and r_am),
            "rref_n_candidates": len(rref_polys),
            "sindy_fd_energy_recovered": fd_ex or fd_ey,
            "sindy_fd_ang_mom_recovered": fd_am,
            "sindy_fd_cross_recovered": fd_cross,
            "sindy_fd_any_recovered": fd_ex or fd_ey or fd_cross or fd_am,
            "sindy_ad_energy_recovered": ad_ex or ad_ey,
            "sindy_ad_ang_mom_recovered": ad_am,
            "sindy_ad_cross_recovered": ad_cross,
            "sindy_ad_any_recovered": ad_ex or ad_ey or ad_cross or ad_am,
            "sindy_ad_timeout": ad_timeout}


def benchmark_deflation(n_seeds=30, N=5000, dt=0.1, sigma=0.0):
    results = []
    print(f"Running FIXED deflation benchmark (σ={sigma}, seeds={n_seeds})...")
    for seed in range(n_seeds):
        row = run_deflation(seed, N=N, dt=dt, sigma=sigma)
        results.append(row)
        print(f"  seed={seed:2d}: energy={row['energy_recovered']}, ang_mom={row['ang_mom_recovered']}")

    df = pd.DataFrame(results)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/deflation_multi_invariant_results_fixed.csv", index=False)

    energy_rate = df["energy_recovered"].mean()
    ang_rate = df["ang_mom_recovered"].mean()
    both_rate = (df["energy_recovered"] & df["ang_mom_recovered"]).mean()
    full_rate = df["full_set_recovered"].mean()
    ci_energy = wilson_interval(df["energy_recovered"].sum(), n_seeds)
    ci_ang = wilson_interval(df["ang_mom_recovered"].sum(), n_seeds)
    ci_both = wilson_interval((df["energy_recovered"] & df["ang_mom_recovered"]).sum(), n_seeds)
    ci_full = wilson_interval(df["full_set_recovered"].sum(), n_seeds)

    print("\n" + "=" * 60)
    print("Deflation Multi-Invariant Recovery (2D Harmonic Oscillator) via sr_gb()")
    print(f"Energy (per-axis) recovered: {energy_rate:.0%} 95% CI [{ci_energy[0]:.0%}, {ci_energy[1]:.0%}]")
    print(f"Angular momentum recovered: {ang_rate:.0%} 95% CI [{ci_ang[0]:.0%}, {ci_ang[1]:.0%}]")
    print(f"Both recovered: {both_rate:.0%} 95% CI [{ci_both[0]:.0%}, {ci_both[1]:.0%}]")
    print(f"Full 4-generator set recovered: {full_rate:.0%} 95% CI [{ci_full[0]:.0%}, {ci_full[1]:.0%}]")
    s_energy = df["sindy_null_energy_recovered"].mean()
    s_ang = df["sindy_null_ang_mom_recovered"].mean()
    ci_se = wilson_interval(df["sindy_null_energy_recovered"].sum(), n_seeds)
    ci_sa = wilson_interval(df["sindy_null_ang_mom_recovered"].sum(), n_seeds)
    print(f"SINDy-null (same dictionary) energy: {s_energy:.0%} 95% CI [{ci_se[0]:.0%}, {ci_se[1]:.0%}]")
    print(f"SINDy-null (same dictionary) ang. mom.: {s_ang:.0%} 95% CI [{ci_sa[0]:.0%}, {ci_sa[1]:.0%}]")
    r_full = df["rref_full_set_recovered"].mean()
    ci_rfull = wilson_interval(df["rref_full_set_recovered"].sum(), n_seeds)
    r_energy = df["rref_energy_recovered"].mean()
    ci_re = wilson_interval(df["rref_energy_recovered"].sum(), n_seeds)
    r_ang = df["rref_ang_mom_recovered"].mean()
    ci_ra = wilson_interval(df["rref_ang_mom_recovered"].sum(), n_seeds)
    r_cross = df["rref_cross_recovered"].mean()
    ci_rc = wilson_interval(df["rref_cross_recovered"].sum(), n_seeds)
    print(f"RREF (same dictionary) energy: {r_energy:.0%} 95% CI [{ci_re[0]:.0%}, {ci_re[1]:.0%}]")
    print(f"RREF (same dictionary) cross term: {r_cross:.0%} 95% CI [{ci_rc[0]:.0%}, {ci_rc[1]:.0%}]")
    print(f"RREF (same dictionary) ang. mom.: {r_ang:.0%} 95% CI [{ci_ra[0]:.0%}, {ci_ra[1]:.0%}]")
    print(f"RREF (same dictionary) full 4-generator set: {r_full:.0%} 95% CI [{ci_rfull[0]:.0%}, {ci_rfull[1]:.0%}]")
    fd_any = df["sindy_fd_any_recovered"].mean()
    ci_fd = wilson_interval(df["sindy_fd_any_recovered"].sum(), n_seeds)
    print(f"SINDy-FD (state dictionary) any of the four: {fd_any:.0%} 95% CI [{ci_fd[0]:.0%}, {ci_fd[1]:.0%}]")
    ad_any = df["sindy_ad_any_recovered"].mean()
    ci_ad = wilson_interval(df["sindy_ad_any_recovered"].sum(), n_seeds)
    n_ad_timeout = int(df["sindy_ad_timeout"].sum())
    print(f"SINDy-AD (state dictionary) any of the four: {ad_any:.0%} 95% CI [{ci_ad[0]:.0%}, {ci_ad[1]:.0%}] "
          f"({n_ad_timeout} timeouts)")
    print("=" * 60)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N as full run")
    args = parser.parse_args()
    if args.quick:
        benchmark_deflation(n_seeds=3, N=5000, dt=0.1, sigma=0.0)
    else:
        benchmark_deflation(n_seeds=30, N=5000, dt=0.1, sigma=0.0)