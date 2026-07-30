#!/usr/bin/env python3
"""
benchmark_dt_discriminator.py - the fixed-dt, d>1 discriminator.

Builds the one comparison table that isolates SR-GB+CSNP's actual moat:
the harmonic oscillator's wide transition dictionary (x_t, v_t, x_next,
v_next; degree 2; 15 monomials), which is genuinely multi-dimensional in
its nullspace (d>1, confirmed by the existing exact-flow measurements in
Results/omp_nullspace_summary.csv and Results/harmonic_oscillator_sindy_comparison.csv),
run under three data-generating regimes:

  1. exact flow    - the exact rotation map (already measured elsewhere;
                     reused here rather than rerun, no integrator, no H_dt).
  2. fixed-dt symplectic Euler - the integrator conserves a modified
                     invariant Q_dt != H exactly; this is the center cell.
  3. variable dt   - dt drawn per-step from a range, so there is no single
                     fixed Q_dt to tie against.

against three methods: SINDy-null (d=1-style sparsity), OMP-on-nullspace
(d>1, no rationality), and SR-GB+CSNP (CSNP+BB). Reports exact recovery
of the true transition invariant x_t^2+v_t^2-x_next^2-v_next^2 in every
cell, run honestly rather than assumed: if SR-GB+CSNP does not cleanly
win the fixed-dt cell (e.g. it instead reproduces the wide-dictionary
tie already documented in Section 4.6/sec:modified-equation-future),
that is reported as such, not papered over.

Also reports SINDy-FD on the two fixed-dt regimes, as a check on whether a
method that searches a different, smaller dictionary faces the same
ambiguity. SINDy-FD does not operate on the transition-pair dictionary at
all: it consumes the time-ordered trajectory and fits a state polynomial
p(x_t,v_t) over 2 variables (6 monomials at degree 2) with a
finite-difference derivative, rather than searching the 15-monomial
transition-pair dictionary the other three methods share. Reported under
two scorings, since it targets a different object than the transition
invariant: direct match against the state energy x_t^2+v_t^2, and match
against the transition invariant via the induced form
p(x_t,v_t)-p(x_next,v_next). Not run on variable_dt, where no single fixed
dt exists for its uniform-step finite difference to use.

Also reports SINDy-AD alongside SINDy-FD, same dictionary and both
scorings, differing only in how the derivative is estimated: SINDy-AD
fits a smoothing spline per state variable and differentiates that,
SINDy-FD takes a raw finite difference. This is a lightweight proxy for
Kaheman et al. 2022's SINDy-AD (arXiv:2009.08810), not a reimplementation
of it: the real method jointly optimizes a denoised trajectory, a
parametric noise distribution, and the sparse dynamics coefficients
together via automatic differentiation; the proxy only replaces the
derivative estimator and reuses the rest of the SINDy-FD pipeline
unchanged. That gap is deliberately not filled here, because this script
runs at sigma=0, where there is no measurement noise for a jointly-learned
noise model to characterize, so the feature being compared is derivative
truncation-error accuracy (spline vs. finite difference), not noise
robustness. Each SINDy-AD call is wrapped in a wall-clock timeout
(utils_stats.call_with_timeout) since a smoothing-spline fit, unlike the
closed-form baselines here, has no hard runtime bound; timeouts are
recorded as their own outcome, not folded into "no candidate found".
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sympy import parse_expr, simplify, symbols
import warnings
warnings.filterwarnings('ignore')

from sr_gb import sr_gb, exact_recovery
from sindy_baselines import sindy_nullspace, sindy_fd_trajectories, sindy_ad_trajectories
from benchmark_omp_nullspace import omp_nullspace
from utils_stats import call_with_timeout

SINDY_AD_TIMEOUT_S = 30

RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VAR_NAMES = ["x_t", "v_t", "x_next", "v_next"]
TRUE_TRANSITION = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")
TRUE_STATE = parse_expr("x_t**2 + v_t**2")


def generate_exact_flow(N, dt, sigma, seed):
    """Exact rotation flow (reused for comparability with existing measurements).
    Also returns the full state trajectory for the SINDy-FD arm, which needs
    a time-ordered sequence to finite-difference rather than transition pairs."""
    from benchmark_harmonic_oscillator_vs_sindy import generate_harmonic_trajectory
    pairs, states = generate_harmonic_trajectory(N=N, dt=dt, sigma=sigma, seed=seed)
    return pairs, states


def generate_symplectic_euler_fixed_dt(N, dt, sigma, seed):
    """Symplectic (semi-implicit) Euler at a single fixed dt."""
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-1, 1)
    v0 = rng.uniform(-1, 1)
    states = [(x0, v0)]
    for _ in range(N):
        x, v = states[-1]
        v_next = v - dt * x
        x_next = x + dt * v_next
        states.append((x_next, v_next))
    states = np.array(states)
    if sigma > 0:
        states = states + rng.normal(0, sigma, states.shape)
    pairs = np.column_stack([states[:-1, 0], states[:-1, 1], states[1:, 0], states[1:, 1]])
    return pairs, states


def generate_symplectic_euler_variable_dt(N, dt_range, sigma, seed):
    """Symplectic Euler with dt drawn independently per step from dt_range."""
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-1, 1)
    v0 = rng.uniform(-1, 1)
    states = [(x0, v0)]
    for _ in range(N):
        x, v = states[-1]
        dt = rng.uniform(*dt_range)
        v_next = v - dt * x
        x_next = x + dt * v_next
        states.append((x_next, v_next))
    states = np.array(states)
    if sigma > 0:
        states = states + rng.normal(0, sigma, states.shape)
    pairs = np.column_stack([states[:-1, 0], states[:-1, 1], states[1:, 0], states[1:, 1]])
    return pairs, None  # no single fixed dt; SINDy-FD's uniform-step finite
    # difference does not apply here, so this mode has no FD arm


def check_recovery(cands, true_expr):
    if cands is None:
        return False
    if not isinstance(cands, (list, tuple)):
        cands = [cands]
    for p in cands:
        if p is None:
            continue
        e = p.as_expr() if hasattr(p, "as_expr") else p
        if hasattr(e, "as_expr"):
            e = e.as_expr()
        if e == 0:
            continue
        try:
            ratio = simplify(e / true_expr)
            if ratio.is_constant() and ratio != 0:
                return True
        except Exception:
            continue
    return False


# Each mode pairs a data generator (returning (pairs, states_or_None)) with
# the fixed dt SINDy-FD needs for its finite-difference derivative, or None
# where no single fixed dt exists (variable_dt) and the FD arm is skipped.
MODES = {
    "exact_flow": {
        "gen": lambda N, sigma, seed: generate_exact_flow(N, 0.1, sigma, seed),
        "fd_dt": 0.1,
    },
    "fixed_dt_symplectic_euler": {
        "gen": lambda N, sigma, seed: generate_symplectic_euler_fixed_dt(N, 0.1, sigma, seed),
        "fd_dt": 0.1,
    },
    "variable_dt": {
        "gen": lambda N, sigma, seed: generate_symplectic_euler_variable_dt(N, (0.05, 0.15), sigma, seed),
        "fd_dt": None,
    },
}


def run(n_seeds=30, N=5000, sigma=0.0):
    rows = []
    x_t, v_t, x_next, v_next = symbols("x_t v_t x_next v_next")
    for mode_name, cfg in MODES.items():
        gen, fd_dt = cfg["gen"], cfg["fd_dt"]
        print(f"\n=== mode: {mode_name} ===")
        t0 = time.time()
        for seed in range(n_seeds):
            pairs, states = gen(N, sigma, seed)

            try:
                gb_csnp = sr_gb(pairs, VAR_NAMES, degree=None, D_max=2, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {mode_name} seed {seed}: sr_gb error: {e}")
                gb_csnp = []
            csnp_ok = exact_recovery(gb_csnp, TRUE_TRANSITION)

            try:
                gb_omp = omp_nullspace(pairs, VAR_NAMES, degree=2, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {mode_name} seed {seed}: omp_nullspace error: {e}")
                gb_omp = []
            omp_ok = check_recovery(gb_omp, TRUE_TRANSITION)

            try:
                null_cands = sindy_nullspace(pairs, VAR_NAMES, 2, sigma_estimate=sigma)
            except Exception as e:
                print(f"  {mode_name} seed {seed}: sindy_nullspace error: {e}")
                null_cands = []
            sindy_ok = check_recovery(null_cands, TRUE_TRANSITION)

            row = {"mode": mode_name, "seed": seed, "csnp_exact": csnp_ok,
                   "omp_exact": omp_ok, "sindy_null_exact": sindy_ok}

            if fd_dt is not None:
                try:
                    poly_fd = sindy_fd_trajectories(states, ["x_t", "v_t"], 2, dt=fd_dt,
                                                    sigma_estimate=sigma)
                except Exception as e:
                    print(f"  {mode_name} seed {seed}: sindy_fd_trajectories error: {e}")
                    poly_fd = None
                fd_state_ok, fd_transition_ok = False, False
                if poly_fd is not None:
                    try:
                        ratio = simplify(poly_fd / TRUE_STATE)
                        fd_state_ok = bool(ratio.is_constant() and ratio != 0)
                    except Exception:
                        pass
                    induced = poly_fd - poly_fd.subs({x_t: x_next, v_t: v_next}, simultaneous=True)
                    if induced != 0:
                        try:
                            ratio2 = simplify(induced / TRUE_TRANSITION)
                            fd_transition_ok = bool(ratio2.is_constant() and ratio2 != 0)
                        except Exception:
                            pass
                row["sindy_fd_state_exact"] = fd_state_ok
                row["sindy_fd_transition_exact"] = fd_transition_ok

                ad_state_ok, ad_transition_ok, ad_timeout = False, False, False
                try:
                    poly_ad = call_with_timeout(
                        sindy_ad_trajectories,
                        args=(states, ["x_t", "v_t"], 2),
                        kwargs={"dt": fd_dt, "sigma_estimate": sigma},
                        timeout_s=SINDY_AD_TIMEOUT_S)
                except TimeoutError:
                    poly_ad = None
                    ad_timeout = True
                except Exception as e:
                    print(f"  {mode_name} seed {seed}: sindy_ad_trajectories error: {e}")
                    poly_ad = None
                if poly_ad is not None:
                    try:
                        ratio = simplify(poly_ad / TRUE_STATE)
                        ad_state_ok = bool(ratio.is_constant() and ratio != 0)
                    except Exception:
                        pass
                    induced_ad = poly_ad - poly_ad.subs({x_t: x_next, v_t: v_next}, simultaneous=True)
                    if induced_ad != 0:
                        try:
                            ratio2 = simplify(induced_ad / TRUE_TRANSITION)
                            ad_transition_ok = bool(ratio2.is_constant() and ratio2 != 0)
                        except Exception:
                            pass
                row["sindy_ad_state_exact"] = ad_state_ok
                row["sindy_ad_transition_exact"] = ad_transition_ok
                row["sindy_ad_timeout"] = ad_timeout
            else:
                row["sindy_fd_state_exact"] = np.nan
                row["sindy_fd_transition_exact"] = np.nan
                row["sindy_ad_state_exact"] = np.nan
                row["sindy_ad_transition_exact"] = np.nan
                row["sindy_ad_timeout"] = np.nan

            rows.append(row)
        print(f"  {n_seeds} seeds in {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    summary = df.groupby("mode").agg(
        csnp_rate=("csnp_exact", "mean"),
        omp_rate=("omp_exact", "mean"),
        sindy_null_rate=("sindy_null_exact", "mean"),
        sindy_fd_state_rate=("sindy_fd_state_exact", "mean"),
        sindy_fd_transition_rate=("sindy_fd_transition_exact", "mean"),
        sindy_ad_state_rate=("sindy_ad_state_exact", "mean"),
        sindy_ad_transition_rate=("sindy_ad_transition_exact", "mean"),
        sindy_ad_n_timeout=("sindy_ad_timeout", "sum"),
        n_seeds=("seed", "count"),
    ).reset_index()
    print("\n" + "=" * 70)
    print("Fixed-dt vs variable-dt discriminator summary")
    print("=" * 70)
    print(summary.to_string(index=False))

    df.to_csv(os.path.join(RESULTS_DIR, "dt_discriminator_results.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, "dt_discriminator_summary.csv"), index=False)
    print(f"\nSaved to {RESULTS_DIR}/dt_discriminator_*.csv")
    return df, summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        n_seeds = 2
    elif len(sys.argv) > 1:
        n_seeds = int(sys.argv[1])
    else:
        n_seeds = 30
    run(n_seeds=n_seeds)
