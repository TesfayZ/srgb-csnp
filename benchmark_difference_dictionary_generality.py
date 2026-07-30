#!/usr/bin/env python3
"""
Difference-dictionary generality probe: sigma > 0 and a second integrator.

The paper's modified-equation discussion (sec:modified-equation-future)
records one structural mitigation: restricting the transition dictionary to
difference features p(x_t,v_t) - p(x_{t+1},v_{t+1}) excludes the update-rule
identity by construction, and on symplectic-Euler harmonic-oscillator data
(dt=0.1, sigma=0) the search over that dictionary returns exactly the
modified invariant Q_dt. Until now every difference-dictionary result was at
sigma=0 and on that single integrator, so the exclusion argument was
"evidence of generality, not a proof of it". This script measures the two
missing axes:

  * representative sampling noise (sigma in {0, 0.01, 0.02, 0.05}), and
  * a second symplectic integrator, Stormer-Verlet, whose exact conserved
    quadratic differs from symplectic Euler's (B=0, C=4/(4-dt^2): no cross
    term, an anisotropic v^2 weight).

Both maps are linear, so the exactly conserved quadratic form
Q_dt = x^2 + B(dt)*xv + C(dt)*v^2 is derived symbolically per (integrator,
dt) with the same closed-form machinery as the dt-sweep gate
(benchmark_dt_sweep_modified_equation.exact_conserved_quadratic) and used as
the per-cell ground truth. dt values are chosen so Q_dt's coefficients are
small rationals inside the pipeline's declared max_denom=16 scope
(euler: -dt at dt in {1/10, 1/2}; verlet: C = 16/15 at dt=1/2, 4/3 at dt=1;
verlet at dt=1/10 gives C = 400/399 and is deliberately OUT of scope, so
including it would measure the rational-snap scope, not the dictionary).

Scored outcomes per trial:
  exact_Qdt   - a returned generator is proportional to Q_dt (the honest
                best case: H itself is not conserved by the map);
  exact_H     - a returned generator is proportional to the true energy
                x^2 + v^2. Under noise this is an expected, interpretable
                collapse mode whenever 3*sigma exceeds |B|: the finalization
                threshold eps = 3*sigma zeroes the xv coefficient and Q_dt
                becomes indistinguishable from H (euler dt=1/10 at
                sigma=0.05 sits exactly in that regime).
  n_generators, returned polynomials (for inspection).

Saves: Results/difference_dictionary_generality_results.csv (per-trial)
       Results/difference_dictionary_generality_rates.csv   (rates + CIs)
"""

import time
import numpy as np
import pandas as pd
import sympy as sp
from sympy import parse_expr, cancel, Rational, fraction, nsimplify

from sr_gb import sr_gb, build_monomial_library
from utils_stats import wilson_interval
from benchmark_dt_sweep_modified_equation import (
    exact_conserved_quadratic, symplectic_euler_map, stormer_verlet_map,
    _numeric_map, dt_sym)

MAX_DENOM = 16

INTEGRATORS = {
    "symplectic_euler": (symplectic_euler_map, [Rational(1, 10), Rational(1, 2)]),
    "stormer_verlet": (stormer_verlet_map, [Rational(1, 2), Rational(1, 1)]),
}


def derive_Qdt(map_fn, dt_val):
    """Exact conserved quadratic of the linear map at this dt, as a sympy
    expression in (x, v); asserts its coefficients are in-scope rationals."""
    B, C = exact_conserved_quadratic(*map_fn())
    b = nsimplify(B.subs(dt_sym, dt_val))
    c = nsimplify(C.subs(dt_sym, dt_val))
    for coeff in (b, c):
        q = fraction(Rational(coeff))[1]
        assert q <= MAX_DENOM, (
            f"Q_dt coefficient {coeff} has denominator {q} > {MAX_DENOM}; "
            "this dt would measure rational-snap scope, not the dictionary")
    x, v = sp.symbols("x v")
    return x ** 2 + b * x * v + c * v ** 2


def exact_match(poly, target):
    if poly is None or poly == 0:
        return False
    try:
        ratio = cancel(poly / target)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return False


def run_trial(map_fn, dt_val, sigma, seed, N=5000):
    """One transition-data trial through the real pipeline: prebuilt
    difference library (monomials=/Phi=, data=None) + full_nullspace=True,
    the same entry point the multi-invariant deflation benchmark uses."""
    xn_f, vn_f = _numeric_map(map_fn)
    rng = np.random.RandomState(seed)
    x_t = rng.uniform(-2.0, 2.0, N)
    v_t = rng.uniform(-2.0, 2.0, N)
    dtf = float(dt_val)
    x_n = xn_f(x_t, v_t, dtf)
    v_n = vn_f(x_t, v_t, dtf)
    old = np.column_stack([x_t, v_t])
    new = np.column_stack([x_n, v_n])
    if sigma > 0:
        old = old + rng.normal(0.0, sigma, old.shape)
        new = new + rng.normal(0.0, sigma, new.shape)

    state_vars = ["x", "v"]
    sym_vars, monomials, evaluate = build_monomial_library(
        state_vars, max_degree=2, min_degree=0, scale=False)
    Phi_old, _, _ = evaluate(old)
    Phi_new, _, _ = evaluate(new)
    Phi_diff = (Phi_old - Phi_new)[:, 1:]   # drop the constant column (1-1=0)
    monomials_nc = monomials[1:]

    try:
        gb = sr_gb(None, state_vars, degree=2, monomials=monomials_nc,
                   Phi=Phi_diff, sigma_estimate=sigma, full_nullspace=True)
    except Exception:
        gb = []
    return [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]


def benchmark(n_seeds, sigmas, N=5000):
    H = parse_expr("x**2 + v**2")
    rows = []
    for name, (map_fn, dt_values) in INTEGRATORS.items():
        for dt_val in dt_values:
            Qdt = derive_Qdt(map_fn, dt_val)
            for sigma in sigmas:
                for seed in range(n_seeds):
                    t0 = time.time()
                    polys = run_trial(map_fn, dt_val, sigma, seed, N=N)
                    elapsed = time.time() - t0
                    rows.append({
                        "integrator": name, "dt": float(dt_val),
                        "sigma": sigma, "seed": seed,
                        "exact_Qdt": any(exact_match(p, Qdt) for p in polys),
                        "exact_H": any(exact_match(p, H) for p in polys),
                        "n_generators": len(polys),
                        "Qdt": str(Qdt),
                        "returned": "; ".join(str(p) for p in polys),
                        "time": round(elapsed, 3),
                    })
                sub = [r for r in rows if r["integrator"] == name
                       and r["dt"] == float(dt_val) and r["sigma"] == sigma]
                print(f"{name:17s} dt={float(dt_val):<5} sigma={sigma:.2f}: "
                      f"Q_dt {sum(r['exact_Qdt'] for r in sub)}/{len(sub)}, "
                      f"H {sum(r['exact_H'] for r in sub)}/{len(sub)}  "
                      f"(Q_dt = {sub[0]['Qdt']})")
    df = pd.DataFrame(rows)

    rates = df.groupby(["integrator", "dt", "sigma"]).agg(
        Qdt_rate=("exact_Qdt", "mean"),
        H_rate=("exact_H", "mean"),
        mean_generators=("n_generators", "mean"),
    ).reset_index()
    for idx, row in rates.iterrows():
        sub = df[(df["integrator"] == row["integrator"])
                 & (df["dt"] == row["dt"]) & (df["sigma"] == row["sigma"])]
        lo, hi = wilson_interval(sub["exact_Qdt"].sum(), len(sub))
        rates.loc[idx, "Qdt_ci_low"] = lo
        rates.loc[idx, "Qdt_ci_high"] = hi
    return df, rates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only; same cells and N")
    args = parser.parse_args()

    sigmas = [0.00, 0.01, 0.02, 0.05]
    n_seeds = 2 if args.quick else 30
    print(f"Difference-dictionary generality probe "
          f"({n_seeds} seeds, sigma in {sigmas})")
    df, rates = benchmark(n_seeds, sigmas)

    print("\n=== Rates ===")
    print(rates.to_string(index=False))
    df.to_csv("Results/difference_dictionary_generality_results.csv", index=False)
    rates.to_csv("Results/difference_dictionary_generality_rates.csv", index=False)
    print("\nSaved to Results/difference_dictionary_generality_*.csv")
