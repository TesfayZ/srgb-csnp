#!/usr/bin/env python3
"""
Test automatic sparse nullspace selection (no expected_support), uses mdl
on two invariant discovery problems:

1. Harmonic oscillator transition: energy invariant
   Data uses random rotation angle per sample to make the energy
   the unique quadratic invariant.

2. Kepler angular momentum conservation: analytic transitions
   Nullspace dimension = 1, automatic CSNP works directly.
"""

import argparse
import numpy as np
from sympy import parse_expr
from sr_gb import sr_gb, exact_recovery
from utils_stats import wilson_interval
import pandas as pd

_parser = argparse.ArgumentParser()
_parser.add_argument("--quick", action="store_true",
                     help="Reduced seed count only, same N as full run")
_args = _parser.parse_args()
_N = 5000
_N_KEP = 2000
_N_SEEDS = 2 if _args.quick else 30

# ------------------------------------------------------------
# Harmonic oscillator with random dt 
# ------------------------------------------------------------
def generate_harmonic_pairs(N=5000, sigma=0.0, seed=42):
    """Generate (x_t, v_t, x_next, v_next) with random dt per sample."""
    np.random.seed(seed)
    x_t = np.random.uniform(-1, 1, N)
    v_t = np.random.uniform(-1, 1, N)
    dt = np.random.uniform(0.05, 0.2, N)
    c = np.cos(dt)
    s = np.sin(dt)
    x_next = c * x_t + s * v_t
    v_next = -s * x_t + c * v_t
    data = np.column_stack([x_t, v_t, x_next, v_next])
    if sigma > 0:
        data += np.random.normal(0, sigma, data.shape)
    return data

# ------------------------------------------------------------
# Kepler 
# ------------------------------------------------------------
def kepler_analytic(a, e, M0, dt, GM=1.0):
    n = np.sqrt(GM / a**3)
    M_t = M0
    M_next = M0 + n * dt
    def E_from_M(M, e):
        E = M
        for _ in range(10):
            E = M + e * np.sin(E)
        return E
    E_t = E_from_M(M_t, e)
    E_next = E_from_M(M_next, e)
    x_t = a * (np.cos(E_t) - e)
    y_t = a * np.sqrt(1 - e**2) * np.sin(E_t)
    x_next = a * (np.cos(E_next) - e)
    y_next = a * np.sqrt(1 - e**2) * np.sin(E_next)
    vx_t = -np.sqrt(GM / a) * np.sin(E_t) / (1 - e * np.cos(E_t))
    vy_t =  np.sqrt(GM / a) * np.sqrt(1 - e**2) * np.cos(E_t) / (1 - e * np.cos(E_t))
    vx_next = -np.sqrt(GM / a) * np.sin(E_next) / (1 - e * np.cos(E_next))
    vy_next =  np.sqrt(GM / a) * np.sqrt(1 - e**2) * np.cos(E_next) / (1 - e * np.cos(E_next))
    return (x_t, y_t, vx_t, vy_t, x_next, y_next, vx_next, vy_next)

def generate_kepler_pairs(N=2000, dt=0.1, seed=42):
    np.random.seed(seed)
    all_pairs = []
    for _ in range(N):
        a = np.random.uniform(0.8, 1.2)
        e = np.random.uniform(0, 0.8)
        M0 = np.random.uniform(0, 2*np.pi)
        pair = kepler_analytic(a, e, M0, dt)
        all_pairs.append(pair)
    return np.array(all_pairs)

# ------------------------------------------------------------
# Test both systems automatically (no expected_support)
# ------------------------------------------------------------
print("=" * 70)
print("Automatic invariant discovery (CSNP, no expected_support)")
print("=" * 70)

# ---- Harmonic oscillator ----
print("\n1. Harmonic oscillator (energy invariant)")
print("   Data: random rotation angle per sample -> well-posed")
data_ho = generate_harmonic_pairs(N=_N, sigma=0.0, seed=42)
var_names_ho = ["x_t", "v_t", "x_next", "v_next"]
true_ho = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")

gb_ho = sr_gb(data_ho, var_names_ho, degree=None, D_max=2, sigma_estimate=0.0)
assert gb_ho, "Recovery failed for the harmonic oscillator energy invariant"
expr = gb_ho[0].as_expr()
print(f"Recovered: {expr}")
exact_ho = exact_recovery(gb_ho, true_ho)
print(f"Exact recovery: {exact_ho}")
assert exact_ho, f"Recovered {expr} does not match the true energy invariant {true_ho}"

# ---- Kepler angular momentum ----
print("\n2. Kepler angular momentum invariant")
print("   Data: analytic Kepler transitions -> null_dim = 1")
data_kep = generate_kepler_pairs(N=_N_KEP, dt=0.1, seed=42)
var_names_kep = ["x_t", "y_t", "vx_t", "vy_t", "x_next", "y_next", "vx_next", "vy_next"]
true_kep = parse_expr("x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next")

gb_kep = sr_gb(data_kep, var_names_kep, degree=None, D_max=2, sigma_estimate=0.0)
assert gb_kep, "Recovery failed for the Kepler angular momentum invariant"
expr = gb_kep[0].as_expr()
print(f"Recovered: {expr}")
exact_kep = exact_recovery(gb_kep, true_kep)
print(f"Exact recovery: {exact_kep}")
assert exact_kep, f"Recovered {expr} does not match the true angular momentum invariant {true_kep}"

# ---- Optional: multiple seeds for harmonic ----
print("\n3. Harmonic oscillator – multiple seeds (auto mode)")
n_seeds = _N_SEEDS
success = []
for seed in range(n_seeds):
    data = generate_harmonic_pairs(N=_N, sigma=0.0, seed=seed)
    gb = sr_gb(data, var_names_ho, degree=None, D_max=2, sigma_estimate=0.0)
    ok = exact_recovery(gb, true_ho) if gb else False
    success.append(ok)
    print(f"  seed={seed}: exact={ok}")
k = sum(success)
ci = wilson_interval(k, n_seeds)
print(f"Exact recovery rate (auto): {sum(success)/n_seeds:.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
assert k == n_seeds, (
    f"Auto (no expected_support) harmonic-oscillator recovery at sigma=0.0 should be "
    f"exact on every seed; got {k}/{n_seeds}. Failing seeds: "
    f"{[s for s, ok in enumerate(success) if not ok]}"
)

# ---- SAVE CSV ----
df = pd.DataFrame({"seed": range(n_seeds), "exact": success})
df.to_csv("Results/test_auto_invariants_harmonic.csv", index=False)
print("Results saved to Results/test_auto_invariants_harmonic.csv")