#!/usr/bin/env python3
"""
Benchmark harmonic oscillator energy invariant: SR-GB+CSNP vs SINDy-null, ST, FD.
SINDy-FD uses finite differences for derivative estimation (not the published SINDy-AD).
"""

import numpy as np
from sklearn.linear_model import Lasso
from sympy import parse_expr, simplify, symbols
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sr_gb import sr_gb, exact_recovery, build_monomial_library
from sindy_baselines import sindy_nullspace, sindy_st_ensemble, sindy_fd_trajectories
from utils_stats import wilson_interval


def generate_harmonic_trajectory(N=5000, dt=0.1, sigma=0.0, seed=42):
    """Generate a single trajectory of length N+1, then extract pairs."""
    np.random.seed(seed)
    x0 = np.random.uniform(-1, 1)
    v0 = np.random.uniform(-1, 1)
    c, s = np.cos(dt), np.sin(dt)
    states = [(x0, v0)]
    for _ in range(N):
        x, v = states[-1]
        x_next = c*x + s*v
        v_next = -s*x + c*v
        states.append((x_next, v_next))
    states = np.array(states)
    if sigma > 0:
        states += np.random.normal(0, sigma, states.shape)
    
    # Pairs: (x_t, v_t, x_{t+1}, v_{t+1}) for t = 0..N-1
    pairs = np.column_stack([states[:-1,0], states[:-1,1], states[1:,0], states[1:,1]])
    return pairs, states   # pairs for null/ST, states for FD


def benchmark_system(name, N=5000, dt=0.1, sigma=0.0, seeds=list(range(30))):
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    transition_invariant = "x_t**2 + v_t**2 - x_next**2 - v_next**2"
    true_transition = parse_expr(transition_invariant)
    
    # State energy for FD comparison
    state_invariant = "x_t**2 + v_t**2"
    true_state = parse_expr(state_invariant)

    results = {"srgb": [], "sindy_null": [], "sindy_st": [],
               "sindy_fd_state": [], "sindy_fd_transition": []}
    for seed in seeds:
        pairs, trajectory = generate_harmonic_trajectory(N=N, dt=dt, sigma=sigma, seed=seed)
        
        # ---- SR-GB+CSNP (uses pairs) – now adaptive, D_max=2 ----
        try:
            gb = sr_gb(pairs, var_names, degree=None, D_max=2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  {name} seed {seed}: sr_gb error: {e}")
            gb = []
        results["srgb"].append(exact_recovery(gb, true_transition))

        # ---- SINDy-null (uses pairs) ----
        try:
            null_cands = sindy_nullspace(pairs, var_names, 2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  {name} seed {seed}: sindy_nullspace error: {e}")
            null_cands = []
        null_ok = False
        if null_cands:
            for p in null_cands:
                ratio = simplify(p / true_transition)
                if ratio.is_constant():
                    null_ok = True
                    break
        results["sindy_null"].append(null_ok)

        # ---- SINDy-ST (uses pairs; all-targets implicit STLSQ ensemble,
        # the same standard configuration every other benchmark uses) ----
        try:
            st_cands = sindy_st_ensemble(pairs, var_names, 2, sigma_estimate=sigma)
        except Exception as e:
            print(f"  {name} seed {seed}: sindy_st_ensemble error: {e}")
            st_cands = []
        st_ok = False
        for p in st_cands:
            ratio = simplify(p / true_transition)
            if ratio.is_constant():
                st_ok = True
                break
        results["sindy_st"].append(st_ok)

        # ---- SINDy-FD (uses the full trajectory, state variables only) ----
        # Dual scoring: FD consumes the time-ordered trajectory rather than
        # transition pairs and recovers a STATE polynomial p(x_t, v_t), a
        # different scoring target than the transition invariant the other rows
        # use. Report BOTH: (a) direct match against the state energy, and
        # (b) match against the transition target via the induced form
        # p(x_t,v_t)-p(x_next,v_next). FD can match under scoring (a) even
        # where it fails under (b), so reporting only one column would mislabel
        # its behaviour.
        try:
            poly_fd = sindy_fd_trajectories(trajectory, ["x_t", "v_t"], 2, dt=dt, sigma_estimate=sigma)
        except Exception as e:
            print(f"  {name} seed {seed}: sindy_fd_trajectories error: {e}")
            poly_fd = None
        fd_state_ok = False
        fd_transition_ok = False
        if poly_fd is not None:
            ratio = simplify(poly_fd / true_state)
            fd_state_ok = bool(ratio.is_constant())
            x_t, v_t, x_next, v_next = symbols("x_t v_t x_next v_next")
            induced = (poly_fd
                       - poly_fd.subs({x_t: x_next, v_t: v_next}, simultaneous=True))
            if induced != 0:
                ratio2 = simplify(induced / true_transition)
                fd_transition_ok = bool(ratio2.is_constant())
        results["sindy_fd_state"].append(fd_state_ok)
        results["sindy_fd_transition"].append(fd_transition_ok)

    print(f"\n{name} (σ={sigma})")
    for key in results:
        k = sum(results[key])
        ci = wilson_interval(k, len(seeds))
        label = key.replace("sindy_", "SINDy-").replace("srgb", "SR-GB+CSNP")
        print(f"  {label}: {k/len(seeds):.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")

    df = pd.DataFrame({
        "seed": seeds,
        "srgb_exact": results["srgb"],
        "sindy_null_exact": results["sindy_null"],
        "sindy_st_exact": results["sindy_st"],
        "sindy_fd_state_exact": results["sindy_fd_state"],
        "sindy_fd_transition_exact": results["sindy_fd_transition"],
    })
    df.to_csv("Results/harmonic_oscillator_sindy_comparison.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N as full run")
    args = parser.parse_args()
    print("Harmonic Oscillator: SR-GB vs SINDy-null, ST, FD")
    if args.quick:
        benchmark_system("HarmonicOscillator", N=5000, dt=0.1, sigma=0.0, seeds=list(range(3)))
    else:
        benchmark_system("HarmonicOscillator", N=5000, dt=0.1, sigma=0.0, seeds=list(range(30)))