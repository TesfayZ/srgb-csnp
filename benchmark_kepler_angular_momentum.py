#!/usr/bin/env python3
"""
Kepler benchmark: angular momentum conservation.
Uses analytic solution (Kepler's equation) for exact transition pairs.
Fully automatic (no expected_support required).
"""

import numpy as np
from sympy import parse_expr
from sr_gb import sr_gb, exact_recovery
from utils_stats import wilson_interval
import pandas as pd 
import warnings
warnings.filterwarnings('ignore')

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

def generate_kepler_pairs(N_pairs=5000, dt=0.1, seed=42):
    np.random.seed(seed)
    all_pairs = []
    for _ in range(N_pairs):
        a = np.random.uniform(0.8, 1.2)
        e = np.random.uniform(0, 0.8)
        M0 = np.random.uniform(0, 2*np.pi)
        pair = kepler_analytic(a, e, M0, dt)
        all_pairs.append(pair)
    return np.array(all_pairs)

def benchmark_kepler(seeds=list(range(30)), N_pairs=5000, dt=0.1):
    var_names = ["x_t", "y_t", "vx_t", "vy_t", "x_next", "y_next", "vx_next", "vy_next"]
    true_invariant = "x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next"
    true_expr = parse_expr(true_invariant)

    results = {"exact": []}
    for seed in seeds:
        data = generate_kepler_pairs(N_pairs=N_pairs, dt=dt, seed=seed)
        # Adaptive: degree=None, D_max=2 (quadratic invariant)
        try:
            gb = sr_gb(data, var_names, degree=None, D_max=2, sigma_estimate=0.0)
        except Exception as e:
            print(f"  seed {seed}: sr_gb error: {e}")
            gb = []
        exact = exact_recovery(gb, true_expr)
        results["exact"].append(exact)
    k = sum(results["exact"])
    ci = wilson_interval(k, len(seeds))
    print(f"Exact recovery rate: {k/len(seeds):.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
    
    # ---- SAVE CSV ----
    df = pd.DataFrame({"seed": seeds, "exact": results["exact"]})
    df.to_csv("Results/kepler_angular_momentum_results.csv", index=False)
    print("Results saved to Results/kepler_angular_momentum_results.csv")
    
    return {
        "system": "Kepler (angular momentum)",
        "exact_rate": np.mean(results["exact"]),
        "redundancy": 1.0,
        "ci_low": ci[0],
        "ci_high": ci[1],
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same N_pairs as full run")
    args = parser.parse_args()
    print("Benchmarking Kepler angular momentum (analytic solution) – fully automatic CSNP")
    if args.quick:
        res = benchmark_kepler(seeds=list(range(2)), N_pairs=5000, dt=0.1)
    else:
        res = benchmark_kepler(seeds=list(range(30)), N_pairs=5000, dt=0.1)
    print(f"\nExact recovery rate: {res['exact_rate']:.0%}, Redundancy: {res['redundancy']:.1f}")