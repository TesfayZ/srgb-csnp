"""
benchmark_robotics.py – Evaluate SR-GB+CSNP on robot trajectory invariants.
Requires MuJoCo. If it is unavailable the benchmark aborts without writing a
result CSV, rather than fabricating trajectories that would report a misleading
0% recovery (see robot_data.generate_robot_trajectories).
"""

import sys
import numpy as np
import pandas as pd
from sympy import parse_expr
from sr_gb import sr_gb, exact_recovery
from robot_data import generate_robot_trajectories, sample_with_variable_dt
from utils_stats import wilson_interval
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Simple policies (dummy)
# ----------------------------------------------------------------------
def null_policy(state):
    return np.zeros(4)  # no-op action

# ----------------------------------------------------------------------
# Generate data from robot trajectories (pairs: state_t, state_{t+dt})
# ----------------------------------------------------------------------
def robot_to_pairs(trajectory):
    states = np.array(trajectory['states'])
    next_states = np.array(trajectory['next_states'])
    # stack states and next states horizontally
    return np.hstack([states[:-1], next_states[:-1]])

# ----------------------------------------------------------------------
# Known invariants for a simple pendulum (dummy)
# ----------------------------------------------------------------------
def pendulum_invariant(var_names):
    # Suppose variables are [q, qdot, q_next, qdot_next]
    # Energy conservation: q^2 + qdot^2 = q_next^2 + qdot_next^2
    return parse_expr("q**2 + qdot**2 - q_next**2 - qdot_next**2")

# ----------------------------------------------------------------------
# Run benchmark
# ----------------------------------------------------------------------
def benchmark_robotics(seeds=30, n_trajectories=10, horizon=100,
                       model_xml="pendulum.xml", sigma=0.0):
    """
    For each seed, generate multiple trajectories, extract pairs,
    and run SR-GB+CSNP to recover the energy invariant.
    """
    var_names = ["q", "qdot", "q_next", "qdot_next"]
    true_inv = pendulum_invariant(var_names)
    results = []
    for seed in range(seeds):
        np.random.seed(seed)
        # Generate trajectories (raises RuntimeError if MuJoCo is unavailable)
        trajs = generate_robot_trajectories(model_xml, null_policy,
                                            horizon, n_trajectories,
                                            noise_sigma=sigma, dt=0.01)
        # Aggregate all pairs from all trajectories
        all_pairs = []
        for traj in trajs:
            pairs = robot_to_pairs(traj)
            all_pairs.append(pairs)
        data = np.vstack(all_pairs)
        # Run SR-GB+CSNP (adaptive, D_max=2)
        gb = sr_gb(data, var_names, degree=None, D_max=2, sigma_estimate=sigma)
        exact = exact_recovery(gb, true_inv)
        results.append({"seed": seed, "exact": exact})
        print(f"Seed {seed}: exact={exact}")

    df = pd.DataFrame(results)
    rate = df["exact"].mean()
    k = df["exact"].sum()
    ci = wilson_interval(k, seeds)
    print(f"\nRobotics benchmark: exact recovery rate = {rate:.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
    df.to_csv("Results/benchmark_robotics.csv", index=False)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same full run otherwise")
    args = parser.parse_args()
    try:
        benchmark_robotics(seeds=2 if args.quick else 30)
    except RuntimeError as e:
        # No MuJoCo -> no real data. Skip loudly (non-zero exit, no CSV written)
        # instead of reporting a fabricated recovery rate.
        print(f"SKIPPED benchmark_robotics: {e}")
        sys.exit(1)