"""
benchmark_gravity_pendulum.py - Negative counterpart to benchmark_robotics.py.

gravity_pendulum.xml is a real MuJoCo rigid-body pendulum whose conserved
energy 0.5*qdot^2 - (g/L)*cos(q) is transcendental in q, not polynomial of any
bounded degree. SR-GB+CSNP's canonicity guarantee (a reduced Groebner basis)
is a polynomial-ring construction, so no bounded-degree call should ever
return a nonempty basis here: unlike benchmark_robotics.py, there is no known
polynomial target to score exact recovery against, only whether the pipeline
correctly abstains rather than snap-rounding some spurious near-fit into a
false positive. Requires MuJoCo; no synthetic fallback (see
robot_data.generate_robot_trajectories).
"""

import sys
import numpy as np
import pandas as pd
from sr_gb import sr_gb
from robot_data import generate_robot_trajectories
from utils_stats import wilson_interval
import warnings
warnings.filterwarnings('ignore')


def null_policy(state):
    return np.zeros(4)  # no-op action


def robot_to_pairs(trajectory):
    states = np.array(trajectory['states'])
    next_states = np.array(trajectory['next_states'])
    return np.hstack([states[:-1], next_states[:-1]])


def benchmark_gravity_pendulum(seeds=30, n_trajectories=10, horizon=100,
                               model_xml="gravity_pendulum.xml", sigma=0.0):
    var_names = ["q", "qdot", "q_next", "qdot_next"]
    results = []
    for seed in range(seeds):
        np.random.seed(seed)
        trajs = generate_robot_trajectories(model_xml, null_policy,
                                            horizon, n_trajectories,
                                            noise_sigma=sigma, dt=0.01)
        data = np.vstack([robot_to_pairs(t) for t in trajs])
        gb = sr_gb(data, var_names, degree=None, D_max=2, sigma_estimate=sigma)
        found = bool(gb)
        spurious = str(gb) if found else ""
        results.append({"seed": seed, "found_invariant": found, "spurious": spurious})
        print(f"Seed {seed}: found_invariant={found}" + (f" -> {spurious}" if found else ""))

    df = pd.DataFrame(results)
    rate = df["found_invariant"].mean()
    k = df["found_invariant"].sum()
    ci = wilson_interval(k, seeds)
    print(f"\nGravity pendulum: spurious-invariant rate = {rate:.0%} 95% CI [{ci[0]:.0%}, {ci[1]:.0%}]")
    print("(0% is the correct/expected outcome: no bounded-degree polynomial")
    print(" invariant exists, so the pipeline should abstain every trial.)")
    df.to_csv("Results/benchmark_gravity_pendulum.csv", index=False)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Reduced seed count only, same full run otherwise")
    args = parser.parse_args()
    try:
        benchmark_gravity_pendulum(seeds=2 if args.quick else 30)
    except RuntimeError as e:
        print(f"SKIPPED benchmark_gravity_pendulum: {e}")
        sys.exit(1)
