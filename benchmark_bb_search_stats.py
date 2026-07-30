#!/usr/bin/env python3
"""
benchmark_bb_search_stats.py - Instruments bb_search() (sr_gb.py) with node
counters (nodes popped, nodes actually evaluated, nodes pruned by the
admissible lower bound, nodes pruned by dominance caching, and the beam
warm-start's upper bound versus the final optimum) and reports them on
three systems.

Two are the transition-dictionary systems Review 8 asked about: Kepler
angular momentum and the 2D harmonic oscillator transition invariant, both
a degree-2, 4-variable (8-column raw / 9-dimensional nullspace at sigma=0)
dictionary. Under the current cost-routed circuit search these are now
resolved by CSNP's progressive d-sweep + closed-form/enumeration path
before general bb_search is ever invoked, so their node counters are all
zero: they document that BB is bypassed on this class of instance rather
than measuring its work. The third system, highd_sparse_rank2, is a
genuinely high-d* nullspace (six variables in a rank-2 latent subspace,
degree-2 library, one sparse rational circuit x0 - x1) on which the
enumeration budget is exceeded and general bb_search does fire; it is where
the node statistics are actually measured.

This does not change bb_search's search behaviour; sr_gb.BB_SEARCH_LOG is
appended to on every bb_search() call regardless of who calls it, so this
script simply resets the log before each sr_gb() call and aggregates what
was appended during that call.
"""
import sys
import os
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import sr_gb
from sr_gb import sr_gb as run_sr_gb, exact_recovery
from sympy import parse_expr
from benchmark_kepler_angular_momentum import generate_kepler_pairs
from benchmark_harmonic_oscillator_vs_sindy import generate_harmonic_trajectory

SIGMAS = [0.0, 0.02]
N = 5000


def generate_highd_sparse(seed, sigma):
    """A genuinely high-d* nullspace where enumeration is over budget and
    general bb_search must fire. Six observed variables all lie in a rank-2
    latent subspace (degree-2 library M=28, nullspace dimension d* ~ 22, so
    the pinning enumeration C(M, d*-1) far exceeds the enumeration budget),
    containing exactly one sparse RATIONAL circuit x0 - x1 (the remaining
    latent dependencies have generic irrational coefficients that rationality
    rejects). This is the same construction validate_sr_gb.py uses to confirm
    bb_search fires on the default/full-nullspace path; it is included
    here because the two transition-dictionary systems below (Kepler, the 2D
    oscillator) are resolved by CSNP's progressive d-sweep + enumeration
    path before general bb_search is ever invoked, so their node counts are
    all zero and do not measure BB's search work.
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(-1, 1, N)
    v = rng.uniform(-1, 1, N)
    A = rng.standard_normal((2, 6))
    A[:, 1] = A[:, 0]                      # x1 == x0 exactly
    data = np.column_stack([u, v]) @ A
    if sigma > 0:
        data = data + rng.normal(0, sigma, data.shape)
    return data


def _run_highd(sigma, seed):
    """Runner for the high-d* sparse system: fixed degree 2, full_nullspace
    mode, k_max=4, which is the path on which bb_search genuinely fires."""
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    data = generate_highd_sparse(seed, sigma)
    sr_gb.reset_bb_search_log()
    gb = run_sr_gb(data, var_names, degree=2, sigma_estimate=sigma,
                   k_max=4, full_nullspace=True)
    exact = exact_recovery(gb, parse_expr("x0 - x1"))
    log = list(sr_gb.BB_SEARCH_LOG)
    return _aggregate(exact, log)


def _run_one(var_names, data, true_expr, sigma):
    sr_gb.reset_bb_search_log()
    gb = run_sr_gb(data, var_names, degree=None, D_max=2, sigma_estimate=sigma)
    exact = exact_recovery(gb, true_expr)
    log = list(sr_gb.BB_SEARCH_LOG)
    return _aggregate(exact, log)


def _aggregate(exact, log):
    if not log:
        return dict(exact=exact, n_bb_calls=0, nodes_popped=0, nodes_evaluated=0,
                    nodes_pruned_lb=0, nodes_pruned_dominance=0,
                    nodes_not_pushed_lb=0,
                    beam_ub_cost=None, final_ub_cost=None)
    agg = dict(
        exact=exact,
        n_bb_calls=len(log),
        nodes_popped=sum(s["nodes_popped"] for s in log),
        nodes_evaluated=sum(s["nodes_evaluated"] for s in log),
        nodes_pruned_lb=sum(s["nodes_pruned_lb"] for s in log),
        nodes_pruned_dominance=sum(s["nodes_pruned_dominance"] for s in log),
        nodes_not_pushed_lb=sum(s["nodes_not_pushed_lb"] for s in log),
    )
    # Report the beam-vs-final UB gap for the call with the most nodes
    # (the search that actually mattered), rather than averaging across
    # trivial d's that terminate immediately.
    biggest = max(log, key=lambda s: s["nodes_popped"])
    agg["beam_ub_cost"] = biggest["beam_ub_cost"]
    agg["final_ub_cost"] = biggest["final_ub_cost"]
    return agg


def run_benchmark(n_seeds=10):
    rows = []

    kepler_vars = ["x_t", "y_t", "vx_t", "vy_t",
                   "x_next", "y_next", "vx_next", "vy_next"]
    kepler_truth = parse_expr(
        "x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next")

    harmonic_vars = ["x_t", "v_t", "x_next", "v_next"]
    harmonic_truth = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")

    for sigma in SIGMAS:
        for seed in range(n_seeds):
            data = generate_kepler_pairs(N_pairs=N, dt=0.1, seed=seed)
            if sigma > 0:
                rng = np.random.RandomState(seed)
                data = data + rng.normal(0, sigma, data.shape)
            stats = _run_one(kepler_vars, data, kepler_truth, sigma)
            stats.update(system="kepler_angular_momentum", sigma=sigma, seed=seed)
            rows.append(stats)
            print(f"kepler   sigma={sigma:<5} seed={seed:2d} "
                  f"popped={stats['nodes_popped']:6d} eval={stats['nodes_evaluated']:6d} "
                  f"prune_lb={stats['nodes_pruned_lb']:6d} prune_dom={stats['nodes_pruned_dominance']:6d} "
                  f"not_pushed={stats['nodes_not_pushed_lb']:6d} "
                  f"beam_ub={stats['beam_ub_cost']} final_ub={stats['final_ub_cost']} exact={stats['exact']}")

            pairs, _ = generate_harmonic_trajectory(N=N, dt=0.1, sigma=sigma, seed=seed)
            stats = _run_one(harmonic_vars, pairs, harmonic_truth, sigma)
            stats.update(system="harmonic_oscillator_2d", sigma=sigma, seed=seed)
            rows.append(stats)
            print(f"harmonic sigma={sigma:<5} seed={seed:2d} "
                  f"popped={stats['nodes_popped']:6d} eval={stats['nodes_evaluated']:6d} "
                  f"prune_lb={stats['nodes_pruned_lb']:6d} prune_dom={stats['nodes_pruned_dominance']:6d} "
                  f"not_pushed={stats['nodes_not_pushed_lb']:6d} "
                  f"beam_ub={stats['beam_ub_cost']} final_ub={stats['final_ub_cost']} exact={stats['exact']}")

            stats = _run_highd(sigma, seed)
            stats.update(system="highd_sparse_rank2", sigma=sigma, seed=seed)
            rows.append(stats)
            print(f"highd    sigma={sigma:<5} seed={seed:2d} "
                  f"popped={stats['nodes_popped']:6d} eval={stats['nodes_evaluated']:6d} "
                  f"prune_lb={stats['nodes_pruned_lb']:6d} prune_dom={stats['nodes_pruned_dominance']:6d} "
                  f"not_pushed={stats['nodes_not_pushed_lb']:6d} "
                  f"beam_ub={stats['beam_ub_cost']} final_ub={stats['final_ub_cost']} exact={stats['exact']}")

    df = pd.DataFrame(rows)
    os.makedirs("Results", exist_ok=True)
    df.to_csv("Results/bb_search_stats_results.csv", index=False)

    summary = df.groupby(["system", "sigma"]).agg(
        mean_nodes_popped=("nodes_popped", "mean"),
        mean_nodes_evaluated=("nodes_evaluated", "mean"),
        mean_nodes_pruned_lb=("nodes_pruned_lb", "mean"),
        mean_nodes_pruned_dominance=("nodes_pruned_dominance", "mean"),
        mean_nodes_not_pushed_lb=("nodes_not_pushed_lb", "mean"),
        exact_rate=("exact", "mean"),
    ).reset_index()
    # Fraction of the total nodes GENERATED (pushed + filtered-at-push) that
    # each mechanism removes, since most LB pruning happens at push time
    # (never added to the heap at all) rather than at pop time.
    total_generated = df.groupby(["system", "sigma"]).apply(
        lambda g: g["nodes_popped"].sum() + g["nodes_not_pushed_lb"].sum())
    summary["pruned_fraction_lb_at_pop"] = df.groupby(["system", "sigma"]).apply(
        lambda g: g["nodes_pruned_lb"].sum() / max(g["nodes_popped"].sum(), 1)).values
    summary["pruned_fraction_lb_at_push"] = (
        df.groupby(["system", "sigma"])["nodes_not_pushed_lb"].sum() / total_generated.replace(0, 1)).values
    summary["pruned_fraction_dominance"] = df.groupby(["system", "sigma"]).apply(
        lambda g: g["nodes_pruned_dominance"].sum() / max(g["nodes_popped"].sum(), 1)).values
    summary.to_csv("Results/bb_search_stats_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("BB Search Statistics Summary")
    print("=" * 100)
    print(summary.to_string(index=False))
    return df, summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        run_benchmark(n_seeds=2)
    else:
        run_benchmark(n_seeds=10)
