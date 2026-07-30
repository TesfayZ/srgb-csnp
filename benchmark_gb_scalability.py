#!/usr/bin/env python3
"""
benchmark_gb_scalability.py – times reduced Gröbner basis computation
(SymPy, grevlex order over QQ) on random sparse polynomial ideals, one
configuration per row of the paper's Table tab:gb-scalability.

This script produces the data and CSV backing that table and the abstract's
Gröbner-stage scaling claim. Each configuration is timed on N_INSTANCES seeded
random instances and the median is reported; generators are sparse (3-6 term)
polynomials with small rational coefficients, the shape snap-rounding
actually hands to the Gröbner stage.

Saves: Results/gb_scalability.csv
"""

import argparse
import os
import time
from fractions import Fraction
from math import comb

import numpy as np
import pandas as pd
from sympy import Rational, groebner, symbols

from sr_gb import build_monomial_library

# (n_vars, degree, n_generators) -- the configurations of tab:gb-scalability.
CONFIGS = [
    (4, 2, 1),
    (6, 2, 2),
    (8, 2, 3),
    (10, 2, 4),
    (12, 2, 5),
    (8, 3, 2),
    (10, 3, 3),
]

COEFF_POOL = [Rational(v) for v in
              (1, -1, 2, -2, 3, -3, Fraction(1, 2), Fraction(-1, 2))]


def random_ideal(n, D, k, rng):
    """k random polynomials of 4-6 terms drawn from the full library, with
    two shared top-degree monomials injected into every generator so the
    ideals genuinely interact (fully independent sparse supports make
    Buchberger terminate almost immediately, understating cost, while a
    heavily collided restricted pool blows up doubly-exponentially and
    times out; this middle ground is the shape snap-rounded pipeline
    outputs actually take: sparse, small-rational, partially overlapping)."""
    var_names = [f"x{i}" for i in range(n)]
    sym_vars, monomials, _ = build_monomial_library(var_names, D, min_degree=0,
                                                    scale=False)
    M = len(monomials)
    shared = [M - 1 - int(rng.integers(0, min(6, M - 1))) for _ in range(2)]
    gens = []
    for _ in range(k):
        n_terms = int(rng.integers(4, 7))
        idx = list(rng.choice(M - 1, size=n_terms, replace=False) + 1)
        support = list(dict.fromkeys(shared + idx))
        poly = sum(COEFF_POOL[int(rng.integers(len(COEFF_POOL)))] * monomials[int(j)]
                   for j in support)
        if poly == 0:
            poly = monomials[int(support[0])]
        gens.append(poly)
    return gens, sym_vars


class _InstanceTimeout(Exception):
    pass


def _timed_groebner(gens, sym_vars, cap_s):
    """Run sympy groebner under a SIGALRM cap; returns elapsed seconds or
    None on timeout (Buchberger's worst case is doubly exponential, and a
    single unlucky random instance must not stall the whole benchmark)."""
    import signal

    def _raise(_sig, _frm):
        raise _InstanceTimeout()

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(cap_s)
    t0 = time.time()
    try:
        groebner(gens, *sym_vars, order="grevlex")
        return time.time() - t0
    except _InstanceTimeout:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="One instance per configuration instead of the median of three")
    parser.add_argument("--outdir", default="Results")
    parser.add_argument("--instance-cap", type=int, default=120,
                        help="Per-instance wall cap in seconds; timeouts are reported as such")
    args = parser.parse_args()
    n_instances = 1 if args.quick else 3

    rows = []
    for n, D, k in CONFIGS:
        M = comb(n + D, D)
        times = []
        n_timeout = 0
        for inst in range(n_instances):
            rng = np.random.default_rng(1000 * n + 100 * D + 10 * k + inst)
            gens, sym_vars = random_ideal(n, D, k, rng)
            dt = _timed_groebner(gens, sym_vars, args.instance_cap)
            if dt is None:
                n_timeout += 1
            else:
                times.append(dt)
        t_med = float(np.median(times)) if times else float("nan")
        rows.append({"n": n, "D": D, "M": M, "generators": k,
                     "n_instances": n_instances, "n_timeout": n_timeout,
                     "instance_cap_s": args.instance_cap,
                     "time_median_s": round(t_med, 4)})
        print(f"n={n:2d} D={D} M={M:3d} k={k}: median {t_med:.3f}s "
              f"over {len(times)} instances ({n_timeout} timed out at "
              f"{args.instance_cap}s)")

    df = pd.DataFrame(rows)
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "gb_scalability.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
