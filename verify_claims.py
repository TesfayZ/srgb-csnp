#!/usr/bin/env python3
"""
verify_claims.py - reproduces the paper's closed-form derivations and
hand-computed worked examples in under a minute, calling the ACTUAL sr_gb.py
functions wherever one exists (rationality_cost, snap_round) rather than
re-implementing the formula, so this also catches paper/code drift, not just
arithmetic mistakes.

Scope is deliberately narrow: only claims that are checkable WITHOUT
generating data or running the search pipeline (no seeds, no benchmarks),
and not already covered by an existing script. Three adjacent claims are
covered elsewhere and intentionally NOT duplicated here:
  - the Q_dt closed forms for symplectic Euler / Stormer-Verlet and their
    dt-scaling log-log slopes (Section~\\ref{app:verlet-derivation}) are
    verified symbolically by benchmark_dt_sweep_modified_equation.py, which
    is already registered under "validation_and_verification";
  - the Normal-Form Pruning Lemma's counterexample (I_G=<x^2-y>, p=x^2,
    NF_G(p)=y) is already covered by test_safe_pruning_counterexample.py,
    which exercises it through the real prune_active_monomials code path,
    not a standalone symbolic reimplementation;
  - headline recovery-rate numbers (73.1%, 96.7%, etc.) require the full
    30-seed benchmarks and are out of scope for a sub-minute script; see
    validate_sr_gb.py and run_all_experiments.py.

Each check prints a PASS/FAIL line; exit code is nonzero on any failure.
A summary is written to Results/verify_claims_results.txt.

    python verify_claims.py
"""

import sys
import os
import time
from datetime import datetime

import numpy as np
from fractions import Fraction

from sr_gb import rationality_cost, snap_round


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond


def test_rationality_cost_floor():
    """Section 3.4: R(c) >= |S|, and a +-1 coefficient costs exactly 1 bit,
    so an all-unit support of size k costs R=k."""
    ones = np.array([1.0, -1.0, 1.0, -1.0])
    R = rationality_cost(ones)
    exact_one_bit = check("rationality_cost.a: all-unit support of size 4 costs R=4",
                          R == 4.0, f"got R={R}")

    # R >= |S|: a support entry whose normalised magnitude falls below eps
    # must still charge >=1 bit when the caller declares it part of the
    # support (the `support=` kwarg), which is what keeps R>=|S| an
    # admissible lower bound for bb_search (Theorem 4.4 step (ii)).
    near_zero = np.array([1.0, 1e-6])
    R_support = rationality_cost(near_zero, support=[0, 1])
    floor_holds = check("rationality_cost.b: sub-eps support entry still charges >=1 bit (R>=|S|)",
                        R_support >= 2.0, f"got R={R_support} for |S|=2")
    return exact_one_bit and floor_holds


def test_worked_example_section_3_4():
    """The harmonic-oscillator worked example at dt=0.1: the energy x^2+v^2's
    noisy SVD estimate (~0.9998 per coefficient) costs R=4 bits; the
    sampling artifact's cos(dt)=cos(0.1) coefficient matches no p/q with
    q<=16 within eps=1e-3, so R=infinity."""
    energy_estimate = np.array([0.9998, 0.9998, 0.9998, 0.9998])
    R_energy = rationality_cost(energy_estimate)
    energy_ok = check("worked_example.a: noisy energy estimate (~0.9998 x4) costs R=4",
                      R_energy == 4.0, f"got R={R_energy}")

    dt = 0.1
    cos_dt = float(np.cos(dt))
    nearest = min(((p, q) for q in range(1, 17) for p in range(1, q + 1)
                   if p <= q), key=lambda pq: abs(cos_dt - pq[0] / pq[1]))
    dist = abs(cos_dt - nearest[0] / nearest[1])
    nearest_ok = check("worked_example.b: nearest p/q (q<=16) to cos(0.1) is 1/1",
                       nearest == (1, 1), f"got {nearest}, distance={dist:.6f}")
    dist_ok = check("worked_example.c: that distance exceeds eps=1e-3",
                    dist > 1e-3, f"distance={dist:.6f}")

    artifact_estimate = np.array([cos_dt, 1.0, -1.0])
    R_artifact = rationality_cost(artifact_estimate)
    artifact_ok = check("worked_example.d: sampling-artifact coefficient (cos(0.1)) costs R=inf",
                        R_artifact == float('inf'), f"got R={R_artifact}")
    return energy_ok and nearest_ok and dist_ok and artifact_ok


def test_denominator_ceiling_normalisation_convention():
    """Section 'Denominator Ceiling': q* must be computed under snap_round's
    OWN normalisation (divide by the largest-magnitude entry), since a
    normalisation chosen instead to minimise the resulting denominator can
    understate the true floor substantially. c=[2,1]: normalising by the
    dominant entry (2) gives [1, 0.5], q*=2; normalising by the OTHER entry
    (1) gives [2,1], q*=1 -- an understated floor. snap_round must match the
    dominant-entry convention, not the denominator-minimising one."""
    result = snap_round([2.0, 1.0])
    dom_norm_ok = check(
        "denom_ceiling.a: snap_round([2,1]) normalises by the dominant entry (2), "
        "giving denominator 2 for the second coefficient, not the understated 1",
        result == [Fraction(1, 1), Fraction(1, 2)], f"got {result}")
    return dom_norm_ok


def test_qmax_alt_formula_pinned_at_ten():
    """Section 'Denominator Ceiling': the rejected alternative
    Q_max = max(1, floor(1/(2*tau))) with tau=max(0.05, 3*sigma) is pinned
    at Q_max=10 at the noise floor (sigma=0)."""
    import math
    tau = max(0.05, 3.0 * 0.0)
    q_alt = max(1, math.floor(1.0 / (2.0 * tau)))
    return check("qmax_alt.a: alternative Q_max formula evaluates to 10 at sigma=0",
                q_alt == 10, f"got Q_max={q_alt}")


if __name__ == "__main__":
    results = []
    t0 = time.time()
    results.append(("rationality_cost_floor", test_rationality_cost_floor()))
    results.append(("worked_example_section_3_4", test_worked_example_section_3_4()))
    results.append(("denominator_ceiling_normalisation_convention",
                    test_denominator_ceiling_normalisation_convention()))
    results.append(("qmax_alt_formula_pinned_at_ten", test_qmax_alt_formula_pinned_at_ten()))
    dt = time.time() - t0
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} claim checks passed in {dt:.2f}s")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    os.makedirs("Results", exist_ok=True)
    with open("Results/verify_claims_results.txt", "w") as f:
        f.write(f"Claim verification run at {datetime.now().isoformat()}\n")
        f.write(f"{n_ok}/{len(results)} checks passed\n\n")
        for name, ok in results:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}\n")
    print(f"\nSummary written to Results/verify_claims_results.txt")

    sys.exit(0 if n_ok == len(results) else 1)
