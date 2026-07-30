#!/usr/bin/env python3
"""
validate_sr_gb.py - regression / smoke tests for key sr_gb.py features:

  1. Column preconditioning (ps3-style rank estimation on wide dynamic ranges).
  2. Full-nullspace deflation: the multi-invariant tail on the default path
     (sigma ~ 0 only) plus the explicit full_nullspace=True mode, in which
     bb_search genuinely fires once the enumeration budget is exceeded.
  3. Prebuilt-Phi entry point (monomials=/Phi=, data=None) through the
     public sr_gb(), gating candidates on the Phi row residual.

Each test prints a PASS/FAIL line; exit code is nonzero on any failure.
A summary is written to Results/validate_sr_gb_results.txt.

    python validate_sr_gb.py
"""

import sys
import time
import os
from datetime import datetime
import numpy as np
from sympy import symbols, parse_expr, cancel

from sr_gb import (sr_gb as run_sr_gb, sr_gb_transition_difference,
                   exact_recovery, build_monomial_library, evaluate_support,
                   reset_bb_search_log, BB_SEARCH_LOG)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not cond else ""))
    return cond


def poly_equiv(p, expr_str, var_names):
    """True if polynomial p (sympy expr or Poly) is a scalar multiple of expr_str."""
    syms = symbols(var_names)
    target = parse_expr(expr_str, local_dict=dict(zip(var_names, syms)))
    p_expr = p.as_expr() if hasattr(p, "as_expr") else p
    try:
        ratio = cancel(p_expr / target)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return False


def test_entangled_multi_invariant():
    """Two independent linear invariants x0-x1, x2-x3 in a degree-2
    (over-lifted, d*=13) monomial library. A single-circuit search returns
    only ONE of the two; the full-nullspace deflation tail (bb firing past
    the enumeration budget) must recover both."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (4000, 6))
    X[:, 1] = X[:, 0]
    X[:, 3] = X[:, 2]
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    gb = run_sr_gb(X, var_names, degree=2, sigma_estimate=0.0)
    polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]
    has_01 = any(poly_equiv(p, "x0 - x1", var_names) for p in polys)
    has_23 = any(poly_equiv(p, "x2 - x3", var_names) for p in polys)
    return check("entangled multi-invariant (deg-2 lib): recovers BOTH x0-x1 and x2-x3",
                 has_01 and has_23,
                 f"got {len(polys)} generators: {polys}")


def test_entangled_multi_invariant_with_noise():
    """Noisy multi-invariant recovery, both paths, documenting the sigma-gate
    contract: on the DEFAULT path at sigma=0.01 the collection tail is
    deliberately gated off (returning the single primary invariant is the
    published noisy behaviour; the gate exists because a generous-d* noisy
    deflation measurably injects spurious generators on distance_3d
    (feynman_polynomials.py, not a verified official Feynman ID) at
    sigma=0.02). The explicit full_nullspace=True opt-in is UNGATED and must
    recover BOTH invariants with nothing spurious."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (4000, 6))
    X[:, 1] = X[:, 0] + rng.normal(0, 0.01, 4000)
    X[:, 3] = X[:, 2] + rng.normal(0, 0.01, 4000)
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]

    gb_def = run_sr_gb(X, var_names, degree=2, sigma_estimate=0.01)
    polys_def = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb_def]
    one_of = (any(poly_equiv(p, "x0 - x1", var_names) for p in polys_def)
              or any(poly_equiv(p, "x2 - x3", var_names) for p in polys_def))
    ok1 = check("noisy entangled, DEFAULT path: exactly 1 genuine invariant (sigma-gate)",
                one_of and len(polys_def) == 1, f"gb={polys_def}")

    gb_fn = run_sr_gb(X, var_names, degree=2, sigma_estimate=0.01,
                      full_nullspace=True)
    polys_fn = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb_fn]
    has_01 = any(poly_equiv(p, "x0 - x1", var_names) for p in polys_fn)
    has_23 = any(poly_equiv(p, "x2 - x3", var_names) for p in polys_fn)
    ok2 = check("noisy entangled, full_nullspace=True: BOTH invariants, no extras",
                has_01 and has_23 and len(polys_fn) == 2, f"gb={polys_fn}")
    return ok1 and ok2


def test_single_invariant_no_spurious_growth():
    """A genuinely SINGLE-invariant, over-lifted degree-2 library must stay at
    exactly 1 generator at sigma = 0, 0.01, 0.02 on BOTH paths -- the widened,
    no-longer-budget-gated deflation tail (and the ungated full_nullspace
    mode) must not invent a fake second generator out of noise directions."""
    ok_all = True
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    for sigma in [0.0, 0.01, 0.02]:
        rng = np.random.RandomState(7)
        X = rng.uniform(-1, 1, (4000, 6))
        if sigma > 0:
            X[:, 1] = X[:, 0] + rng.normal(0, sigma, 4000)
        else:
            X[:, 1] = X[:, 0]
        for mode in (False, True):
            gb = run_sr_gb(X, var_names, degree=2, sigma_estimate=sigma,
                           full_nullspace=mode)
            polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]
            recovered = exact_recovery(gb, parse_expr("x0 - x1"))
            ok_all &= check(
                f"single-invariant no-spurious-growth sigma={sigma} full_nullspace={mode}",
                recovered and len(polys) == 1, f"gb={polys}")
    return ok_all


def test_three_invariants_no_linear_combo_redundancy():
    """Regression test for the ideal-membership redundancy check in
    _collect_additional_invariants: with THREE independent exact linear
    invariants (x0-x1, x2-x3, x4-x5) in a plain degree-1 library, the
    deflation tail must return exactly those 3 generators -- not more.

    This exercises a gap a scalar-multiple-only redundancy check has: once
    p1=x0-x1 and p2=x2-x3 are known, p1+p2 is also a valid exact degree-1
    relation but is algebraically redundant (already in the ideal <p1,p2>)
    without being a scalar multiple of either alone -- only the Groebner
    ideal-membership reduction catches it. A plain degree-1 library is used
    rather than an over-lifted degree-2 one: the redundancy risk applies just
    as directly at degree 1, and an over-lifted library blows up runtime with
    extra product-of-relation degenerate directions unrelated to the thing
    being tested."""
    rng = np.random.RandomState(11)
    X = rng.uniform(-1, 1, (1500, 6))
    X[:, 1] = X[:, 0]
    X[:, 3] = X[:, 2]
    X[:, 5] = X[:, 4]
    var_names = [f"x{i}" for i in range(6)]
    gb = run_sr_gb(X, var_names, degree=1, sigma_estimate=0.0, full_nullspace=True)
    polys = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb]
    has_01 = any(poly_equiv(p, "x0 - x1", var_names) for p in polys)
    has_23 = any(poly_equiv(p, "x2 - x3", var_names) for p in polys)
    has_45 = any(poly_equiv(p, "x4 - x5", var_names) for p in polys)
    return check("three entangled invariants: exactly x0-x1, x2-x3, x4-x5, "
                 "no redundant linear-combination generators",
                 has_01 and has_23 and has_45 and len(polys) == 3,
                 f"got {len(polys)} generators: {polys}")


def test_bb_fires_by_default_fallback():
    """Does bb_search ever fire on the DEFAULT call path (no flags)? 30
    variables with 15 independent exact pairwise-equal relations give a
    degree-1 library with d*=15, far above the sweep's cap (d_cap=4 for
    20<M<=60) and outside the d_try_jump heuristic (M<=20 only), so the
    capped sweep finds nothing and the sigma~0 empty-sweep fallback hands the
    problem to full_nullspace_deflation_search, where enumeration is
    hopeless (C(31,14) huge) and bb at the affordable support cap is exactly
    what should fire. k_max=2 matches the true sparse support size, the
    loop-invariant benchmarks' calling convention."""
    rng = np.random.RandomState(5)
    N = 2500
    n_vars = 30
    X = rng.uniform(-1, 1, (N, n_vars))
    for lo in range(2, n_vars, 2):
        X[:, lo + 1] = X[:, lo]
    X[:, 1] = X[:, 0]
    var_names = [f"x{i}" for i in range(n_vars)]
    reset_bb_search_log()
    gb = run_sr_gb(X, var_names, degree=1, sigma_estimate=0.0, k_max=2)
    fired = len(BB_SEARCH_LOG) > 0
    recovered = exact_recovery(gb, parse_expr("x0 - x1"))
    ok = check("bb_search fires on the DEFAULT path via the empty-sweep fallback",
               fired, f"BB_SEARCH_LOG entries={len(BB_SEARCH_LOG)}")
    ok2 = check("...and recovers the sparse target x0 - x1 with no explicit flags",
                recovered, f"gb={gb}")
    return ok and ok2


def test_noise_robustness_circle():
    """Regression guard: the circle must still recover exactly under noise,
    with exactly one generator (no spurious extras from the collection)."""
    ok_all = True
    for sigma in [0.0, 0.01, 0.02]:
        rng = np.random.RandomState(1)
        theta = rng.uniform(0, 2 * np.pi, 3000)
        X = np.column_stack([np.cos(theta), np.sin(theta)])
        if sigma > 0:
            X = X + rng.normal(0, sigma, X.shape)
        gb = run_sr_gb(X, ["x", "y"], degree=None, D_max=3, sigma_estimate=sigma)
        target = parse_expr("x**2 + y**2 - 1")
        ok = exact_recovery(gb, target) and len(gb) == 1
        ok_all &= check(f"circle recovery at sigma={sigma} (exactly 1 generator)",
                        ok, f"gb={gb}")
    return ok_all


def test_ps3_like_large_dynamic_range():
    """The ps3 case: i^4 - 2*i^3 + i^2 - 4*s = 0 with s reaching large
    magnitude, so the raw degree-4 library spans many orders of magnitude.
    Column preconditioning must give the correct d*=1 recovery instead of a
    spurious high-d estimate."""
    rng = np.random.RandomState(2)
    i = rng.uniform(-40, 40, 4000)
    s = (i**4 - 2 * i**3 + i**2) / 4.0
    X = np.column_stack([i, s])
    gb = run_sr_gb(X, ["i", "s"], degree=4, sigma_estimate=0.0)
    target = parse_expr("i**4 - 2*i**3 + i**2 - 4*s")
    return check("ps3-like large-dynamic-range recovery",
                 exact_recovery(gb, target), f"gb={gb}")


def test_bb_fires_high_d_sparse():
    """A genuinely high-d* exact nullspace where enumeration is over budget:
    6 observed variables all lying in a rank-2 latent subspace (degree-2
    library M=28, d* ~ 22, so C(M, d*-1) ~ 1.2M > the 200k enumeration
    budget), containing exactly one sparse RATIONAL circuit x0 - x1 (the
    other latent dependencies have generic irrational coefficients, so
    rationality rejects them). bb_search must actually fire
    (BB_SEARCH_LOG non-empty) and recover x0 - x1."""
    rng = np.random.RandomState(3)
    u = rng.uniform(-1, 1, 3000)
    v = rng.uniform(-1, 1, 3000)
    A = rng.standard_normal((2, 6))
    A[:, 1] = A[:, 0]                      # x1 == x0 exactly
    X = np.column_stack([u, v]) @ A
    var_names = ["x0", "x1", "x2", "x3", "x4", "x5"]
    reset_bb_search_log()
    gb = run_sr_gb(X, var_names, degree=2, sigma_estimate=0.0, k_max=4,
                   full_nullspace=True)
    fired = len(BB_SEARCH_LOG) > 0
    recovered = exact_recovery(gb, parse_expr("x0 - x1"))
    ok = check("bb_search fires in full_nullspace mode on a high-d* sparse case",
               fired, f"BB_SEARCH_LOG entries={len(BB_SEARCH_LOG)}")
    ok2 = check("...and recovers the sparse target x0 - x1", recovered, f"gb={gb}")
    return ok and ok2


def test_prebuilt_phi_difference_library():
    """sr_gb() with a prebuilt difference library (Phi_old -
    Phi_new, data=None). A conserved quantity Q satisfies Q(old)-Q(new)=0 on
    every SAMPLE but does NOT vanish pointwise on state data, so this only
    works if the residual gates run on Phi rows. 2D rotation conserving
    x^2 + v^2."""
    rng = np.random.RandomState(4)
    N = 3000
    dt = 0.1
    x = rng.uniform(-1, 1, N)
    v = rng.uniform(-1, 1, N)
    c, s = np.cos(dt), np.sin(dt)
    xn = c * x + s * v
    vn = -s * x + c * v
    sym_vars, monomials, evaluate = build_monomial_library(["x", "v"], 2, min_degree=0)
    Phi_old, _, _ = evaluate(np.column_stack([x, v]))
    Phi_new, _, _ = evaluate(np.column_stack([xn, vn]))
    Phi_diff = (Phi_old - Phi_new)[:, 1:]
    monomials_nc = monomials[1:]
    gb = run_sr_gb(None, ["x", "v"], degree=2, monomials=monomials_nc,
                   Phi=Phi_diff, sigma_estimate=0.0, full_nullspace=True)
    ok = check("prebuilt-Phi difference library recovers x^2 + v^2",
               exact_recovery(gb, parse_expr("x**2 + v**2")), f"gb={gb}")

    gb_helper = sr_gb_transition_difference(
        np.column_stack([x, v]), np.column_stack([xn, vn]), ["x", "v"], 2,
        sigma_estimate=0.0, full_nullspace=True)
    ok_helper = check("transition-difference helper recovers x^2 + v^2",
                      exact_recovery(gb_helper, parse_expr("x**2 + v**2")),
                      f"gb={gb_helper}")
    # The adaptive path must reject prebuilt Phi/monomials.
    raised = False
    try:
        run_sr_gb(None, ["x", "v"], degree=None, monomials=monomials_nc, Phi=Phi_diff)
    except ValueError:
        raised = True
    ok2 = check("adaptive path rejects monomials=/Phi= without a fixed degree", raised)
    return ok and ok_helper and ok2


def test_noisy_oracle_guard():
    """Known true supports in the published noisy 2D-oscillator diagnostic
    must never be dominance-pruned as Hard Infeasible.  The test uses the
    diagnostic's fixed nullspace construction so it exercises exactly the
    rank classifier used by BB, not a separate recovery path."""
    from scipy.linalg import svd
    from benchmark_oracle_misclassification import (
        generate_2d_harmonic_pairs, estimate_rank_robust, support_of,
        TRUE_GENERATORS, evaluate, N, DT, SIGMA)

    hard = []
    for seed in range(30):
        data = generate_2d_harmonic_pairs(N=N, dt=DT, sigma=SIGMA, seed=seed)
        Phi_old, _, _ = evaluate(data[:, :4])
        Phi_new, _, _ = evaluate(data[:, 4:])
        Phi_diff = (Phi_old - Phi_new)[:, 1:]
        _, s, Vt = svd(Phi_diff, full_matrices=False)
        d = Phi_diff.shape[1] - estimate_rank_robust(s, sigma_estimate=SIGMA, N=N)
        V_null = Vt[-d:, :].T
        for name, expr in TRUE_GENERATORS.items():
            result = evaluate_support(support_of(expr), V_null, d,
                                      max_denom=16, eps=3 * SIGMA,
                                      sigma_estimate=SIGMA, Phi=Phi_diff,
                                      N=Phi_diff.shape[0])
            if result['status'] == 'HARD_INFEASIBLE':
                hard.append((seed, name))
    return check("noisy oracle guard: no known 2D-oscillator generator is Hard-pruned",
                 not hard, f"hard-pruned={hard}")


if __name__ == "__main__":
    results = []
    t0 = time.time()
    results.append(("entangled_multi_invariant", test_entangled_multi_invariant()))
    results.append(("entangled_multi_invariant_with_noise", test_entangled_multi_invariant_with_noise()))
    results.append(("single_invariant_no_spurious_growth", test_single_invariant_no_spurious_growth()))
    results.append(("three_invariants_no_linear_combo_redundancy", test_three_invariants_no_linear_combo_redundancy()))
    results.append(("bb_fires_by_default_fallback", test_bb_fires_by_default_fallback()))
    results.append(("noise_robustness_circle", test_noise_robustness_circle()))
    results.append(("ps3_like_large_dynamic_range", test_ps3_like_large_dynamic_range()))
    results.append(("bb_fires_high_d_sparse", test_bb_fires_high_d_sparse()))
    results.append(("prebuilt_phi_difference_library", test_prebuilt_phi_difference_library()))
    results.append(("noisy_oracle_guard", test_noisy_oracle_guard()))
    dt = time.time() - t0
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} tests passed in {dt:.1f}s")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    # Write summary to Results/
    os.makedirs("Results", exist_ok=True)
    with open("Results/validate_sr_gb_results.txt", "w") as f:
        f.write(f"Validation run at {datetime.now().isoformat()}\n")
        f.write(f"{n_ok}/{len(results)} tests passed\n\n")
        for name, ok in results:
            f.write(f"{'PASS' if ok else 'FAIL'}  {name}\n")
    print(f"\nSummary written to Results/validate_sr_gb_results.txt")

    sys.exit(0 if n_ok == len(results) else 1)