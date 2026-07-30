#!/usr/bin/env python3
"""
dt-sweep gate for the modified-equation separation claim.

For a fixed-step *symplectic* integrator applied to the harmonic oscillator,
the map exactly conserves a modified quadratic form Q_dt(x,v) that differs
from the true energy x^2+v^2. Because the map is linear, Q_dt can be solved
for exactly (no data, no noise) as the nullspace of (M(dt)^T (x)_2 - I) acting
on the space of quadratic forms, where M(dt) is the one-step map. This script:

  1. Derives Q_dt in closed form for symplectic Euler (order 1) and
     Stormer-Verlet (order 2), and measures ||Q_dt - Q_true|| vs dt on a
     log-log grid to check whether the slope matches the integrator order.
  2. Confirms non-symplectic integrators (explicit Euler, explicit RK2) admit
     no exact conserved quadratic form at all (the energy secularly drifts),
     which scopes the theorem to structure-preserving integrators.

This is a closed-form, symbolic result (no sampled data, no noise, no
statistical fitting): the gap is solved for exactly as the nullspace of a
linear system at each dt, so the log-log slope is not an empirical estimate
subject to noise, it is evaluated at machine precision from an exact
rational-function expression in dt.
"""
import numpy as np
import pandas as pd
import sympy as sp
from sympy import parse_expr

from sr_gb import sr_gb, exact_recovery

dt_sym, A, B, C, x, v = sp.symbols('dt A B C x v')


def exact_conserved_quadratic(x_next_expr, v_next_expr):
    """Solve for the quadratic form Q = A x^2 + B x v + C v^2 (A=1) exactly
    conserved by the linear map (x,v) -> (x_next_expr, v_next_expr), if any."""
    Q = A * x**2 + B * x * v + C * v**2
    Q_next = sp.expand(Q.subs({x: x_next_expr, v: v_next_expr}, simultaneous=True))
    diff = sp.expand(Q_next - Q)
    poly = sp.Poly(diff, x, v)
    eqs = [sp.Eq(c, 0) for c in poly.coeffs()]
    sols = sp.solve(eqs, [B, C], dict=True)
    if not sols:
        return None
    sol = sols[0]
    return sp.simplify(sol[B].subs(A, 1)), sp.simplify(sol[C].subs(A, 1))


def symplectic_euler_map():
    v_next = v - dt_sym * x
    x_next = x + dt_sym * v_next
    return sp.expand(x_next), sp.expand(v_next)


def stormer_verlet_map():
    v_half = v - dt_sym / 2 * x
    x_next = x + dt_sym * v_half
    v_next = v_half - dt_sym / 2 * x_next
    return sp.expand(x_next), sp.expand(v_next)


def explicit_euler_map():
    return sp.expand(x + dt_sym * v), sp.expand(v - dt_sym * x)


def explicit_rk2_map():
    k1x, k1v = v, -x
    xm, vm = x + dt_sym / 2 * k1x, v + dt_sym / 2 * k1v
    return sp.expand(x + dt_sym * vm), sp.expand(v - dt_sym * xm)


def part1_exact_gap_table(dt_values, range_label="full"):
    """Closed-form ||Q_dt - Q_true|| for each integrator, no data involved."""
    integrators = {
        "symplectic_euler": (symplectic_euler_map(), 1),
        "stormer_verlet": (stormer_verlet_map(), 2),
        "explicit_euler": (explicit_euler_map(), None),
        "explicit_rk2": (explicit_rk2_map(), None),
    }
    rows = []
    for name, ((xn, vn), order) in integrators.items():
        sol = exact_conserved_quadratic(xn, vn)
        if sol is None:
            for dt_val in dt_values:
                rows.append({"integrator": name, "order": order, "dt": dt_val,
                             "range": range_label,
                             "has_exact_invariant": False, "gap": np.nan})
            continue
        Bexpr, Cexpr = sol
        Bf = sp.lambdify(dt_sym, Bexpr, "numpy")
        Cf = sp.lambdify(dt_sym, Cexpr, "numpy")
        for dt_val in dt_values:
            b_val = float(Bf(dt_val))
            c_val = float(Cf(dt_val))
            # true invariant: A=1, B=0, C=1
            gap = np.sqrt(b_val**2 + (c_val - 1.0)**2)
            rows.append({"integrator": name, "order": order, "dt": dt_val,
                         "range": range_label,
                         "has_exact_invariant": True, "gap": gap})
    df = pd.DataFrame(rows)

    print(f"\n=== Part 1: exact modified-equation gap (symbolic, no noise), range={range_label} ===")
    for name in df["integrator"].unique():
        sub = df[(df["integrator"] == name) & df["has_exact_invariant"]]
        if len(sub) < 2:
            print(f"  {name}: no exact conserved quadratic form (non-symplectic, energy drifts)")
            continue
        # fit slope on log-log
        logs = np.log(sub["dt"].values)
        logg = np.log(sub["gap"].values)
        slope, intercept = np.polyfit(logs, logg, 1)
        order = sub["order"].iloc[0]
        print(f"  {name} (formal order {order}): fitted log-log slope = {slope:.4f}")
    return df


def _numeric_map(map_fn):
    xn_expr, vn_expr = map_fn()
    xn_f = sp.lambdify((x, v, dt_sym), xn_expr, "numpy")
    vn_f = sp.lambdify((x, v, dt_sym), vn_expr, "numpy")
    return xn_f, vn_f


def _generate_transition_data(xn_f, vn_f, dt_val, N, seed):
    rng = np.random.RandomState(seed)
    x_t = rng.uniform(-2.0, 2.0, N)
    v_t = rng.uniform(-2.0, 2.0, N)
    x_next = xn_f(x_t, v_t, dt_val)
    v_next = vn_f(x_t, v_t, dt_val)
    return np.column_stack([x_t, v_t, x_next, v_next])


def part2a_direct_projection_sweep(dt_values, n_seeds=30, N=5000):
    """Fast, unambiguous empirical check: solve directly for the (B,C)
    coefficients of the conserved quadratic form Q = x^2 + B*xv + C*v^2 from
    SAMPLED transition data, via ordinary least squares on the three
    transition-difference monomials

        (x_t^2 - x_next^2), (x_t*v_t - x_next*v_next), (v_t^2 - v_next^2)

    which are exactly the combinations that vanish identically for a
    conserved quadratic form of this shape. This targets the known
    2-parameter family the theorem is derived in directly, rather than
    handing the full 4-variable/15-monomial dictionary to sr_gb()'s
    combinatorial CSNP/BB search and hoping it selects the physically
    meaningful relation among the many other exact relations that dictionary
    admits on purely deterministic transition data (see
    part2_degeneracy_probe below for a concrete instance of that ambiguity).
    Runs in milliseconds per (dt, seed): a single lstsq on an Nx2 system, no
    search, no ties to break.
    """
    integrators = {
        "symplectic_euler": (symplectic_euler_map, 1),
        "stormer_verlet": (stormer_verlet_map, 2),
    }
    rows = []
    for name, (map_fn, order) in integrators.items():
        xn_f, vn_f = _numeric_map(map_fn)
        for dt_val in dt_values:
            for seed in range(n_seeds):
                data = _generate_transition_data(xn_f, vn_f, dt_val, N, seed)
                x_t, v_t, x_next, v_next = data.T
                col_b = x_t * v_t - x_next * v_next
                col_c = v_t ** 2 - v_next ** 2
                rhs = -(x_t ** 2 - x_next ** 2)
                A_mat = np.column_stack([col_b, col_c])
                (b_hat, c_hat), *_ = np.linalg.lstsq(A_mat, rhs, rcond=None)
                gap = float(np.sqrt(b_hat ** 2 + (c_hat - 1.0) ** 2))
                rows.append({"integrator": name, "order": order, "dt": dt_val,
                             "seed": seed, "b_hat": b_hat, "c_hat": c_hat,
                             "gap": gap})
    df = pd.DataFrame(rows)
    summary = df.groupby(["integrator", "order", "dt"])["gap"].mean().reset_index()

    print("\n=== Part 2a: direct nullspace projection on sampled data "
          "(fast, unambiguous) ===")
    for name in summary["integrator"].unique():
        sub = summary[summary["integrator"] == name]
        logs = np.log(sub["dt"].values)
        logg = np.log(sub["gap"].values)
        slope, intercept = np.polyfit(logs, logg, 1)
        order = sub["order"].iloc[0]
        print(f"  {name} (formal order {order}): fitted log-log slope from "
              f"sampled data = {slope:.4f}")
    return df, summary


def part2_degeneracy_probe(dt_val=0.05, seed=0, N=3000, sigma_estimate=0.0):
    """Illustrates the finding that motivated Part 2a: on purely
    deterministic (noiseless) transition data, sr_gb()'s full 4-variable/
    15-monomial dictionary admits multiple exactly-vanishing degree-2
    relations tied at the same minimal rationality cost, not just H and
    Q_dt. The tie-break (sparsity, then geometric uniqueness) has no reason
    to prefer either physically meaningful relation over these other exact
    algebraic consequences of the deterministic map, so sr_gb() run on the
    unrestricted dictionary can return a relation that is neither. One call,
    ~10-25s (this dictionary has nullspace dimension 9, so BB search here is
    combinatorially heavier than the typical d=1 case elsewhere in this
    repo); not intended to be swept over a grid, just to document the
    instance that motivated Part 2a's narrower, direct-projection design.
    """
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    true_H = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")
    xn_f, vn_f = _numeric_map(symplectic_euler_map)
    data = _generate_transition_data(xn_f, vn_f, dt_val, N, seed)
    gb = sr_gb(data, var_names, degree=2, sigma_estimate=sigma_estimate)
    returned = [g.as_expr() if hasattr(g, "as_expr") else g for g in gb] if gb else []
    is_H = bool(exact_recovery(gb, true_H)) if gb else False
    print(f"\n=== Part 2 degeneracy probe (symplectic Euler, dt={dt_val}, "
          f"seed={seed}, noiseless) ===")
    print(f"  sr_gb() returned: {returned}")
    print(f"  matches true H (x_t^2+v_t^2-x_next^2-v_next^2) up to scalar: {is_H}")
    return {"dt": dt_val, "seed": seed, "returned": [str(r) for r in returned],
            "matches_H": is_H}


def part2b_full_pipeline_with_noise(dt_values, sigma, n_seeds=3, N=3000):
    """Runs the actual sr_gb() pipeline (full 4-variable/15-monomial
    dictionary, CSNP/BB search, snap-rounding), not the direct projection of
    Part 2a, but with representative sampling noise `sigma` added to the
    transition data so the exact-tie degeneracy documented in
    part2_degeneracy_probe is broken the way real (non-deterministic) data
    would break it. sigma_estimate is passed through matching the injected
    sigma, so tau=max(0.05, 3*sigma) is fixed within a run: dt is still the
    only quantity that varies across the sweep.
    """
    var_names = ["x_t", "v_t", "x_next", "v_next"]
    true_invariant = parse_expr("x_t**2 + v_t**2 - x_next**2 - v_next**2")
    integrators = {
        "symplectic_euler": (symplectic_euler_map, 1),
        "stormer_verlet": (stormer_verlet_map, 2),
    }
    rows = []
    for name, (map_fn, order) in integrators.items():
        xn_f, vn_f = _numeric_map(map_fn)
        for dt_val in dt_values:
            for seed in range(n_seeds):
                data = _generate_transition_data(xn_f, vn_f, dt_val, N, seed)
                rng = np.random.RandomState(1000 + seed)
                data = data + rng.normal(0, sigma, data.shape)
                gb = sr_gb(data, var_names, degree=2, sigma_estimate=sigma)
                exact = bool(exact_recovery(gb, true_invariant)) if gb else False
                rows.append({"integrator": name, "order": order, "dt": dt_val,
                             "sigma": sigma, "seed": seed, "srgb_exact_H": exact})
                print(f"  {name:16s} dt={dt_val:<7} seed={seed} exact_H={exact}")
    df = pd.DataFrame(rows)
    summary = (df.groupby(["integrator", "order", "dt"])["srgb_exact_H"]
               .mean().reset_index().rename(columns={"srgb_exact_H": "exact_H_rate"}))
    print(f"\n=== Part 2b: full-pipeline recovery of true H vs dt "
          f"(sigma={sigma}, representative noise) ===")
    print(summary.to_string(index=False))
    return df, summary


if __name__ == "__main__":
    # Full range dt in {0.01, ..., 0.2}: this is the range quoted in the
    # paper's headline slope numbers. Stormer-Verlet's conserved form is an
    # asymptotic series (Equation eq:verlet-invariant), not a pure dt^2
    # power law, so at the top of this range (dt=0.2) the dt^4/16 term is
    # already large enough to bend the fitted slope measurably away from 2.
    # The narrower "leading_order" range stays inside the region where the
    # dt^2 term dominates, isolating the formal order itself. Reporting both
    # rather than only the range that looks cleanest is the point.
    dt_grid_full = np.array([0.01, 0.02, 0.05, 0.1, 0.2])
    dt_grid_leading = np.array([0.001, 0.002, 0.005, 0.01, 0.02])
    exact_df_full = part1_exact_gap_table(dt_grid_full, range_label="full")
    exact_df_leading = part1_exact_gap_table(dt_grid_leading, range_label="leading_order")
    exact_df = pd.concat([exact_df_full, exact_df_leading], ignore_index=True)
    exact_df.to_csv("Results/dt_sweep_exact_gap.csv", index=False)

    # Part 2a: fast, unambiguous check that the closed-form slope also shows
    # up when solved directly from finite-N sampled data (not just symbolic
    # dt substitution). No combinatorial search, no ties.
    dt_grid_2a = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    proj_df, proj_summary = part2a_direct_projection_sweep(
        dt_grid_2a, n_seeds=30, N=5000)
    proj_df.to_csv("Results/dt_sweep_direct_projection_results.csv", index=False)
    proj_summary.to_csv("Results/dt_sweep_direct_projection_summary.csv", index=False)

    # Single illustrative call documenting the exact-tie degeneracy that
    # motivated Part 2a's narrower design (see function docstring). Not a
    # sweep: one call at this dictionary's nullspace dimension (d=9) costs
    # ~10-25s of combinatorial BB search.
    probe = part2_degeneracy_probe(dt_val=0.05, seed=0, N=3000, sigma_estimate=0.0)
    pd.DataFrame([probe]).to_csv("Results/dt_sweep_degeneracy_probe.csv", index=False)
