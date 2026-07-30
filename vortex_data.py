#!/usr/bin/env python3
"""
vortex_data.py - planar point-vortex trajectory generation.

N identical point vortices (unit circulation) evolve under the Kirchhoff
equations of motion in the plane. The motion is Hamiltonian with a
logarithmic (non-polynomial) energy, but it carries three *polynomial*
first integrals from translational and rotational symmetry:

    P = sum_j x_j                  (linear impulse, degree 1)
    Q = sum_j y_j                  (linear impulse, degree 1)
    I = sum_j (x_j^2 + y_j^2)      (angular impulse, degree 2)

These are the recovery targets. The Hamiltonian itself is deliberately not
a target: it is not polynomial.

Why this makes a clean, *dense* benchmark. Each initial condition is
projected onto a common conserved level set before integration: positions
are centred (so P = Q = 0) and rescaled (so I = I0). Every trajectory
therefore shares the same values of P, Q, I, so when trajectories from
different initial conditions are pooled the only low-degree polynomials
that vanish on the whole pool are exactly P, Q, and I - I0 (and their
products). Pooling independent-value trajectories, by contrast, would
cancel the invariants, which is precisely the failure mode that made an
earlier per-simulation-perturbed setup recover nothing. Lifting the pooled
data to degree 3-4 produces a large monomial library with a smoothly
decaying, gap-free singular spectrum: an over-lifted dense regime in which
the three low-degree invariants must be separated from a high-dimensional
nullspace.
"""

import hashlib
import inspect
import json
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

DATA_DIR = "vortex_data"


def _velocity(t, state, n, core):
    """Kirchhoff velocity field for n unit-circulation point vortices.

    ``state`` packs positions as [x_1..x_n, y_1..y_n]. ``core`` is a small
    regularisation added to r^2 so the field stays finite if two vortices
    approach; with the level-set initial conditions used here vortices stay
    well separated and it is numerically negligible.
    """
    x = state[:n]
    y = state[n:]
    dx = np.zeros(n)
    dy = np.zeros(n)
    for j in range(n):
        dxj = xk = 0.0
        for k in range(n):
            if k == j:
                continue
            r2 = (x[j] - x[k]) ** 2 + (y[j] - y[k]) ** 2 + core
            dx[j] += -(y[j] - y[k]) / r2
            dy[j] += (x[j] - x[k]) / r2
    dx /= (2.0 * np.pi)
    dy /= (2.0 * np.pi)
    return np.concatenate([dx, dy])


def _project_to_level_set(pos, I0):
    """Centre positions (P=Q=0) and rescale so sum |r_j|^2 == I0."""
    pos = pos - pos.mean(axis=0)          # unit circulations => centroid = mean
    cur = np.sum(pos ** 2)
    pos = pos * np.sqrt(I0 / cur)
    return pos


def simulate_vortices(n, n_sims=8, time_points=250, t_max=20.0,
                      I0=None, seed=0, core=1e-9):
    """Integrate ``n_sims`` vortex trajectories that share P=Q=0, I=I0.

    Returns a tidy DataFrame with columns sim_id, time, x1..xn, y1..yn.
    """
    if I0 is None:
        I0 = float(n)
    rng = np.random.RandomState(seed)
    t_eval = np.linspace(0.0, t_max, time_points)
    frames = []
    for sim_id in range(n_sims):
        pos = rng.uniform(-1.0, 1.0, size=(n, 2))
        pos = _project_to_level_set(pos, I0)
        state0 = np.concatenate([pos[:, 0], pos[:, 1]])
        sol = solve_ivp(_velocity, (0.0, t_max), state0, t_eval=t_eval,
                        args=(n, core), rtol=1e-11, atol=1e-12,
                        method="DOP853")
        if not sol.success:
            raise RuntimeError(f"integration failed: {sol.message}")
        cols = {"sim_id": sim_id, "time": t_eval}
        for j in range(n):
            cols[f"x{j + 1}"] = sol.y[j]
            cols[f"y{j + 1}"] = sol.y[n + j]
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def _params_fingerprint(n, kwargs):
    """Hash of everything that determines simulate_vortices's output, so a
    cache written under one (n, n_sims, time_points, t_max, I0, seed, core)
    combination is never silently reused for a different one.

    Binds against simulate_vortices's own signature, not just the caller's
    explicit kwargs, so an unspecified argument's default value is part of
    the hash too; otherwise a changed default would go unnoticed by every
    call site that relies on it.
    """
    bound = inspect.signature(simulate_vortices).bind(n, **kwargs)
    bound.apply_defaults()
    payload = json.dumps(dict(bound.arguments), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_or_simulate(name, n, **kwargs):
    """Return cached trajectories for ``name``, simulating + caching if absent
    or if the simulation parameters have changed since the cache was written
    (tracked via a sidecar ``<name>_trajectories.params.json`` fingerprint)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_file = os.path.join(DATA_DIR, f"{name}_trajectories.csv")
    meta_file = os.path.join(DATA_DIR, f"{name}_trajectories.params.json")
    fingerprint = _params_fingerprint(n, kwargs)
    if os.path.exists(csv_file) and os.path.exists(meta_file):
        with open(meta_file) as f:
            cached = json.load(f).get("fingerprint")
        if cached == fingerprint:
            return pd.read_csv(csv_file)
    df = simulate_vortices(n, **kwargs)
    df.to_csv(csv_file, index=False)
    with open(meta_file, "w") as f:
        json.dump({"fingerprint": fingerprint}, f)
    return df


def true_invariants(n, I0=None):
    """Ground-truth polynomial invariants as (label, sympy-expr) pairs."""
    import sympy as sp
    if I0 is None:
        I0 = float(n)
    xs = sp.symbols([f"x{j + 1}" for j in range(n)])
    ys = sp.symbols([f"y{j + 1}" for j in range(n)])
    P = sum(xs)
    Q = sum(ys)
    Ivar = sum(x ** 2 for x in xs) + sum(y ** 2 for y in ys) - sp.Integer(int(round(I0)))
    return [("P=sum x", P), ("Q=sum y", Q), ("I=sum r^2 - I0", Ivar)]


if __name__ == "__main__":
    for name, n in [("vortex3", 3), ("vortex4", 4), ("vortex5", 5)]:
        df = load_or_simulate(name, n)
        x = df[[f"x{j+1}" for j in range(n)]].to_numpy()
        y = df[[f"y{j+1}" for j in range(n)]].to_numpy()
        P = x.sum(1); Q = y.sum(1); I = (x**2 + y**2).sum(1)
        print(f"{name}: N={n} rows={len(df)} "
              f"P range {np.ptp(P):.2e} Q range {np.ptp(Q):.2e} I range {np.ptp(I):.2e}")
