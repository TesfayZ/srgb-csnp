"""
data_generator.py
Parametric sampling on implicit polynomial varieties p(x) = 0.

Strategy:
1. Solve for one variable symbolically (prefer linear, then quadratic solutions).
2. Use lambdify for vectorised evaluation — O(N) cost, not O(N * sympy_eval).
3. Filter complex solutions.
4. Optionally add Gaussian noise.

This replaces rejection sampling, which fails for measure-zero varieties.
"""

import numpy as np
from sympy import (Symbol, parse_expr, solve, lambdify, sqrt,
                   im, re, Abs, I, simplify)
import warnings
warnings.filterwarnings('ignore')


def generate_variety_data(expr_str, var_names, ranges, N=5000, sigma=0.0, seed=42):
    """
    Generate N points exactly on the variety p(x)=0, then optionally add noise.

    Tries each variable as the one to solve for, in order.
    For each candidate, builds a vectorised numpy function via lambdify,
    samples free variables uniformly, and evaluates the solved variable.

    Parameters
    ----------
    expr_str : str
    var_names : list[str]
    ranges    : dict {var_name: (low, high)}
    N         : int
    sigma     : float
    seed      : int

    Returns
    -------
    X : np.ndarray (n_generated, n_vars)
    """
    np.random.seed(seed)
    syms = {v: Symbol(v) for v in var_names}
    expr = parse_expr(expr_str, local_dict=syms)
    n = len(var_names)

    for solve_idx, solve_var in enumerate(var_names):
        free_vars = [v for v in var_names if v != solve_var]
        free_syms = [syms[v] for v in free_vars]
        free_ranges = [ranges.get(v, (-2.0, 2.0)) for v in free_vars]

        try:
            solutions = solve(expr, syms[solve_var])
        except Exception:
            continue
        if not solutions:
            continue

        # Build lambdified functions for each solution branch
        sol_funcs = []
        for sol in solutions:
            try:
                f = lambdify(free_syms, sol, modules='numpy')
                sol_funcs.append(f)
            except Exception:
                pass
        if not sol_funcs:
            continue

        # Sample free variables — oversample to account for filtering
        oversample = max(N * 4, 20000)
        free_samples = [np.random.uniform(lo, hi, oversample)
                        for lo, hi in free_ranges]

        collected = []
        for f in sol_funcs:
            try:
                vals = f(*free_samples)
                vals = np.asarray(vals, dtype=complex)

                # Keep only real values
                is_real = np.abs(vals.imag) < 1e-8
                vals_real = vals.real[is_real]
                frees_real = [fs[is_real] for fs in free_samples]

                # Filter to valid range for solve_var
                lo, hi = ranges.get(solve_var, (-1e9, 1e9))
                in_range = (vals_real >= lo) & (vals_real <= hi)
                vals_real = vals_real[in_range]
                frees_real = [fs[in_range] for fs in frees_real]

                if len(vals_real) == 0:
                    continue

                # Assemble full-variable array in original order
                M = len(vals_real)
                X_chunk = np.zeros((M, n))
                fi = 0
                for i, v in enumerate(var_names):
                    if v == solve_var:
                        X_chunk[:, i] = vals_real
                    else:
                        X_chunk[:, i] = frees_real[fi]
                        fi += 1
                collected.append(X_chunk)
            except Exception:
                continue

        if not collected:
            continue

        X_all = np.vstack(collected)
        if len(X_all) < N // 2:
            continue  # try next variable

        # Subsample to exactly N
        idx = np.random.choice(len(X_all), min(N, len(X_all)), replace=False)
        X = X_all[idx]

        if sigma > 0:
            X += np.random.normal(0, sigma, X.shape)
        return X

    # Final fallback (should rarely trigger)
    warnings.warn(f"Parametric sampling failed for '{expr_str}', returning empty array.")
    return np.zeros((0, n))


# ──────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import time

    tests = [
        ("x**2 + y**2 - 1",       ["x","y"],          {"x":(-1.5,1.5),"y":(-1.5,1.5)}),
        ("x**2+y**2+z**2-1",      ["x","y","z"],       {"x":(-1.5,1.5),"y":(-1.5,1.5),"z":(-1.5,1.5)}),
        ("F - m*a",                ["F","m","a"],       {"F":(-10,10),"m":(0.5,5),"a":(-5,5)}),
        ("m1*v1 + m2*v2 - P",     ["m1","v1","m2","v2","P"],
                                   {"m1":(0.5,2),"v1":(-2,2),"m2":(0.5,2),"v2":(-2,2),"P":(-5,5)}),
        ("2*KE - m*v**2",          ["KE","m","v"],      {"KE":(0,20),"m":(0.5,5),"v":(-3,3)}),
        ("x**3 + 2*x*y + y**2",   ["x","y"],           {"x":(-2,2),"y":(-2,2)}),
        ("F*r**2 - k*q1*q2",      ["F","r","k","q1","q2"],
                                   {"F":(-20,20),"r":(0.5,3),"k":(1,2),"q1":(-2,2),"q2":(-2,2)}),
    ]

    for expr_str, var_names, ranges in tests:
        t0 = time.time()
        X = generate_variety_data(expr_str, var_names, ranges, N=5000)
        dt = time.time() - t0
        from sympy import parse_expr, Symbol, lambdify
        syms = {v: Symbol(v) for v in var_names}
        p = parse_expr(expr_str, local_dict=syms)
        pf = lambdify([syms[v] for v in var_names], p, 'numpy')
        resid = pf(*[X[:,i] for i in range(len(var_names))])
        resid = np.max(np.abs(resid)) if len(X) > 0 else float('nan')
        print(f"{expr_str[:35]:35s}  N={len(X):5d}  resid={resid:.1e}  t={dt:.3f}s")
