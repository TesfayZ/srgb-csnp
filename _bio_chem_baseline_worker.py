#!/usr/bin/env python3
"""
Worker invoked as a subprocess by bio_chem_attempts.py so that a single slow
(system, method) combination can be killed on a wall-clock timeout without
leaving the whole sweep stuck: numpy/scipy linear algebra calls run in C and
do not yield to Python signal handlers until they return, so an in-process
alarm cannot preempt them, but killing the whole subprocess from the parent
(subprocess.run(..., timeout=...)) can.

Usage: python3 _bio_chem_baseline_worker.py <system_csv> <D_max> <method> <out_json>
"""
import sys
import json
import time
import numpy as np
import pandas as pd
import sympy as sp
import warnings
warnings.filterwarnings('ignore')

from sindy_baselines import sindy_nullspace, sindy_st_ensemble
from avi_baselines import avi_border_basis
from benchmark_omp_nullspace import omp_nullspace


def prepare_design_matrix(df):
    keep_cols = [c for c in df.columns if c not in ("sim_id", "time")]
    return df[keep_cols].values, keep_cols


def as_expr_list(result):
    if result is None:
        return []
    if not isinstance(result, (list, tuple)):
        result = [result]
    exprs = []
    for p in result:
        if p is None:
            continue
        e = p.as_expr() if hasattr(p, "as_expr") else p
        if hasattr(e, "as_expr"):
            e = e.as_expr()
        if e == 0:
            continue
        exprs.append(e)
    return exprs


def residual_stats(expr, X, var_names):
    symbols_ = sp.symbols(var_names)
    try:
        func = sp.lambdify(list(symbols_), expr, modules="numpy")
        vals = func(*[X[:, j] for j in range(X.shape[1])])
        vals = np.asarray(vals, dtype=float)
        if vals.ndim == 0:
            vals = np.full(X.shape[0], float(vals))
    except Exception as e:
        return {"mean": None, "max_abs": None, "note": f"eval error: {e}"}
    abs_v = np.abs(vals)
    return {"mean": float(np.mean(vals)), "max_abs": float(np.max(abs_v))}


def main():
    csv_path, d_max, method, out_json = sys.argv[1:5]
    d_max = int(d_max)
    df = pd.read_csv(csv_path)
    X, var_names = prepare_design_matrix(df)

    fns = {
        "sindy_null": lambda: sindy_nullspace(X, var_names, d_max, sigma_estimate=0.0),
        # All-targets implicit STLSQ ensemble, the standard configuration
        # (the earlier call here was the single-target variant).
        "sindy_st": lambda: sindy_st_ensemble(X, var_names, d_max, sigma_estimate=0.0),
        "omp_nullspace": lambda: omp_nullspace(X, var_names, degree=d_max, sigma_estimate=0.0),
        "avi_border_basis": lambda: avi_border_basis(X, var_names, d_max, sigma_estimate=0.0),
    }

    t0 = time.time()
    try:
        result = fns[method]()
        elapsed = time.time() - t0
        exprs = as_expr_list(result)
        rows = []
        if not exprs:
            rows.append({"generator_idx": None, "expression": None, "status": "abstained"})
        else:
            for i, e in enumerate(exprs):
                stats = residual_stats(e, X, var_names)
                rows.append({"generator_idx": i, "expression": str(e), "status": "returned",
                             **{k: v for k, v in stats.items() if k != "note"}})
        with open(out_json, "w") as f:
            json.dump({"elapsed": elapsed, "rows": rows}, f)
    except Exception as e:
        with open(out_json, "w") as f:
            json.dump({"elapsed": time.time() - t0, "rows": [], "error": str(e)}, f)


if __name__ == "__main__":
    main()
