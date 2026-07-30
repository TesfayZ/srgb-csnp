#!/usr/bin/env python3
"""
Diagnostic (not a paper artifact): measures how the raw SVD nullspace
coefficient vector's deviation from the true polynomial's coefficients grows
with sigma, comparing a degree-2 and a degree-3 Feynman equation on the same
grid of noise levels, to check the Section "Noise Ceiling for Higher-Degree
Invariants" perturbation argument against real data rather than an assumed
O(1) column-scale.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sympy import parse_expr, Symbol, Poly
from sr_gb import build_monomial_library
from data_generator import generate_variety_data
from feynman_polynomials import feynman_polynomials

N = 5000
SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.05]
N_SEEDS = 10

_by_id = {fid: (expr_str, var_names, ranges)
          for fid, expr_str, var_names, ranges in feynman_polynomials}
_circle = _by_id["circle_locus"]
_i122 = _by_id["I.12.2"]

SYSTEMS = {
    "circle_locus (deg 2)": dict(expr_str=_circle[0], var_names=_circle[1],
                           ranges=_circle[2], degree=2),
    "I.12.2 (deg 3)": dict(expr_str=_i122[0], var_names=_i122[1],
                           ranges=_i122[2], degree=3),
}

rows = []
for name, spec in SYSTEMS.items():
    var_names = spec["var_names"]
    syms = {v: Symbol(v) for v in var_names}
    truth = parse_expr(spec["expr_str"], local_dict=syms)
    sym_vars, monomials_sym, evaluate = build_monomial_library(var_names, spec["degree"])

    # Ground-truth coefficient vector in the same monomial ordering.
    poly = Poly(truth, *sym_vars)
    coeff_map = dict(zip(poly.monoms(), poly.coeffs()))
    c_true = np.zeros(len(monomials_sym))
    for i, mon in enumerate(monomials_sym):
        exp = Poly(mon, *sym_vars).monoms()[0]
        c_true[i] = float(coeff_map.get(exp, 0))
    lead = np.argmax(np.abs(c_true))
    c_true_norm = c_true / c_true[lead]

    for sigma in SIGMAS:
        for seed in range(N_SEEDS):
            data = generate_variety_data(spec["expr_str"], var_names, spec["ranges"],
                                          N=N, sigma=sigma, seed=seed)
            Phi, _, _ = evaluate(data)
            U, s, Vt = np.linalg.svd(Phi, full_matrices=False)
            v_null = Vt[-1, :]
            gap_ratio = s[-2] / s[-1] if s[-1] > 1e-14 else np.inf
            v_norm = v_null / v_null[lead]
            err = np.linalg.norm(v_norm - c_true_norm)
            rows.append({
                "system": name, "degree": spec["degree"], "sigma": sigma,
                "seed": seed, "s_min": s[-1], "s_second_min": s[-2],
                "gap_ratio": gap_ratio, "coeff_l2_error": err,
            })
            print(f"{name:16s} sigma={sigma:<6} seed={seed} "
                  f"s_min={s[-1]:.4e} gap_ratio={gap_ratio:8.2f} err={err:.4f}")

df = pd.DataFrame(rows)
df.to_csv("Results/noise_ceiling_probe.csv", index=False)
summary = df.groupby(["system", "degree", "sigma"]).agg(
    mean_coeff_l2_error=("coeff_l2_error", "mean"),
    std_coeff_l2_error=("coeff_l2_error", "std"),
    mean_gap_ratio=("gap_ratio", "mean"),
).reset_index()
summary.to_csv("Results/noise_ceiling_probe_summary.csv", index=False)
print("\n" + "=" * 90)
print(summary.to_string(index=False))
