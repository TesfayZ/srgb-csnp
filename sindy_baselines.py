"""
sindy_baselines.py – SINDy baselines with corrected STLSQ target selection.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from sympy import symbols, groebner, Poly, Rational, simplify
import warnings
warnings.filterwarnings('ignore')

from sr_gb import (build_monomial_library, snap_round, estimate_rank,
                   _l1_nullspace_fallback, _column_preconditioned_svd,
                   _orthonormal_null_basis, _tau_resid)

# All baseline SVDs go through the same column-preconditioned front-end
# SR-GB+CSNP uses (_column_preconditioned_svd + _orthonormal_null_basis), so
# the comparison is fair: running the baselines on the raw spectrum instead
# would expose them to the rank-estimation unreliability on large
# cross-monomial dynamic ranges that sr_gb.py's own docstrings document, an
# unreliability the pipeline itself avoids via preconditioning.


def sindy_nullspace(data, var_names, degree, sigma_estimate=0.0,
                    n_bootstrap=15, bootstrap_frac=0.8, max_sparsity=6,
                    Phi=None, monomials=None):
    """
    SINDy-null (KRONIC) – L1 sparsity on the nullspace.

    Pass a prebuilt Phi/monomials pair (with data=None) to run the baseline
    on a purpose-built dictionary, e.g. the difference library
    Phi_old - Phi_new of benchmark_deflation_multi_invariant.py, so it sees
    the IDENTICAL matrix the pipeline sees. Bootstrap resampling operates on
    Phi rows either way.
    """
    if Phi is None or monomials is None:
        sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)
        Phi, _, _ = evaluate(data, noise_sigma=sigma_estimate)
    N, M = Phi.shape
    s, Vt = _column_preconditioned_svd(Phi)
    r = estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d = M - r

    if d <= 1:
        c = Vt[-1, :].copy()
        c[np.abs(c) < 1e-3] = 0.0
        rounded = snap_round(c, sigma_estimate)
        poly = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
        return [poly] if poly != 0 else []

    candidates = []
    rng = np.random.RandomState(0)

    for _ in range(n_bootstrap):
        idx = rng.choice(N, size=int(bootstrap_frac * N), replace=True)
        Phi_b = Phi[idx, :]
        s_b, Vt_b = _column_preconditioned_svd(Phi_b)
        r_b = estimate_rank(s_b, sigma_estimate=sigma_estimate, N=len(idx))
        d_b = M - r_b
        if d_b <= 1:
            continue
        V_null_b = _orthonormal_null_basis(Vt_b, M, d_b)
        tau_resid = _tau_resid(sigma_estimate, len(idx), Phi_b)

        c = _l1_nullspace_fallback(V_null_b, Phi_b, tau_resid)
        if c is None:
            continue
        if np.sum(np.abs(c) > 1e-3) > max_sparsity:
            continue

        c[np.abs(c) < 1e-3] = 0.0
        rounded = snap_round(c, sigma_estimate)
        poly = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
        if poly != 0 and poly not in candidates:
            candidates.append(poly)
    return candidates


def _numeric_rref(V_rows, tol=1e-9):
    """Numeric reduced row-echelon form via Gaussian elimination with
    column-wise partial pivoting, processing columns left to right in the
    caller's fixed order. This is what makes RREF the specific,
    pivot-order-dependent canonicalisation Oellerich & Emelianenko (2024)
    use, unlike a Groebner basis which is order-independent as a set.
    V_rows: (d, M) array whose row space is the nullspace to canonicalise.
    Returns the (d, M) RREF matrix (rows past the discovered rank are left
    as whatever the elimination produced, i.e. all zero for a full-rank
    row space)."""
    A = np.array(V_rows, dtype=float, copy=True)
    d, M = A.shape
    pivot_row = 0
    for col in range(M):
        if pivot_row >= d:
            break
        col_vals = np.abs(A[pivot_row:, col])
        best = int(np.argmax(col_vals))
        if col_vals[best] < tol:
            continue
        best_row = pivot_row + best
        if best_row != pivot_row:
            A[[pivot_row, best_row]] = A[[best_row, pivot_row]]
        A[pivot_row] = A[pivot_row] / A[pivot_row, col]
        for r in range(d):
            if r != pivot_row and A[r, col] != 0.0:
                A[r] -= A[r, col] * A[pivot_row]
        pivot_row += 1
    return A


def rref_nullspace(data, var_names, degree, sigma_estimate=0.0,
                   Phi=None, monomials=None, threshold=1e-3):
    """
    RREF disambiguation (Oellerich & Emelianenko 2024) -- canonicalise the
    same SVD nullspace basis CSNP and SINDy-null consume by reducing it to
    reduced row-echelon form over the library monomials in their build
    order, instead of any sparsity or rationality criterion. For a linear
    ideal this coincides with a Groebner basis (no tie-break needed); this
    baseline tests whether it also disentangles the nonlinear,
    overlapping-monomial-support generators this paper targets.

    Same signature/calling convention as sindy_nullspace: pass a prebuilt
    Phi/monomials pair (data=None) to run on an identical purpose-built
    dictionary, and the identical column-preconditioned SVD front end
    (_column_preconditioned_svd / estimate_rank / _orthonormal_null_basis)
    every other baseline and CSNP itself use, so the comparison is fair.
    Deterministic given Phi: no bootstrap, matching how Oellerich &
    Emelianenko's reduction is a single one-shot canonicalisation of the
    nullspace basis, not a resampled ensemble.
    """
    if Phi is None or monomials is None:
        sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)
        Phi, _, _ = evaluate(data, noise_sigma=sigma_estimate)
    N, M = Phi.shape
    s, Vt = _column_preconditioned_svd(Phi)
    r = estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d = M - r

    if d < 1:
        return []
    if d == 1:
        rows = Vt[-1:, :].copy()
    else:
        V_null = _orthonormal_null_basis(Vt, M, d)  # (M, d)
        rows = V_null.T.copy()  # (d, M): each row one basis vector

    R = _numeric_rref(rows)

    candidates = []
    for row in R:
        c = row.copy()
        c[np.abs(c) < threshold] = 0.0
        if not np.any(c):
            continue
        rounded = snap_round(c, sigma_estimate)
        poly = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
        if poly != 0 and poly not in candidates:
            candidates.append(poly)
    return candidates


def _stlsq(X, y, threshold, max_iter_outer=25):
    """Brunton et al.'s sequential thresholded least squares: unregularized
    least squares alternated with hard thresholding of small coefficients,
    iterated until the active set is stable. This is the STLSQ the paper
    cites, not a Lasso-in-the-loop variant (a different and weaker
    algorithm)."""
    M = X.shape[1]
    active = np.ones(M, dtype=bool)
    coef = np.zeros(M)
    for _ in range(max_iter_outer):
        if not np.any(active):
            break
        c_act, _, _, _ = np.linalg.lstsq(X[:, active], y, rcond=None)
        keep = np.abs(c_act) >= threshold
        idx = np.where(active)[0]
        coef[:] = 0.0
        coef[idx[keep]] = c_act[keep]
        if keep.all():
            break
        active[idx[~keep]] = False
    return coef


def _implicit_st_candidate(Phi, monomials, target_idx, threshold,
                           sigma_estimate, max_iter_outer=25):
    """One implicit-STLSQ fit with monomial `target_idx` as the regression
    target (its coefficient fixed to 1); returns the thresholded,
    snap-rounded sympy polynomial, or None if it collapses to zero."""
    M = Phi.shape[1]
    X = np.delete(Phi, target_idx, axis=1)
    y = -Phi[:, target_idx]
    coef = _stlsq(X, y, threshold, max_iter_outer)
    c = np.zeros(M)
    c[target_idx] = 1.0
    c[np.arange(M) != target_idx] = coef
    c[np.abs(c) < threshold] = 0.0
    rounded = snap_round(c, sigma_estimate)
    poly = sum(Rational(v) * m for v, m in zip(rounded, monomials) if v != 0)
    return poly if poly != 0 else None


def sindy_st(data, var_names, degree, threshold=1e-2, sigma_estimate=0.0,
             max_iter_outer=25, target_idx=None):
    """
    SINDy-ST, single-target form: true STLSQ (lstsq + hard thresholding) in
    the implicit adaptation, regressing one target monomial on the rest.
    The default target is the monomial with the largest column norm.

    NOTE: a single fixed target makes recovery structurally impossible
    whenever the true relation does not contain that monomial (measured:
    23 of 26 Feynman equations). Benchmarks therefore use
    sindy_st_ensemble (all targets) as the standard configuration; this
    single-target entry point is kept only for callers that explicitly
    want the restricted variant.
    """
    sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data, noise_sigma=sigma_estimate)
    if target_idx is None:
        target_idx = int(np.argmax(np.linalg.norm(Phi, axis=0)))
    return _implicit_st_candidate(Phi, monomials, target_idx, threshold,
                                  sigma_estimate, max_iter_outer)


def sindy_st_ensemble(data, var_names, degree, sigma_estimate=0.0,
                      threshold=1e-2, max_sparsity=None, max_iter_outer=25):
    """
    SINDy-ST, standard configuration: all-targets implicit STLSQ in the
    style of SINDy-PI. Every library monomial serves once as the regression
    target; each converged fit is hard-thresholded, snap-rounded, and
    deduplicated up to scalar multiples via its monic representative.

    No sparsity cap by default: in Brunton et al.'s STLSQ the hard
    threshold is the only sparsity mechanism, and an external cap can
    structurally exclude true relations (I.8.14's relation has 7 terms).
    Pass max_sparsity to restrict candidates if a caller wants parity with
    sindy_nullspace's capped bootstrap path.

    Candidate targets are swept rather than regularization strengths against
    ONE fixed target: a single-target Lasso-alpha sweep is structurally unable
    to recover any relation not containing that fixed monomial, and sweeping
    candidate targets is what the published implicit-SINDy family does to avoid
    exactly that failure.
    """
    sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)
    Phi, _, _ = evaluate(data, noise_sigma=sigma_estimate)
    M = Phi.shape[1]
    col_norms = np.linalg.norm(Phi, axis=0)

    candidates = []
    seen = set()
    for j in range(M):
        if col_norms[j] <= 0.0:
            continue
        poly = _implicit_st_candidate(Phi, monomials, j, threshold,
                                      sigma_estimate, max_iter_outer)
        if poly is None:
            continue
        p = Poly(poly, *sym_vars)
        if max_sparsity is not None and len(p.coeffs()) > max_sparsity:
            continue
        key = p.monic().as_expr()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(poly)
    return candidates


def sindy_fd_trajectories(data_trajectories, var_names, degree, dt, sigma_estimate=0.0,
                           threshold=1e-2, max_sparsity=6):
    """
    SINDy-FD (finite differences) – standard practice for dynamics.
    """
    N, n_vars = data_trajectories.shape
    sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)

    Phi_x, _, _ = evaluate(data_trajectories, noise_sigma=sigma_estimate)
    dPhi_dt = np.gradient(Phi_x, dt, axis=0)

    # Drop the trivial constant column (its derivative is exactly 0 everywhere)
    dPhi_dt_nc = dPhi_dt[:, 1:]
    monomials_nc = monomials[1:]
    M_nc = dPhi_dt_nc.shape[1]

    s, Vt = _column_preconditioned_svd(dPhi_dt_nc)
    r = estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d = M_nc - r

    if d <= 1:
        c = Vt[-1, :].copy()
    else:
        V_null = _orthonormal_null_basis(Vt, M_nc, d)
        tau_resid = _tau_resid(sigma_estimate, N, dPhi_dt_nc)
        c = _l1_nullspace_fallback(V_null, dPhi_dt_nc, tau_resid)
        if c is None:
            c = Vt[-1, :].copy()

    c[np.abs(c) < threshold] = 0.0
    if np.sum(np.abs(c) > 1e-3) > max_sparsity:
        return None

    rounded = snap_round(c, sigma_estimate)
    poly = sum(Rational(v) * m for v, m in zip(rounded, monomials_nc) if v != 0)
    return poly if poly != 0 else None


def sindy_ad_trajectories(data_trajectories, var_names, degree, dt, sigma_estimate=0.0,
                           threshold=1e-2, max_sparsity=6, spline_smoothing=None):
    """
    SINDy-AD (lightweight proxy for the principle of denoising before differentiation).

    When spline_smoothing is left at its default (None) and sigma_estimate>0,
    each monomial column is fit with its own error-propagated noise variance
    as a per-point spline weight, rather than one flat smoothing budget
    shared across columns. Same first-order argument as the errors-in-
    variables column correction in sr_gb.py: for a monomial m(x)=prod x_i^e_i
    under additive iid coordinate noise of std sigma_estimate, the delta
    method gives Var(m) ~= sum_i (dm/dx_i)^2 sigma_estimate^2, which grows
    with the column's degree. A single scalar s calibrated for one column's
    noise level under-smooths every higher-degree column and over-smooths
    every lower-degree one; explicit spline_smoothing still overrides this
    (kept for callers that want the old flat-budget behavior).
    """
    N, n_vars = data_trajectories.shape
    sym_vars, monomials, evaluate = build_monomial_library(var_names, degree, min_degree=0, scale=False)

    Phi_x, _, _ = evaluate(data_trajectories, noise_sigma=sigma_estimate)
    t = np.arange(N) * dt
    monoms = [Poly(m, *sym_vars).monoms()[0] if m != 1 else tuple([0] * n_vars)
              for m in monomials]

    def _column_weights(exps):
        grad2 = np.zeros(N)
        for var_idx, e in enumerate(exps):
            if e == 0:
                continue
            partial = np.full(N, float(e))
            for k2, e2 in enumerate(exps):
                power = e2 - 1 if k2 == var_idx else e2
                if power > 0:
                    partial = partial * data_trajectories[:, k2] ** power
            grad2 += partial ** 2
        col_var = np.maximum(grad2 * sigma_estimate ** 2, 1e-12)
        return 1.0 / np.sqrt(col_var)

    dPhi_dt = np.zeros_like(Phi_x)
    for j in range(Phi_x.shape[1]):
        col = Phi_x[:, j]
        if np.allclose(col, col[0]):
            dPhi_dt[:, j] = 0.0
            continue
        try:
            if spline_smoothing is not None:
                spline = UnivariateSpline(t, col, k=3, s=spline_smoothing)
            elif sigma_estimate > 0:
                spline = UnivariateSpline(t, col, k=3, w=_column_weights(monoms[j]))
            else:
                spline = UnivariateSpline(t, col, k=3, s=0.0)
            dPhi_dt[:, j] = spline.derivative()(t)
        except Exception:
            dPhi_dt[:, j] = np.gradient(col, dt)

    # Drop the trivial constant column
    dPhi_dt_nc = dPhi_dt[:, 1:]
    monomials_nc = monomials[1:]
    M_nc = dPhi_dt_nc.shape[1]

    s, Vt = _column_preconditioned_svd(dPhi_dt_nc)
    r = estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d = M_nc - r

    if d <= 1:
        c = Vt[-1, :].copy()
    else:
        V_null = _orthonormal_null_basis(Vt, M_nc, d)
        tau_resid = _tau_resid(sigma_estimate, N, dPhi_dt_nc)
        c = _l1_nullspace_fallback(V_null, dPhi_dt_nc, tau_resid)
        if c is None:
            c = Vt[-1, :].copy()

    c[np.abs(c) < threshold] = 0.0
    if np.sum(np.abs(c) > 1e-3) > max_sparsity:
        return None

    rounded = snap_round(c, sigma_estimate)
    poly = sum(Rational(v) * m for v, m in zip(rounded, monomials_nc) if v != 0)
    return poly if poly != 0 else None