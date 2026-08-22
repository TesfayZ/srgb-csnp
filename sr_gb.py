"""
sr_gb.py – Sparse Regression + Gröbner Basis + Combinatorial Sparse Nullspace Pursuit
(Enhanced with improved BB, full MDL, fast d=2, degree minimality)
"""

import numpy as np
from scipy.linalg import svd, qr, qr_insert
from sympy import (symbols, groebner, Poly, Rational, simplify, cancel, lambdify,
                   factor, div, factor_list, reduced)
from itertools import combinations, combinations_with_replacement, islice
from fractions import Fraction
from collections import defaultdict
import heapq
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
#  Rationality cost (bit‑cost)
# ============================================================================
def rationality_cost(c, max_denom=16, eps=1e-3, support=None):
    """Bit-cost of representing c's entries as small rationals.

    `support`: optional iterable of indices the caller's search has declared
    nonzero (the combinatorial support S). When given, every support entry
    charges AT LEAST 1 bit even if its normalised magnitude falls below eps,
    and entries outside the support are ignored entirely. This restores the
    R >= |S| bound that bb_search's lower bound lb = |S| requires for
    admissibility (Theorem 4.4 step (ii)): without it, a support entry that
    normalises to just under eps contributed 0 bits, R could drop below |S|,
    and the true optimum could be pruned against an incumbent. When support
    is None (dense scoring of a finalized vector), sub-eps entries are
    treated as zeros.
    """
    norm = np.max(np.abs(c))
    if norm < 1e-12:
        return 0.0 if support is None else float(len(list(support)))
    c_norm = c / norm
    indices = range(len(c_norm)) if support is None else support
    total_cost = 0.0
    for j in indices:
        val = c_norm[j]
        if abs(val) < eps:
            if support is not None:
                total_cost += 1.0
            continue
        best_cost = float('inf')
        found = False
        for q in range(1, max_denom + 1):
            p = round(val * q)
            if abs(p / q) < eps:
                continue
            if abs(val - p / q) < eps:
                cost = np.log2(abs(p)) + np.log2(q) + 1.0
                if cost < best_cost:
                    best_cost = cost
                    found = True
        if not found:
            return float('inf')
        total_cost += best_cost
    return total_cost


def _finalized_rational_cost(coeffs):
    """Exact MDL bit-cost of a finalized generator's rational coefficients.

    rationality_cost above scores RAW float vectors, where the intended
    fraction is unknown and the eps window is the honest uncertainty: it
    charges each entry the CHEAPEST fraction within eps. That optimism is
    correct pre-snap but wrong post-snap: a finalized candidate's
    coefficients are exact rationals, and charging 8/15 two bits because 1/2
    sits inside the same noise-widened window hides exactly the awkwardness
    the rationality prior exists to penalise. Measured consequence (the
    algebraic_cubic_toy benchmark entry (feynman_polynomials.py, not a
    verified official Feynman ID), sigma=0.02: the spurious x^2 - xy - 8/15*y^2 scored R=4.0 under
    the window proxy while the true x^3 + 2xy + y^2 scored R=5.0, so the
    sweep's finalized-cost sort picked the junk; under the exact cost below
    they score 9.9 and 4.0 and the truth wins. Same convention as
    rationality_cost (log2|p| + log2 q + 1 bit per nonzero entry), evaluated
    on the actual fractions of the canonical generator. At sigma ~ 0 the
    window is tight enough (Farey-16 spacing ~4e-3 >> eps) to contain only
    the snapped fraction itself, so this equals the proxy there.
    """
    total = 0.0
    for v in coeffs:
        # sympy Rational exposes p/q; QQ domain elements (PythonRational,
        # gmpy mpq) and Fraction expose numerator/denominator.
        p_ = int(getattr(v, 'numerator', getattr(v, 'p', 0)))
        q_ = int(getattr(v, 'denominator', getattr(v, 'q', 1)))
        if p_ == 0:
            continue
        total += float(np.log2(abs(p_)) + np.log2(q_) + 1.0)
    return total

# ============================================================================
#  Snap‑rounding
# ============================================================================
def snap_round(v, sigma_estimate=0.0, max_denom=16, prefer="cost"):
    """prefer="cost" (default): among fractions within the tau band, pick the
    lowest bit-cost one (the MDL prior; published behavior). prefer="nearest":
    pick the fraction closest to the observed coefficient (ties by cost).
    _finalize_candidate retries with "nearest" when the cost-preferred snap
    fails its post-snap degradation gate: at moderate noise the band can
    contain both the true fraction and a simpler wrong one (0.242 with
    tau=0.15 holds both 1/4 and the cheaper 1/3), and only the data residual
    can adjudicate; the retry converts a wrong answer into a right one
    instead of an abstention."""
    v = list(v)
    eps = 1e-3
    v = [0.0 if abs(x) < eps else x for x in v]
    if all(x == 0.0 for x in v):
        return [Fraction(0)] * len(v)
    dom_idx = max(range(len(v)), key=lambda i: abs(v[i]))
    dom = v[dom_idx]
    if abs(dom) < eps:
        return [Fraction(0)] * len(v)
    sign = 1.0 if dom > 0 else -1.0
    normed = [x / (sign * dom) for x in v]
    tau = max(0.05, 3.0 * sigma_estimate)
    # A residual this small is a machine-precision/exact-algebraic match, not
    # merely a noise-plausible approximation, regardless of how loose tau is
    # (tau floors at 0.05 even when sigma_estimate=0). It must be tried
    # separately from the tau-gated scan below: minimum-cost selection among
    # everything that merely clears tau still prefers a simpler LOWER-cost
    # fraction over an exact HIGHER-cost one whenever both clear tau, e.g.
    # c=3/7 at tau=0.05 accepts q=5 (2/5, residual 0.0286) at lower bit-cost
    # than q=7 (3/7, residual ~0), silently rounding an exact 3/7 input to
    # 2/5 even at sigma=0. Exactness must outrank simplicity; simplicity only
    # decides ties among candidates that are actually exact matches (e.g.
    # picking 1/2 over 2/4), and remains the tie-break within the tau band
    # when no exact match exists at all (the genuinely noisy regime).
    eps_exact = 1e-4
    # Denominator window: scan the SAME denominator budget the search oracle
    # certifies (max_denom, default 16), rather than a noise-dependent
    # Q_max = 1/(2*tau) that shrinks as sigma grows (10/8/3 at sigma =
    # 0/0.02/0.05). A shrinking window has two failure modes: an EXACT input
    # like 1/11 can fall outside it and never reach the exact-match track
    # (silent abstention on an in-regime invariant), and under noise a
    # representable coefficient like 1/4 can fall outside it, so the band
    # scan snaps it to a wrong simpler fraction (4x - y at sigma=0.05 coming
    # back as 3x - y). The band's minimum-cost selection still prefers the
    # simplest fraction within tau, so the fixed wide window only ever adds
    # reachable targets; the post-snap degradation gate in _finalize_candidate
    # is what rejects a wrong band pick (abstain rather than answer wrong).
    result = []
    for c in normed:
        if abs(c) < eps:
            result.append(Fraction(0))
            continue
        best_frac = None
        best_key = (float('inf'), float('inf'))
        best_exact_frac = None
        best_exact_cost = float('inf')
        for q in range(1, max_denom + 1):
            p = round(c * q)
            if p == 0 and abs(c) > eps:
                continue
            frac = Fraction(p, q)
            residual = abs(c - float(frac))
            cost = np.log2(abs(p)) + np.log2(q) if p != 0 else 0.0
            if residual < eps_exact and cost < best_exact_cost:
                best_exact_cost = cost
                best_exact_frac = frac
            elif residual < tau:
                key = (residual, cost) if prefer == "nearest" else (cost, residual)
                if key < best_key:
                    best_key = key
                    best_frac = frac
        if best_exact_frac is not None:
            best_frac = best_exact_frac
        elif best_frac is None:
            best_frac = Fraction(float(c)).limit_denominator(max_denom)
        result.append(best_frac)
    if sign < 0:
        result = [Fraction(-1) * r for r in result]
    return result

# ============================================================================
#  Monomial library builder
# ============================================================================
def build_monomial_library(var_names, max_degree, min_degree=0, scale=False):
    n_vars = len(var_names)
    sym_vars = symbols(var_names)
    exponents = []
    for d in range(min_degree, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            exp = [0] * n_vars
            for idx in combo:
                exp[idx] += 1
            exponents.append(tuple(exp))
    monomials_sym = []
    for exp in exponents:
        mon = 1
        for i, e in enumerate(exp):
            if e > 0:
                mon *= sym_vars[i] ** e
        monomials_sym.append(mon)
    zero_exp = tuple([0] * n_vars)

    def evaluate(data, noise_sigma=0.0):
        """noise_sigma > 0 opts in to errors-in-variables column correction.

        With additive iid coordinate noise z = x + e, e ~ N(0, sigma^2), the
        plain monomial columns are BIASED estimates of the true monomials:
        E[z^2] = x^2 + sigma^2, E[z^3] = x^3 + 3 sigma^2 x, and so on. Past
        sigma ~ 0.1 that bias is what breaks snap-rounding: on the circle the
        SVD direction fitted to the biased columns shrinks the squared
        coefficients to ~0.94, the wrong nearest-fraction snap (15/16) then
        beats the true -1 snap on data residual, and recovery collapses (the
        sigma=0.12-0.18 failures seen in the noise ablation). The
        unbiased column estimator is classical errors-in-variables: replace
        z^a by the Hermite-style polynomial h_a(z) defined by h_0 = 1,
        h_1 = z, h_{a+1} = z h_a - a sigma^2 h_{a-1} (so h_2 = z^2 - sigma^2,
        h_3 = z^3 - 3 sigma^2 z, ...), which satisfies E[h_a(x+e)] = x^a
        exactly; independence across coordinates makes the product of
        per-coordinate estimators unbiased for every mixed monomial. This is
        an estimator correction driven by the sigma_estimate the pipeline
        already receives, not a prior: measured on the circle it lifts the
        exact-recovery ceiling from sigma=0.10 to 0.14-0.16 (partial at
        0.18) with no change below sigma=0.10. At noise_sigma = 0 the
        plain power evaluation is used (bit-identical, not merely
        equivalent: the h-recurrence would compute z^e by repeated
        multiplication, which can differ from `z ** e` in the last ulp)."""
        N = data.shape[0]
        Phi = np.ones((N, len(exponents)))
        power_tables = None
        if noise_sigma and noise_sigma > 0:
            max_pow = max((max(exp) for exp in exponents), default=0)
            s2 = float(noise_sigma) ** 2
            power_tables = []
            for var_idx in range(n_vars):
                z = data[:, var_idx]
                H = [np.ones(N), z]
                for a in range(1, max_pow):
                    H.append(z * H[a] - a * s2 * H[a - 1])
                power_tables.append(H)
        for i, exp in enumerate(exponents):
            if exp == zero_exp:
                continue
            val = np.ones(N)
            for var_idx, e in enumerate(exp):
                if e > 0:
                    if power_tables is not None:
                        val = val * power_tables[var_idx][e]
                    else:
                        val *= data[:, var_idx] ** e
            Phi[:, i] = val
        if scale:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler(with_mean=False, with_std=True)
            Phi = scaler.fit_transform(Phi)
            return Phi, monomials_sym, scaler
        else:
            return Phi, monomials_sym, None
    return sym_vars, monomials_sym, evaluate


def build_transition_difference_library(old_data, new_data, var_names, degree,
                                        noise_sigma=0.0):
    """Build ``p(z_t) - p(z_{t+1})`` features for conserved quantities.

    A coefficient vector ``c`` over the returned monomials represents a
    state polynomial ``Q(z)``; ``Phi @ c`` is therefore
    ``Q(z_t) - Q(z_{t+1})``.  This deliberately excludes mixed-time
    monomials such as ``x_t*x_next``, which are the source of the known
    unrestricted-transition-dictionary tie.  It recovers exact invariants
    of the sampled map (for example a symplectic integrator's ``Q_dt``), not
    an unobserved continuous-time Hamiltonian.

    The constant column is removed because it is identically zero in a
    difference library.  The returned ``(sym_vars, monomials, Phi)`` can be
    passed directly to ``sr_gb(data=None, ..., monomials=..., Phi=...)``.
    """
    old_data = np.asarray(old_data)
    new_data = np.asarray(new_data)
    if old_data.ndim != 2 or new_data.ndim != 2:
        raise ValueError("old_data and new_data must be two-dimensional arrays")
    if old_data.shape != new_data.shape:
        raise ValueError("old_data and new_data must have the same shape")
    if old_data.shape[1] != len(var_names):
        raise ValueError("data column count must equal len(var_names)")

    sym_vars, monomials, evaluate = build_monomial_library(
        var_names, degree, min_degree=0, scale=False)
    Phi_old, _, _ = evaluate(old_data, noise_sigma=noise_sigma)
    Phi_new, _, _ = evaluate(new_data, noise_sigma=noise_sigma)
    # min_degree=0 guarantees the constant is the first library feature.
    return sym_vars, monomials[1:], (Phi_old - Phi_new)[:, 1:]

# ============================================================================
#  Rank estimation (gap-based)
# ============================================================================
def estimate_rank(s, sigma_estimate=0.0, N=None, min_gap_ratio=2.0):
    """
    Numerical rank estimate feeding the nullspace dimension d = M - r.

    Hybrid gap + relative-threshold criterion. The paper's original
    criterion (Eq. rank-estimate, Section 3.3) takes
    r = argmax_i (sigma_i/sigma_{i+1}) + 1 over the spectrum. This gap
    heuristic works well when the spectrum has a clear, large elbow (e.g.
    the circle noise ablation, where the gap ratio stays large even at high
    noise). It degrades when the spectrum is FLAT with no genuine elbow --
    the largest ratio may be only ~1.5 -- which arises for some degree-3
    Feynman equations (giving a rank that is too high, null_dim too small,
    and recovery failure) and, more severely, for continuous ODE trajectory
    data lifted to high monomial degree (e.g. Wolf2000 D=4, where the single
    largest ratio ~6x sits between sigma_1 and sigma_2 rather than at a real
    nullspace boundary; see paper Remark 3.2).

    Refinement (guard + fallback): trust the gap only when the largest ratio clears
    min_gap_ratio (default 2.0). Otherwise fall back to a robust relative
    absolute-threshold rank tol = max(1e-8, 1e-6 * sigma_max), the standard
    numerical-rank definition (Golub & Van Loan, already cited), which does
    not depend on the spectrum having any particular shape. When the gap is
    large (circle, well-separated spectra) the guard is never triggered and
    behaviour is identical to the paper's original heuristic; when the gap
    is small (flat degree-3 spectra) the threshold fallback prevents a
    spurious rank estimate. This is consistent with Proposition 4.1,
    which assumes r correctly identifies the true rank of Phi; the guard only
    affects how r is estimated so that a flat spectrum does not fool it.

    Note: on the main discovery path this function does not solely
    determine d. The progressive_nullspace_search sweep (Remark 3.3) tries
    d = 1, 2, 3, ... directly and selects by lexicographic cost, so a single
    imperfect rank estimate here is not catastrophic. estimate_rank is
    still used directly in the expected_support branch and remains available
    as a fast starting-point heuristic.
    """
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return 1
    ratios = s[:-1] / (s[1:] + 1e-300)
    gap_idx = int(np.argmax(ratios))
    if ratios[gap_idx] < min_gap_ratio:
        tol = max(1e-8, 1e-6 * s[0])
        rank = int(np.sum(s > tol))
        # Allow a genuinely full-rank spectrum to report full rank (d = 0);
        # the earlier min(rank, len(s) - 1) clamp manufactured a spurious
        # nullspace direction on full-rank designs and left the downstream
        # constant-polynomial and residual gates to clean up after it.
        return max(1, min(rank, len(s)))
    return gap_idx + 1

def estimate_sigma_from_svd(s, N, M):
    """NOTE: with column preconditioning (the default), the spectrum `s` this
    receives on the main path comes from the column-NORMALISED matrix, so the
    returned value is dimensionless rather than in data units. No caller in
    this repo currently relies on the sigma_estimate=None auto-estimation
    path; if one ever does, convert back to data units (e.g. multiply by the
    median column norm) or estimate from an unscaled spectrum."""
    if len(s) == 0:
        return 1e-6
    tail_frac = 0.30
    k = max(1, int(tail_frac * len(s)))
    tail = s[-k:]
    sigma_est = np.median(tail) / np.sqrt(N)
    return max(sigma_est, 1e-8)

# ============================================================================
#  Column preconditioning (for ps3-style rank-estimation on wide dynamic ranges)
# ============================================================================
def _unscale_Vt(Vt_scaled, col_norms):
    """Undo column preconditioning on a matrix of right-singular-vectors.

    Vt_scaled was computed from a column-normalised copy of Phi (each column
    divided by its L2 norm before the SVD), which is what lets estimate_rank
    see a genuine elbow instead of being fooled by a huge cross-monomial
    dynamic range (e.g. ps3's raw library spans ~23 orders of magnitude
    because `s` reaches ~76000, so `s^2 ~ 6e9` sits next to O(1) terms).
    Column-normalising before the SVD drops the bottom singular value to
    4.5e-16 with a clean gap, giving the correct d*=1 instead of an estimated
    d*=14.

    Every row of Vt_scaled is a vector in SCALED monomial coordinates; dividing
    entry j by col_norms[j] maps it back to REAL monomial coordinates (still
    an exact null/singular direction of the real Phi, since this is just an
    invertible diagonal change of basis). We then renormalise each row back to
    unit L2 norm, purely so every downstream gate calibrated on a unit-scale
    candidate (the rel_pre pre-gate, tau_resid, snap_round's tau band, ...)
    sees the same convention it always has, whether or not preconditioning
    changed the raw magnitude -- renormalising by a positive scalar changes
    nothing about direction or nullness, only the arbitrary overall scale.
    """
    Vt_real = Vt_scaled / col_norms[None, :]
    row_norms = np.linalg.norm(Vt_real, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-300, 1.0, row_norms)
    return Vt_real / row_norms


def _column_preconditioned_svd(Phi):
    """SVD of Phi with columns normalised first; returns (s, Vt) with Vt
    already unscaled back to real monomial coordinates via _unscale_Vt, so
    callers never need to know preconditioning happened.

    Undersampled case (N < M): the thin SVD returns only N right-singular
    rows, which cannot represent the exact nullspace (dimension >= M - N)
    and made downstream Vt[M-d:] slicing crash outright. A full SVD's extra
    M - N rows are exactly those null directions, so we switch to it when
    the matrix is wide."""
    col_norms = np.linalg.norm(Phi, axis=0)
    col_norms = np.where(col_norms < 1e-300, 1.0, col_norms)
    wide = Phi.shape[0] < Phi.shape[1]
    _, s, Vt_scaled = svd(Phi / col_norms, full_matrices=wide)
    return s, _unscale_Vt(Vt_scaled, col_norms)


def _tau_resid(sigma_estimate, N, Phi):
    """Scale-relative residual tolerance for ||Phi @ c|| gates (unit-norm c).

    The earlier absolute form max(1e-4, 3*sigma*sqrt(N)) implicitly assumed
    O(1) monomial columns: on a large-dynamic-range library (the ps3 case)
    even an EXACT invariant's float residual can exceed the absolute 1e-4
    floor, while on tiny-scale data the gate goes vacuous. Multiplying by
    the RMS entry of Phi makes the gate dimensionless (residual per sample
    relative to the typical column magnitude), matching the scale-relative
    convention _finalize_candidate's gates already use, and reproduces the
    old numbers up to that O(1) factor on unit-scale data."""
    phi_rms = np.sqrt(np.mean(Phi ** 2)) if Phi is not None and Phi.size else 1.0
    return max(1e-4, 3.0 * sigma_estimate) * np.sqrt(max(N, 1)) * max(phi_rms, 1e-12)


def _orthonormal_null_basis(Vt, M, d):
    """Slice the d smallest right-singular directions of Vt into an (M, d)
    nullspace basis, column-orthonormalising when needed.

    After _unscale_Vt the rows of Vt are unit norm but no longer mutually
    orthogonal, and on a large dynamic range two unscaled directions can end
    up nearly collinear, which degrades the batched pinning cross-products in
    enumerate_nullspace_generator and the _submatrix_rank gap tests inside
    bb_search/evaluate_support. QR re-orthonormalisation preserves both the
    span and the zero-ROW structure (a coordinate that is zero in every basis
    vector stays zero under column operations), which is all the pinning/rank
    machinery depends on. When the slice is already orthonormal (no
    preconditioning, or trivial column norms) it is returned unchanged, so
    that path stays bit-identical to the pre-preconditioning behaviour.
    """
    V = Vt[M - d:, :].T
    if d <= 1:
        return V
    G = V.T @ V
    if np.max(np.abs(G - np.eye(d))) < 1e-12:
        return V
    return np.linalg.qr(V)[0]

# ============================================================================
#  L1 fallback
# ============================================================================
def _l1_nullspace_fallback(V_null, Phi, tau_resid):
    try:
        from sklearn.linear_model import Lasso
    except ImportError:
        # Keep the public fallback contract independent of scikit-learn.
        # The optional Lasso solve is a sparsity heuristic, not a correctness
        # prerequisite: when it is unavailable, return the least-residual
        # normalised basis direction that clears the same residual gate.  This
        # is deliberately weaker than L1 but gives the routing layer a valid
        # anytime seed instead of silently returning no candidate.
        best_c = None
        best_key = (float('inf'), float('inf'))
        for j in range(V_null.shape[1]):
            c_cand = np.asarray(V_null[:, j], dtype=float).copy()
            norm = np.linalg.norm(c_cand)
            if norm < 1e-10:
                continue
            c_cand /= norm
            resid = np.linalg.norm(Phi @ c_cand)
            if resid >= tau_resid:
                continue
            key = (np.sum(np.abs(c_cand) > 1e-3), resid)
            if key < best_key:
                best_key, best_c = key, c_cand
        return best_c
    M, d = V_null.shape
    best_c = None
    best_sparsity = np.inf
    for fixed_dim in range(d):
        for sign in [1.0, -1.0]:
            other_dims = [j for j in range(d) if j != fixed_dim]
            if len(other_dims) == 0:
                alpha = np.array([sign])
            else:
                target = -sign * V_null[:, fixed_dim]
                X_rest = V_null[:, other_dims]
                lasso = Lasso(alpha=0.05, fit_intercept=False, max_iter=5000)
                lasso.fit(X_rest, target)
                alpha = np.zeros(d)
                alpha[fixed_dim] = sign
                for j, od in enumerate(other_dims):
                    alpha[od] = lasso.coef_[j]
            c_cand = V_null @ alpha
            norm = np.linalg.norm(c_cand)
            if norm < 1e-10:
                continue
            c_cand = c_cand / norm
            resid = np.linalg.norm(Phi @ c_cand)
            sparsity = np.sum(np.abs(c_cand) > 1e-3)
            if resid < tau_resid and sparsity < best_sparsity:
                best_sparsity = sparsity
                best_c = c_cand
    return best_c

# ============================================================================
#  Submatrix rank (robust)
# ============================================================================
def _submatrix_rank(sa, min_gap_ratio=20.0):
    sa = np.asarray(sa)
    if len(sa) == 0:
        return 0
    if len(sa) == 1:
        return 1 if sa[0] > 1e-12 else 0
    floor = max(sa[0] * 1e-12, 1e-300) if sa[0] > 0 else 1e-300
    ratios = sa[:-1] / (sa[1:] + floor)
    gap_idx = int(np.argmax(ratios))
    if ratios[gap_idx] < min_gap_ratio:
        return len(sa)
    return gap_idx + 1


# Hand-set, not fit to any particular benchmark. This is a rank-classification
# margin (paper Table tab:thresholds' rho_min / rho_min^A family), a
# different quantity from the file's residual/coefficient noise tolerances
# (tau_resid, snap_round's tau, the eps floor in sr_gb_fixed all use
# 3.0*sigma), so it is not tied to their convention.
# Sensitivity across a range of multipliers is characterized in
# ablation_noisy_rank_guard.py: the guard is needed at higher noise (rescue
# rate drops well below 100% with it disabled), 3.5 clears the diagnostic's
# own sigma=0.02 case with margin where 3.0 leaves a knife-edge miss, and
# raising k up to 7.0 shows no added over-admission cost on two known-good
# noisy multi-invariant cases.
_NOISY_RANK_GUARD_MULTIPLIER = 3.5


def _conservative_noisy_rank(rank_estimate, sa, d, sigma_estimate):
    """Avoid dominance-pruning a support with a noise-scale near-kernel.

    A false ``HARD_INFEASIBLE`` result removes the support and all of its
    subsets from branch-and-bound.  In noisy data, a small final singular
    value can be a perturbed one-dimensional kernel even when its gap misses
    the deliberately stringent rank threshold.  Such cases are evaluated as
    candidate supports (rank d-1) rather than pruned.  Clearly full-rank
    supports retain the existing fast path.

    The test is relative to the leading singular value, against
    ``_NOISY_RANK_GUARD_MULTIPLIER * sigma_estimate``: an empirical guard,
    not an exact numerical-rank theorem.
    """
    if (rank_estimate != d or d < 2 or sigma_estimate is None
            or sigma_estimate <= 0 or len(sa) == 0):
        return rank_estimate
    rel_smallest = sa[-1] / max(sa[0], 1e-300)
    if rel_smallest <= _NOISY_RANK_GUARD_MULTIPLIER * sigma_estimate:
        return d - 1
    return rank_estimate

# ============================================================================
#  Evaluate support (with primal subset detection)
# ============================================================================
def evaluate_support(S, V_null, d, max_denom, eps, sigma_estimate, Phi, N):
    M = V_null.shape[0]
    notS = [i for i in range(M) if i not in S]
    if not notS:
        return {'status': 'HARD_INFEASIBLE'}

    A = V_null[notS, :]
    # full_matrices=True: when A has fewer rows than columns (M - |S| < d),
    # the thin SVD's last Vt row is NOT a kernel vector (the kernel rows are
    # simply missing), so alpha extraction below would return a non-null
    # direction. The full SVD always carries the complete d-column right
    # basis; for the common tall case it returns the identical Vt.
    Ua, sa, Vta = np.linalg.svd(A, full_matrices=True)
    rankA = _conservative_noisy_rank(
        _submatrix_rank(sa), sa, d, sigma_estimate)

    if rankA == d:
        return {'status': 'HARD_INFEASIBLE'}
    elif rankA <= d - 2:
        # Soft infeasible: try to find a primal subset
        S_list = list(S)
        best_S_star = None
        best_c = None
        best_cost = (float('inf'), float('inf'))
        for r in range(1, len(S_list)+1):
            for combo in combinations(S_list, r):
                S_star = set(combo)
                notS_star = [i for i in range(M) if i not in S_star]
                if not notS_star:
                    continue
                A_star = V_null[notS_star, :]
                Ua2, sa2, Vta2 = np.linalg.svd(A_star, full_matrices=True)
                rankA2 = _conservative_noisy_rank(
                    _submatrix_rank(sa2), sa2, d, sigma_estimate)
                if rankA2 == d - 1:
                    alpha = Vta2[-1, :]
                    c_cand = V_null @ alpha
                    norm = np.linalg.norm(c_cand)
                    if norm < 1e-10:
                        continue
                    c_cand = c_cand / norm
                    R = rationality_cost(c_cand, max_denom, eps, support=S_star)
                    if R == float('inf'):
                        continue
                    k = len(S_star)
                    cost = (R, k)
                    if cost < best_cost:
                        best_cost = cost
                        best_S_star = S_star
                        best_c = c_cand
        if best_S_star is not None:
            return {'status': 'PRIMAL', 'cost': best_cost, 'c': best_c, 'S_star': best_S_star}
        else:
            return {'status': 'SOFT_INFEASIBLE'}
    else:  # rankA == d-1
        alpha = Vta[-1, :]
        c_cand = V_null @ alpha
        norm = np.linalg.norm(c_cand)
        if norm < 1e-10:
            # A genuine null-vector combination exists at this support (rankA ==
            # d-1), it's just degenerate here. This is a fact about this specific
            # S's SVD output, not a structural exclusion of S itself -- unlike the
            # rankA == d case above, it is NOT monotonic under subset/superset, so
            # it must NOT be reported as HARD_INFEASIBLE: bb_search's dominance
            # cache treats HARD_INFEASIBLE as "every subset of this S is also
            # infeasible," which is false here and would wrongly prune subsets
            # that have a perfectly good candidate of their own.
            return {'status': 'SOFT_INFEASIBLE'}
        c_cand = c_cand / norm
        R = rationality_cost(c_cand, max_denom, eps, support=S)
        if R == float('inf'):
            # Same reasoning: a valid null vector exists, it's just not
            # representable as a nice-enough rational at this max_denom. Not
            # dominance-safe -- see comment above.
            return {'status': 'SOFT_INFEASIBLE'}
        k = len(S)
        cost = (R, k)
        return {'status': 'FEASIBLE', 'cost': cost, 'c': c_cand}

# ============================================================================
#  Improved Branch‑and‑Bound with warm‑start and dominance pruning
#  (no backjumping: extract_core below is an identity placeholder)
# ============================================================================
BB_SEARCH_LOG = []  # node-level counters per bb_search() call; see reset_bb_search_log()


def reset_bb_search_log():
    """Clear BB_SEARCH_LOG. Call before a benchmark run whose bb_search()
    node statistics (nodes popped/evaluated/pruned, beam warm-start UB
    quality) are being measured, e.g. benchmark_bb_search_stats.py."""
    BB_SEARCH_LOG.clear()


def bb_search(V_null, d, k_max, max_denom, eps, sigma_estimate, Phi, N, M,
              max_nodes=None):
    # max_nodes: optional ANYTIME bound on nodes evaluated in the exact BB
    # sweep. None (the default, and what every pre-existing caller gets)
    # means run to exhaustion, i.e. the certified exact search. A finite cap
    # returns the best incumbent found so far when exceeded -- used by
    # _recover_circuit's cost-routing so a mispredicted instance cannot stall
    # the pipeline; the beam warm start below means there is usually a decent
    # incumbent long before the cap could bite.
    #
    # Node-level counters are always collected (cheap integer increments)
    # and appended to BB_SEARCH_LOG on return; this does not change search
    # behaviour, only what is recorded about it.
    stats = dict(d=d, capped=False, nodes_popped=0, nodes_evaluated=0,
                 nodes_pruned_lb=0, nodes_pruned_dominance=0,
                 nodes_not_pushed_lb=0,
                 beam_ub_cost=None, final_ub_cost=None)
    UB = (float('inf'), float('inf'))
    best_S = None
    best_c = None
    # Bucketed by |inf_S| rather than a single flat set: S.issubset(inf_S)
    # is only possible when len(S) <= len(inf_S), so a bucketed lookup lets
    # is_dominated skip every bucket smaller than the query instead of
    # scanning the whole cache. With a flat set this call was measured
    # (feynman_polynomials.py's I.13.4/kinetic_energy_scaled entries under
    # noise) making O(|infeasible_cache|)
    # comparisons per BB node -- 111.7M set.issubset calls across 14,953
    # nodes in one 90s window, ~1750s total for that single cell -- with
    # no change in which nodes get pruned, since the set of returned
    # results is identical, only the number of comparisons needed to get
    # each result changes.
    infeasible_by_size = defaultdict(set)
    heap = []

    def update_ub(cost, c, S):
        nonlocal UB, best_S, best_c
        if cost < UB:
            UB = cost
            best_S = S
            best_c = c

    # ---- Warm‑start with beam search ----
    def beam_search(width=20):
        beam = [set()]
        for _ in range(k_max):
            new_beam = []
            for S in beam:
                last = max(S) if S else -1
                for i in range(last+1, M):
                    new_S = set(S)
                    new_S.add(i)
                    result = evaluate_support(new_S, V_null, d, max_denom, eps, sigma_estimate, Phi, N)
                    if result['status'] in ['FEASIBLE', 'PRIMAL']:
                        if result['cost'] < UB:
                            update_ub(result['cost'], result['c'], new_S)
                    # Beam-search HARD_INFEASIBLE finds are deliberately NOT fed
                    # into infeasible_by_size. _submatrix_rank's rank test is a
                    # noise-aware gap heuristic, not an exact linear-algebra rank
                    # test, so its HARD_INFEASIBLE verdict on one support S is not
                    # guaranteed monotonic under subset/superset relations against
                    # a *different* support -- caching it for cross-candidate
                    # dominance pruning during beam search's partial, heuristic
                    # exploration wrongly prunes genuinely feasible subsets (e.g.
                    # exact_flow's true invariant was pruned this way; see
                    # benchmark_dt_discriminator.py). The exact sweep below still
                    # populates infeasible_by_size from its own finds, which is
                    # safe in practice (it explores strictly by non-decreasing
                    # |S|, so a cached entry can only ever match same-size
                    # duplicates, which canonical growth order never produces).
                    new_beam.append(new_S)
            if not new_beam:
                break
            new_beam.sort(key=lambda s: len(s))
            beam = new_beam[:width]

    # Run beam search
    beam_search(20)
    stats['beam_ub_cost'] = UB[0] if UB[0] != float('inf') else None

    # ---- Exact BB with dominance pruning (extract_core is an identity;
    # no conflict-driven backjumping is implemented) ----
    def is_dominated(S):
        len_S = len(S)
        frozen_S = frozenset(S)
        # Equal-size case: S.issubset(inf_S) with len(S)==len(inf_S) holds
        # iff S == inf_S, so this is an O(1) hash-set membership check
        # rather than a scan -- and the same-size bucket is exactly the one
        # that grows largest during the search, so this is the case that
        # matters for performance.
        same_size_bucket = infeasible_by_size.get(len_S)
        if same_size_bucket and frozen_S in same_size_bucket:
            return True
        # Strictly-larger sizes need a real subset check, but the BB queue
        # explores nodes in non-decreasing support size (heap keyed on
        # len(S)), so no size > len_S has been recorded yet during normal
        # traversal; this loop only does anything on constructions that
        # don't preserve that invariant.
        for size in range(len_S + 1, M + 1):
            bucket = infeasible_by_size.get(size)
            if not bucket:
                continue
            for inf_S in bucket:
                if frozen_S.issubset(inf_S):
                    return True
        return False

    def is_symmetric(S):
        return False  # canonical ordering ensures no duplicates

    def extract_core(infeasible_S):
        return infeasible_S  # simple fallback

    heapq.heappush(heap, (0, 0, set()))
    node_id = 1

    while heap:
        if max_nodes is not None and stats['nodes_evaluated'] >= max_nodes:
            # Anytime cap: return the best incumbent found so far. The flag
            # records that this run carries NO exhaustive-search certificate
            # (Theorem 4.4's optimality guarantee assumes exhaustion), so
            # BB_SEARCH_LOG consumers can tell certified runs from capped
            # ones instead of the cap biting silently.
            stats['capped'] = True
            break
        lb, _, S = heapq.heappop(heap)
        stats['nodes_popped'] += 1
        if lb > UB[0]:
            stats['nodes_pruned_lb'] += 1
            continue
        if is_dominated(S):
            stats['nodes_pruned_dominance'] += 1
            continue
        if is_symmetric(S):
            continue

        stats['nodes_evaluated'] += 1
        result = evaluate_support(S, V_null, d, max_denom, eps, sigma_estimate, Phi, N)

        if result['status'] == 'FEASIBLE':
            if result['cost'] < UB:
                update_ub(result['cost'], result['c'], S)
        elif result['status'] == 'HARD_INFEASIBLE':
            frozen_S = frozenset(S)
            infeasible_by_size[len(frozen_S)].add(frozen_S)
            core = extract_core(S)
            if core != S:
                frozen_core = frozenset(core)
                infeasible_by_size[len(frozen_core)].add(frozen_core)
        elif result['status'] == 'PRIMAL':
            if result['cost'] < UB:
                update_ub(result['cost'], result['c'], result['S_star'])

        if len(S) < k_max:
            last = max(S) if S else -1
            for i in range(last+1, M):
                if i in S:
                    continue
                new_S = set(S)
                new_S.add(i)
                new_lb = len(new_S)
                if new_lb <= UB[0]:
                    heapq.heappush(heap, (new_lb, node_id, new_S))
                    node_id += 1
                else:
                    stats['nodes_not_pushed_lb'] += 1

    stats['final_ub_cost'] = UB[0] if UB[0] != float('inf') else None
    BB_SEARCH_LOG.append(stats)

    return best_S, best_c, UB

# ============================================================================
#  Full MDL cost (coefficient + support + degree + residual)
# ============================================================================
def full_mdl_cost(poly_expr, data, sigma_estimate, var_names, D_max, monomials, Phi,
                   max_denom=16):
    from math import comb, log2
    if poly_expr == 0:
        return float('inf')
    sym_vars = symbols(var_names)
    # Coefficient cost
    poly = Poly(poly_expr, *sym_vars, domain='QQ')
    coeffs = np.array([float(c) for c in poly.coeffs()], dtype=float)
    if np.allclose(coeffs, 0):
        return float('inf')
    dom = coeffs[np.argmax(np.abs(coeffs))]
    coeffs_norm = coeffs / dom
    R = rationality_cost(coeffs_norm, max_denom=max_denom, eps=1e-3)
    if R == float('inf'):
        return float('inf')
    # Support cost
    k = len(poly.terms())
    M = len(monomials)
    support_cost = log2(comb(M, k)) if k <= M else 0.0
    # Degree cost: constant log2(D_max) per the paper's Eq. (mdl-cost)
    degree_cost = log2(D_max) if D_max > 0 else 0.0
    # Residual cost: Gaussian negative log-likelihood converted to bits
    # (x log2(e)), so all four MDL terms share one unit (paper Eq. mdl-cost)
    f = lambdify(sym_vars, poly_expr, modules='numpy')
    try:
        vals = f(*[data[:, i] for i in range(len(var_names))])
        residual = (np.sum(vals**2) / (2.0 * sigma_estimate**2) * log2(np.e)
                    if sigma_estimate > 0 else 0.0)
    except Exception:
        residual = float('inf')
    return R + support_cost + degree_cost + residual

# ============================================================================
#  Degree minimality verification
# ============================================================================
def is_degree_minimal(poly_expr, ideal_gens, sym_vars):
    """
    Check if poly_expr is a minimal generator of the ideal <ideal_gens>: there
    is no non-unit polynomial factor f of poly_expr such that poly_expr / f is
    itself a non-constant member of the ideal. Such an f would make poly_expr
    a redundant multiple of a strictly simpler existing generator, e.g.
    y*(x**2+y**2-1) against the ideal <x**2+y**2-1>.

    Uses full ideal membership via Groebner reduction (mirroring
    verification.ideal_membership), not just a literal-factor-list lookup:
    `factor(expr).args` returns the ADDENDS of an Add for irreducible input,
    not multiplicative factors, so a literal-membership check is vacuous for
    the common irreducible case.
    """
    if poly_expr == 0:
        return False
    ideal_gens = [g for g in ideal_gens if g != 0]
    if not ideal_gens:
        return True

    gb = groebner([Poly(g, *sym_vars, domain='QQ') for g in ideal_gens],
                  *sym_vars, order='grevlex')

    def _in_ideal(expr):
        _, rem = reduced(Poly(expr, *sym_vars, domain='QQ').as_expr(),
                         list(gb), *sym_vars, order='grevlex')
        return rem == 0

    _, factors = factor_list(poly_expr, *sym_vars)
    for f, _mult in factors:
        if f.is_Number:
            continue
        quotient = cancel(poly_expr / f)
        if quotient != 0 and not quotient.is_Number and _in_ideal(quotient):
            return False
    return True

def _finalize_candidate(c, Phi, monomials, data, sym_vars, sigma_estimate, eps,
                         max_denom, N, var_names, return_details=False):
    """
    Shared post-processing tail: norm-based residual gate -> thresholding ->
    snap-rounding -> algebraic minimality -> lead-coefficient normalisation
    -> properly-scaled max-residual gate -> Groebner canonicalization.
    Returns None if the candidate is rejected at any stage (signalling the
    caller to try the next candidate), otherwise the same (gb | (gb,c,S,R,k))
    shape sr_gb_fixed has always returned.
    """
    if c is None:
        return None

    # Lenient scale-relative PRE-gate: only rejects directions that are
    # clearly not near the nullspace, cheaply, before the more expensive
    # snap-rounding + minimality steps. It is deliberately loose because a
    # genuine rational invariant's RAW float SVD coefficient vector can carry
    # substantial residual at moderate noise -- snap-rounding is precisely
    # what restores exactness -- so a strict pre-gate here would discard
    # recoverable candidates. The authoritative acceptance test is the
    # scale-relative POST-snap gate further below. We scale the norm residual
    # by the RMS column magnitude of Phi so this pre-gate, like the post-gate,
    # is invariant to the arbitrary overall scale of the data.
    col_rms = np.sqrt(np.mean(Phi ** 2)) if Phi.size else 1.0
    col_rms = max(col_rms, 1e-12)
    rel_pre = np.linalg.norm(Phi @ c) / (np.sqrt(max(N, 1)) * col_rms)
    tau_pre = max(0.5, 20.0 * max(sigma_estimate, 1e-6))
    if rel_pre > tau_pre:
        return None

    c = np.array(c, dtype=float)
    c[np.abs(c) < eps] = 0.0

    # Relative RMS residual scorer, the reference statistic for the snap
    # adjudication driver at the bottom of this function. RMS, not max: at
    # moderate noise the max over thousands of samples is dominated by the
    # noise tail and cannot see a wrong fraction's systematic residual
    # (measured on 4x - y at sigma=0.05: the max statistic differed 4%
    # between the wrong and the right snap while the RMS differed 60%).
    _col_rms_terms = (np.sqrt(np.mean(Phi ** 2, axis=0))
                      if Phi is not None and Phi.size else None)

    def _rel_rms(vec):
        nz = np.abs(vec) > 0
        if _col_rms_terms is None or not np.any(nz):
            return None
        ts = max(np.max(np.abs(vec[nz]) * _col_rms_terms[nz]), 1e-12)
        return float(np.sqrt(np.mean((Phi @ vec) ** 2)) / ts)

    rel_float = _rel_rms(c)

    def _attempt(rounded):
        c_rounded = [Rational(val) for val in rounded]

        poly_expr = sum(coef * mon for coef, mon in zip(c_rounded, monomials) if coef != 0)
        if poly_expr == 0:
            return None
        # A nonzero CONSTANT polynomial (e.g. snap-rounding leaves only the
        # constant monomial with every other coefficient thresholded to zero)
        # can never be a genuine invariant: "constant = 0" has an empty zero
        # locus and is satisfied by no real data, so this is not merely a low-
        # quality candidate but a definitionally invalid one, regardless of how
        # small its residual happens to look under a loose noise tolerance.
        # This arises when a design matrix is actually full column rank (no
        # true relation exists at this degree) but the constant/low-degree
        # monomial happens to have a smaller column norm than the rest, making
        # its singular vector look like a plausible near-null direction.
        if poly_expr.is_number:
            return None

        poly_expr = reduce_to_minimal_generator(poly_expr, data, sigma_estimate, sym_vars=sym_vars)
        if poly_expr == 0:
            return None
        if poly_expr.is_number:
            return None

        poly_expr = poly_expr.expand()
        terms = poly_expr.as_ordered_terms()
        lead_term = None
        max_deg_seen = -1
        for term in terms:
            if term == 0:
                continue
            tdeg = 0 if term.is_Number else int(sum(term.as_powers_dict().values()))
            if tdeg > max_deg_seen:
                max_deg_seen = tdeg
                lead_term = term
        if lead_term is not None:
            lead_coef = lead_term.as_coeff_Mul()[0] if not lead_term.is_Number else lead_term
            if lead_coef != 0:
                poly_expr = simplify(poly_expr / lead_coef)

        # Scale-RELATIVE final gate. A polynomial's residual magnitude scales
        # with the magnitude of its own monomials on the data: the same
        # geometric relation p(x)=0 evaluated on data ranging over [0,5] with
        # squared terms produces residuals ~25x larger than on [0,1] data, even
        # at identical noise. An ABSOLUTE max-residual threshold therefore
        # wrongly rejects a genuine invariant on large-magnitude data while
        # accepting junk on small-magnitude data. We instead require the max
        # residual to be small RELATIVE to the polynomial's own typical term
        # magnitude on the data -- the standard relative-residual criterion,
        # which makes acceptance invariant to the arbitrary overall scale of the
        # variables (this is exactly the scale-invariance the norm-based
        # ||Phi@c|| check enjoys automatically via c's normalisation, applied
        # here for the per-point max check). tau_rel is the dimensionless
        # tolerance derived from the noise level (see the "Scale-Relative
        # Residual Acceptance" note in the paper's Implementation Notes).
        if data is None:
            # Prebuilt-Phi caller (e.g. a difference library Phi_old - Phi_new):
            # the invariant vanishes as a ROW RESIDUAL of Phi, not pointwise on
            # any state data, so both the residual and the characteristic term
            # magnitude are computed from Phi directly. Same gate semantics: max
            # |Phi @ c| must be small relative to the largest per-term magnitude
            # |c_j| * rms(Phi[:, j]).
            cvec = _coeff_vector_over_monomials(poly_expr, monomials, sym_vars)
            if cvec is None:
                return None
            vals = Phi @ cvec
            max_abs_resid = np.max(np.abs(vals))
            col_rms_terms = np.sqrt(np.mean(Phi ** 2, axis=0))
            nz = np.abs(cvec) > 0
            term_scale = (max(np.max(np.abs(cvec[nz]) * col_rms_terms[nz]), 1e-12)
                          if np.any(nz) else 1.0)
        else:
            try:
                f = lambdify(sym_vars, poly_expr, modules='numpy')
                vals = f(*[data[:, i] for i in range(len(var_names))])
                vals = np.asarray(vals, dtype=float)
                max_abs_resid = np.max(np.abs(vals))
            except Exception:
                max_abs_resid = np.inf
                vals = None

            # Characteristic term magnitude: RMS over data of the polynomial's
            # individual monomial terms (coefficient * monomial value), which sets
            # the natural scale a genuine ~0 residual must be small compared to.
            term_scale = 1.0
            try:
                term_mags = []
                for term in poly_expr.as_ordered_terms():
                    if term == 0 or term.is_Number:
                        continue
                    tf = lambdify(sym_vars, term, modules='numpy')
                    tv = np.asarray(tf(*[data[:, i] for i in range(len(var_names))]), dtype=float)
                    term_mags.append(np.sqrt(np.mean(tv ** 2)))
                if term_mags:
                    term_scale = max(np.max(term_mags), 1e-12)
            except Exception:
                # Unlike every other gate here, silently defaulting term_scale
                # to 1.0 can WIDEN the acceptance tolerance rather than narrow
                # it (if the true term magnitudes are < 1). We couldn't verify
                # the characteristic scale, so reject rather than guess, the
                # same treatment already given to a failed max_abs_resid
                # evaluation just above.
                max_abs_resid = np.inf
                term_scale = 1.0

        # Dimensionless noise-derived tolerance: the sqrt(2 ln N) max-vs-noise
        # scaling, applied here as a RELATIVE bound.
        base_rel = max(3.0 * sigma_estimate, 1e-6)
        tau_rel = base_rel * np.sqrt(2.0 * np.log(max(N, 2))) * 3.0
        tau_abs_floor = 1e-6  # allow essentially-exact fits through regardless of scale
        if max_abs_resid > max(tau_abs_floor, tau_rel * term_scale):
            return None

        gb = groebner([Poly(poly_expr, *sym_vars)], *sym_vars, order='grevlex')

        if not return_details:
            return list(gb)

        p_gen = gb[0].as_expr() if hasattr(gb[0], 'as_expr') else gb[0]
        p_poly = Poly(p_gen, *sym_vars, domain='QQ')
        coeff_dict = {}
        for mon, coeff in p_poly.terms():
            m = 1
            for i, e in enumerate(mon):
                if e > 0:
                    m *= sym_vars[i] ** e
            coeff_dict[m] = coeff
        M = len(monomials)
        c_red = np.zeros(M)
        S_red = set()
        for idx, m in enumerate(monomials):
            if m in coeff_dict:
                val = float(coeff_dict[m])
                if abs(val) > 1e-6:
                    c_red[idx] = val
                    S_red.add(idx)
        norm = np.linalg.norm(c_red)
        if norm > 0:
            c_red = c_red / norm
        # Finalized (R, k): the canonical generator's coefficients are exact
        # rationals at this point, so score their EXACT bit-cost rather than
        # the eps-window float proxy rationality_cost applies to raw vectors
        # (see _finalized_rational_cost's docstring for the measured
        # junk-outranks-truth failure the proxy caused at noise-widened eps).
        # Sparsity counts every nonzero coefficient of the generator itself,
        # library-basis or not, which also covers the case where
        # canonicalization produced monomials outside the (pruned) active
        # library and c_red silently truncates it; c_red is still returned as
        # the library-coordinates projection for the deflation machinery,
        # which only uses its direction.
        R = _finalized_rational_cost(coeff_dict.values())
        k = sum(1 for v in coeff_dict.values() if abs(float(v)) > 1e-6)
        return list(gb), c_red, S_red, R, k

    # Snap adjudication (see snap_round's docstring). Build both the
    # cost-preferred and the nearest-fraction snap, score each snapped
    # vector's relative RMS residual on the data, and:
    #   * discard snaps whose residual degrades the float candidate's own
    #     level by more than 1.5x (true snaps measure ~1.01x on the noisy
    #     circle, wrong band picks 1.6-1.8x on the 4x-y / 3x-7y repros;
    #     discarding both means the tau band held no data-consistent
    #     fraction, and abstention beats returning a wrong invariant),
    #   * try the survivors best-residual-first, with the cost-preferred
    #     snap winning near-ties (the MDL simplicity prior only breaks ties
    #     the data cannot decide).
    attempts = []
    seen_snaps = []
    for prefer in ("cost", "nearest"):
        rounded = snap_round(c, sigma_estimate=sigma_estimate,
                             max_denom=max_denom, prefer=prefer)
        if rounded in seen_snaps:
            continue
        seen_snaps.append(rounded)
        vec = np.array([float(f) for f in rounded], dtype=float)
        attempts.append((_rel_rms(vec), rounded))
    if rel_float is not None:
        gate = max(1.5 * rel_float, 1e-6)
        attempts = [a for a in attempts if a[0] is None or a[0] <= gate]
    if len(attempts) == 2 and attempts[0][0] is not None and attempts[1][0] is not None:
        if attempts[1][0] < 0.95 * attempts[0][0]:
            attempts.reverse()
    for _, rounded in attempts:
        result = _attempt(rounded)
        if result is not None:
            return result
    return None


def _thresholded_support_ok(c, S, Phi, sigma_estimate, N, col_rms=None):
    """Is this candidate still a NOISE-CONSISTENT near-null vector once its
    sub-threshold entries are zeroed?

    The in-search (R, k) score is computed on the noise-thresholded support S
    (sub-eps residues are not genuine support; see the `thresh` comment in
    enumerate_nullspace_generator). That score only predicts the finalized
    cost honestly when the THRESHOLDED vector still vanishes on the data at
    the noise level. Two measured failure modes otherwise, both at Feynman
    sigma=0.05: (i) a direction of the form (one large entry + sub-eps fuzz)
    truncates to a single-monomial candidate with O(1) relative residual,
    whose in-search (R, k) = (1, 1) evicts every genuine candidate from the
    single-winner enumeration even though _finalize_candidate then rejects it
    (I.13.4: the fuzz direction displaced the true 2-term circuit at the
    d_try where it was recoverable); (ii) a sparse near-relation whose
    truncated residual is far above the noise level (frequency_shift_linear
    (feynman_polynomials.py, not a verified official ID): f - f0, RMS
    residual ~8 sigma relative to its own term scale) still clears the
    deliberately loose noise-widened final gate and then beats the true
    3-term invariant on sparsity. The guard therefore requires the truncated
    vector's RMS residual, relative to its largest per-term magnitude, to sit
    inside the same 5-sigma noise-consistency band the deflation collection
    loop already uses for accepting extra invariants: a genuine invariant's
    residual is O(sigma) by construction, so this skips only candidates the
    truncation itself has falsified. Candidates skipped here are not ranked
    at all (ranking them under a score finalization cannot distinguish is
    the multiple-comparisons hazard the enumeration docstring warns about);
    abstention is preferred over returning a noise-inconsistent answer."""
    idx = [j for j in S]
    if not idx:
        return False
    c_t = np.zeros_like(c)
    c_t[idx] = c[idx]
    vals = Phi @ c_t
    rms_resid = np.sqrt(np.mean(vals ** 2))
    if col_rms is None:
        col_rms = np.sqrt(np.mean(Phi ** 2, axis=0))
    term_scale = max(np.max(np.abs(c_t[idx]) * col_rms[idx]), 1e-12)
    rel_tol = 5.0 * max(sigma_estimate, 1e-6)
    return rms_resid <= rel_tol * term_scale


_RATIONAL_WINDOW_CACHE = {}


def _rational_window_edges(max_denom, eps):
    """Merged interval edges of the entrywise rationality-acceptance set.

    An entry magnitude v (candidates are max-abs normalised, so v <= 1) passes
    rationality_cost's per-entry test iff v < eps (thresholded away) or v lies
    within eps of some fraction p/q with q <= max_denom and p/q >= eps. That set
    is a union of closed intervals independent of the candidate, so it is built
    once per (max_denom, eps), merged, and cached; membership then reduces to a
    searchsorted parity check.
    """
    key = (int(max_denom), float(eps))
    edges = _RATIONAL_WINDOW_CACHE.get(key)
    if edges is None:
        fracs = sorted({p / q for q in range(1, max_denom + 1)
                        for p in range(1, int(np.ceil(q * (1.0 + eps))) + 1)
                        if p / q >= eps})
        intervals = [(0.0, eps)] + [(f - eps, f + eps) for f in fracs]
        intervals.sort()
        merged = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        edges = np.array([e for ab in merged for e in ab])
        _RATIONAL_WINDOW_CACHE[key] = edges
    return edges


def enumerate_nullspace_generator(V_null, d, k_max, max_denom, eps,
                                  sigma_estimate, Phi, N, budget=200_000,
                                  collect_all=False):
    """CSNP recovery in nullspace-coefficient space (replaces the support-space BB).

    Any candidate invariant is c = V_null @ alpha for alpha in R^d. A coordinate is
    zero iff alpha is orthogonal to that row of V_null, so a vector vanishing on a set
    of d-1 independent rows is pinned up to scale, giving an EXACT nullvector. We
    enumerate the C(M, d-1) size-(d-1) coordinate subsets and score each pinned vector
    by the same (rationality cost, sparsity) test bb_search's oracle applies, keeping
    the least-cost one.

    This is faithful to the paper: near-invariants are rejected exactly as
    `evaluate_support` rejects them, by the rationality of the RAW nullvector -- a
    near-miss such as v_next*x_t - v_t*x_next has irrational raw coefficients so
    rationality_cost is inf. (Snapping the vector before this test would launder that
    irrationality away and wrongly accept it, so the test runs on the raw nullvector.)
    It differs from calling
    `evaluate_support` only in not re-deriving the support: the vanishing set already
    gives the exact nullvector, so we skip evaluate_support's SOFT_INFEASIBLE branch,
    whose 2^|S| sub-enumeration would explode on dense supports. Cost is O(C(M, d-1)),
    polynomial in M for fixed d and INDEPENDENT of the support size k -- which is what
    lets it recover dense invariants that bb_search (exponential in k) cannot.

    Returns the winner coefficient vector (finalised by _finalize_candidate exactly as
    a bb_search result), or None if the enumeration exceeds `budget` (caller then uses
    the L1 fallback). With `collect_all=True` it instead returns a list of
    (cost, c) for every distinct rational support found, sorted by cost, which the
    caller uses for multi-invariant recovery: when several conservation laws hold
    at once (e.g. CRN moieties) each holds as its OWN pinned rational direction in
    this same enumeration, so a linearly independent low-cost subset of them is the
    generating set, recovered without the orthogonal-complement deflation that only
    works when the invariants happen to be orthogonal as coefficient vectors (they
    generally are not, e.g. two affine conservation laws sharing a constant term).

    The enumeration is vectorised: the null direction pinned by a size-(d-1) row
    subset is the generalised cross product of those rows (the vector of signed
    (d-1)-minors), computed for whole chunks of subsets at once with batched
    determinants, and rationality feasibility is prescreened for the whole chunk
    before the exact per-candidate rationality_cost is evaluated on the rare
    survivors. The budget is calibrated in ACTUAL subsets enumerated (each now
    costing microseconds, not the 200-300us a per-subset Python SVD cost), and is
    deliberately modest: beyond it the caller's L1 fallback takes over, which is
    the certified-regime contract the paper states, and enumerating millions of
    pinned directions on a NOISY nullspace is a multiple-comparisons hazard (each
    tested direction is one more chance for a noise-fit to snap to simple
    rationals), so a bounded search is what the selection guarantee assumes, not
    merely a speed concession.
    """
    from math import comb
    M = V_null.shape[0]
    if d <= 1:
        if d != 1:
            return [] if collect_all else None
        c0 = V_null[:, 0]
        if collect_all:
            c0n = c0 / max(np.max(np.abs(c0)), 1e-12)
            S0 = set(np.nonzero(np.abs(c0n) > max(1e-2, eps))[0])
            R0 = rationality_cost(c0n, max_denom, eps, support=S0)
            return [((R0, len(S0)), c0)]
        return c0
    if comb(M, d - 1) > budget:
        return [] if collect_all else None       # over budget -> signal L1 fallback
    # Support threshold for scoring, noise-aware: a pinned in-span candidate
    # carries O(sigma)-scale residue entries on coordinates the underlying
    # invariant does not use (the estimated nullspace only contains the truth
    # up to noise). With a fixed 1e-2 threshold those residues enter S once
    # sigma is large enough (3*sigma > 1e-2), each charging >= 1 bit, so EVERY
    # candidate's (R, k) is inflated by phantom support and the comparison
    # becomes noise-driven: on distance_3d (feynman_polynomials.py, not a
    # verified official ID) at sigma=0.05 the true
    # x^2+y^2+z^2-d^2 (R=4, k=4) lost to a spurious univariate candidate this
    # way. Thresholding at the same noise-widened eps that rationality_cost
    # and _finalize_candidate already use makes the in-search score agree
    # with the finalized score; at sigma ~ 0 (eps <= 1e-2) this reduces to
    # a fixed 1e-2 threshold.
    thresh = max(1e-2, eps)
    best_cost = (float('inf'), float('inf'))
    best_c = None
    found = []                                    # (cost, c) per distinct support
    seen = set()
    col_rms = None                                # lazy, for _thresholded_support_ok
    cols = np.arange(d)
    subset_iter = combinations(range(M), d - 1)
    chunk_size = max(1, min(20_000, budget))
    while True:
        chunk = list(islice(subset_iter, chunk_size))
        if not chunk:
            break
        idx = np.asarray(chunk, dtype=np.intp)               # (K, d-1)
        A = V_null[idx, :]                                    # (K, d-1, d)
        # Generalised cross product: alpha_j = (-1)^j det(A with column j removed).
        # alpha is orthogonal to every row of A, i.e. the pinned null direction;
        # its norm is the (d-1)-volume of the row parallelepiped, so alpha ~ 0
        # exactly when the rows are rank-deficient (the SVD-based skip of the
        # scalar version).
        K = idx.shape[0]
        alpha = np.empty((K, d))
        for j in range(d):
            sub = A[:, :, cols != j]                          # (K, d-1, d-1)
            alpha[:, j] = ((-1.0) ** j) * np.linalg.det(sub)
        row_norms = np.linalg.norm(A, axis=2)                 # (K, d-1)
        vol_scale = np.prod(row_norms, axis=1)                # Hadamard bound on |alpha|
        good = np.linalg.norm(alpha, axis=1) > 1e-9 * np.maximum(vol_scale, 1e-300)
        if not np.any(good):
            continue
        C = alpha[good] @ V_null.T                            # (K', M) candidate vectors
        m = np.max(np.abs(C), axis=1)
        nz = m > 1e-12
        if not np.any(nz):
            continue
        C = C[nz] / m[nz, None]
        # Vectorised rationality prescreen, same test rationality_cost applies
        # entrywise: every |entry| >= eps must sit within eps of a nonzero p/q,
        # q <= max_denom. The set of acceptable magnitudes is a fixed union of
        # intervals around the Farey fractions of order max_denom (plus the
        # below-eps band), so membership is one searchsorted against cached,
        # merged interval edges: an odd insertion index means "inside a window".
        # Candidates failing on ANY entry are discarded in bulk; only the rare
        # all-rational survivors pay the exact Python-level rationality_cost.
        edges = _rational_window_edges(max_denom, eps)
        entry_ok = (np.searchsorted(edges, np.abs(C), side='right') % 2) == 1
        surv = np.nonzero(entry_ok.all(axis=1))[0]
        for i in surv:
            c = C[i]
            S = frozenset(int(j) for j in np.nonzero(np.abs(c) > thresh)[0])
            if not S or S in seen:
                continue
            # If the noise-widened thresh dropped entries the base 1e-2
            # threshold would have kept, the (R, k) score below describes the
            # TRUNCATED candidate; only rank it if that truncation would
            # survive _finalize_candidate's residual gate (see
            # _thresholded_support_ok). Not added to `seen` when skipped.
            if thresh > 1e-2 and len(S) < int(np.sum(np.abs(c) > 1e-2)):
                if col_rms is None:
                    col_rms = np.sqrt(np.mean(Phi ** 2, axis=0))
                if not _thresholded_support_ok(c, S, Phi, sigma_estimate, N,
                                               col_rms=col_rms):
                    continue
            seen.add(S)
            R = rationality_cost(c, max_denom, eps, support=S)
            if R == float('inf'):
                continue
            cost = (R, len(S))
            if collect_all:
                found.append((cost, c.copy()))
            elif cost < best_cost:
                best_cost, best_c = cost, c.copy()
    if collect_all:
        found.sort(key=lambda t: t[0])
        return found
    return best_c


def _coeff_vector_over_monomials(poly_expr, monomials, sym_vars):
    """Coefficient vector of poly_expr in the `monomials` basis, or None if
    poly_expr uses a monomial outside the library (then Phi cannot evaluate it)."""
    try:
        p_poly = Poly(poly_expr, *sym_vars, domain='QQ')
    except Exception:
        return None
    coeff_dict = {}
    for mon, coeff in p_poly.terms():
        m = 1
        for i, e in enumerate(mon):
            if e > 0:
                m *= sym_vars[i] ** e
        coeff_dict[m] = float(coeff)
    cvec = np.zeros(len(monomials))
    for idx, m in enumerate(monomials):
        if m in coeff_dict:
            cvec[idx] = coeff_dict.pop(m)
    if coeff_dict:
        return None  # a term's monomial is not in the library
    return cvec


def _rel_rms_residual(poly_expr, data, sym_vars, Phi=None, monomials=None):
    """RMS residual of poly_expr on the data, relative to its own RMS term scale.

    A genuine invariant vanishes on the data up to noise, so this ratio is O(sigma);
    a polynomial that merely sits NEAR zero over the sampled range (a least-squares
    artefact such as a parabola hugging its minimum, which can slip past the loose
    max-residual gate in _finalize_candidate at moderate noise) has a ratio at the
    scale of the data's own spread instead. Returns inf if evaluation fails.

    With data=None (prebuilt-Phi callers, e.g. a difference library where the
    invariant vanishes as a ROW of Phi rather than pointwise on any state
    data), the same two quantities are computed from Phi directly: residual
    rows are Phi @ c for c the polynomial's coefficient vector over
    `monomials`, and the term scale is max_j |c_j| * rms(Phi[:, j]).
    """
    if data is None:
        if Phi is None or monomials is None:
            return float('inf')
        cvec = _coeff_vector_over_monomials(poly_expr, monomials, sym_vars)
        if cvec is None:
            return float('inf')
        vals = Phi @ cvec
        rms = np.sqrt(np.mean(vals ** 2))
        col_rms = np.sqrt(np.mean(Phi ** 2, axis=0))
        nz = np.abs(cvec) > 0
        scale = np.max(np.abs(cvec[nz]) * col_rms[nz]) if np.any(nz) else 1.0
        return rms / max(scale, 1e-12)
    try:
        f = lambdify(sym_vars, poly_expr, modules='numpy')
        args = [data[:, i] for i in range(len(sym_vars))]
        vals = np.asarray(f(*args), dtype=float)
        rms = np.sqrt(np.mean(vals ** 2))
        term_mags = []
        for term in poly_expr.as_ordered_terms():
            if term == 0 or term.is_Number:
                continue
            tv = np.asarray(lambdify(sym_vars, term, modules='numpy')(*args), dtype=float)
            term_mags.append(np.sqrt(np.mean(tv ** 2)))
        scale = max(term_mags) if term_mags else 1.0
        return rms / max(scale, 1e-12)
    except Exception:
        return float('inf')


_ENUM_BUDGET = 200_000
# Predicted bb work units (sum_k C(M,k) * d^2) below which the primal-side
# exact search is attempted. A budget of 1e6 would be effectively vacuous for
# the default k_max=6: there is essentially no (M, d) where enumeration is
# over ITS budget (C(M, d-1) > 200k needs M and d both sizeable) while
# sum C(M,k<=6) * d^2 still fits in 1e6 -- e.g. M=30, d=7 predicts 594k for
# enumeration but 37.6M for bb, so the routing would always skip straight from
# enumeration to L1 and bb would never fire inside sr_gb. 2e8 units opens a real
# bb regime between the two (M=28/k_max=6/d=12, the over-lifted entangled
# 2-law case, predicts 72M; M=45/k_max=6/d=4 predicts 148M) while still
# refusing the genuinely hopeless corners (M=100/k_max=6 predicts 1.3e9
# already at d=1). Predicted units are only a loose upper bound on nodes
# actually evaluated -- once the beam warm start or an early feasible support
# sets a sparse incumbent, the lb > UB prune discards almost everything (a
# measured M=28/d=12 call runs ~1s) -- and the max_nodes anytime cap bounds
# the mispredicted worst case, which is what makes a generous prediction
# budget safe.
_BB_BUDGET = 200_000_000
_BB_NODE_CAP = 100_000


_UCS_NODE_CAP = 32

UCS_SEARCH_LOG = []  # truncation flags per unified_circuit_search() call; see reset_ucs_search_log()


def reset_ucs_search_log():
    """Clear UCS_SEARCH_LOG. Call before a run whose unified_circuit_search()
    truncation behaviour (collect_top ranked-list truncation, branching-corner
    node-cap exhaustion) is being measured."""
    UCS_SEARCH_LOG.clear()


def unified_circuit_search(V_null, d, Phi, N, M, k_max, max_denom, eps,
                           sigma_estimate, tau_resid, use_bb=True,
                           collect_top=0):
    """One-objective circuit search over the d-dimensional nullspace of Phi.

    Every exact engine in this module already certifies the SAME lexicographic
    objective C(c) = (rationality cost R, sparsity k): enumeration scores
    pinned directions by it, bb_search's evaluate_support oracle scores
    supports by it, and the progressive sweep compares candidates across d by
    it. Enumeration and bb are the two extreme branching policies of ONE
    search tree over per-coordinate zero/nonzero assignments -- dual (pin
    coordinates to zero, cheap for small d at any density) and primal (commit
    support coordinates, cheap for sparse c at any d). They cannot collapse
    into a single parameterisation cheap in both regimes at once (their costs
    are C(M, d-1) and sum_k C(M,k) d^2 respectively), so this function
    unifies the ROUTING and the fallback corner instead:

      * Root fast paths apply the primary cost routing directly:
        enumeration when C(M, d-1) fits its budget, else bb_search when its
        predicted units fit, with the L1 heuristic if an in-budget exact
        engine finds no rational circuit. In these regimes the route is the
        same as calling the corresponding engine standalone. The bb_search
        attempt is gated purely on the predicted-units budget at the
        largest affordable support cap k' <= k_max (no separate monomial-
        count cap): only when even k'=1 fails to fit does routing fall
        through to the branching corner below instead.
      * Only in the corner where BOTH exact engines exceed budget does the
        tree actually branch. A node is
        (Z, S): Z coordinates forced zero (the alpha-space basis V shrinks by
        _complement_basis of the pinned row, so d falls toward the
        enumeration-feasible regime), S coordinates forced nonzero (|S| grows
        toward the bb-feasible regime and yields the lower bound R >= k >= |S|
        that prunes against the incumbent). Each node first retries the two
        bulk engines on its REDUCED instance; every candidate found anywhere
        in the tree is a globally valid nullspace vector, so all of them
        compete as incumbents under the one objective. The L1 vector seeds
        the incumbent as a starting bound.
      * ANYTIME: a node cap (_UCS_NODE_CAP) bounds the branching corner; on
        exhaustion the best incumbent is returned, which is never worse under
        (R, k) than the raw L1 vector alone. Exhausting
        the tree within budget certifies the lexicographic optimum, the same
        certificate Theorem 4.4 gives each engine separately.

    Returns the winning coefficient vector c (in the full M-coordinate
    space), or None if nothing was found at all.

    collect_top > 0 opts in to returning the top-collect_top RANKED list of
    candidate vectors (best in-search (R, k) first) instead of the single
    winner, and only in the root ENUMERATION regime; every other regime
    returns a one-element list. Rationale (measured on algebraic_cubic_toy
    (feynman_polynomials.py, not a verified official Feynman ID) at
    sigma=0.02, seeds 2/3/4): the in-search (R, k) score is a PROXY computed
    on raw float entries, and at noise-widened eps the rational windows are
    wide enough that several spurious directions score 1-bit fractions on
    every entry. A junk direction (there: x*(x^2-xy-8/15*y^2), in-search
    (4.0, 3)) can then outrank the true circuit (in-search (5.0, 3)) for the
    single winner slot even though their FINALIZED costs order the other way
    round ((4, 3) junk vs (1, 3) truth): snap-rounding is what collapses the
    truth's float entries to their cheap fractions, and the proxy cannot see
    that. Handing the sweep the top few candidates lets it adjudicate by the
    finalized cost, the objective the paper actually states, instead of by
    the proxy. This can only ever REPLACE a winner with one that is strictly
    better under the pipeline's own (R, k) ordering.
    """
    from math import comb

    if not use_bb:
        c_l1_only = _l1_nullspace_fallback(V_null, Phi, tau_resid)
        return [c_l1_only] if collect_top else c_l1_only

    if collect_top and comb(M, d - 1) <= _ENUM_BUDGET:
        found = enumerate_nullspace_generator(V_null, d, k_max, max_denom,
                                              eps, sigma_estimate, Phi, N,
                                              collect_all=True)
        if found:
            UCS_SEARCH_LOG.append(dict(d=d, regime='enumeration',
                                       collect_top=collect_top,
                                       n_found=len(found),
                                       capped=len(found) > collect_top))
            return [c for _cost, c in found[:collect_top]]
        c_l1 = _l1_nullspace_fallback(V_null, Phi, tau_resid)
        return [c_l1] if c_l1 is not None else []

    best = {'cost': (float('inf'), float('inf')), 'c': None}

    def consider(c):
        if c is None:
            return
        m = np.max(np.abs(c))
        if m < 1e-12:
            return
        cn = c / m
        # Same noise-aware support threshold as enumerate_nullspace_generator
        # (see comment there): sub-eps residue entries are not genuine
        # support. And the same truncation-validity guard: only rank the
        # candidate on its truncated support if that truncation would survive
        # _finalize_candidate's residual gate.
        thresh = max(1e-2, eps)
        S_nz = set(np.nonzero(np.abs(cn) > thresh)[0])
        if not S_nz:
            return
        if thresh > 1e-2 and len(S_nz) < int(np.sum(np.abs(cn) > 1e-2)):
            if not _thresholded_support_ok(cn, S_nz, Phi, sigma_estimate, N):
                return
        R = rationality_cost(cn, max_denom, eps, support=S_nz)
        if R == float('inf'):
            return
        k = len(S_nz)
        if (R, k) < best['cost']:
            best['cost'], best['c'] = (R, k), c

    def bulk_complete(V, Zmask, dd):
        """Try the two exact engines on the node's reduced instance (rows in
        Z dropped: they are identically zero there and would only produce
        degenerate pinning subsets / dead support choices). Returns True if
        an engine RAN (the node's whole region is then covered), False if
        both were over budget."""
        keep = ~Zmask
        Mk = int(np.sum(keep))
        if dd < 1 or Mk <= dd:
            return True
        Vk = V[keep, :]
        if comb(Mk, dd - 1) <= _ENUM_BUDGET:
            ck = enumerate_nullspace_generator(Vk, dd, k_max, max_denom, eps,
                                               sigma_estimate, Phi[:, keep], N)
            if ck is not None:
                c = np.zeros(M)
                c[keep] = ck
                consider(c)
            return True
        # bb's predicted units sum_k C(M,k) * d^2 are dominated by the
        # LARGEST support size, so a caller-supplied k_max well above the
        # true circuit's support (e.g. the library default k_max=6 when
        # the genuine circuit has 2 terms) can blow the budget on support
        # sizes the answer never needed. Rather than all-or-nothing, run
        # bb at the LARGEST affordable cap k' <= k_max: bb with k_max=k'
        # still finds any true circuit of support <= k' (it just cannot
        # rule out a denser one beyond k'), which keeps genuinely sparse
        # circuits reachable without asking every caller to hand-tune
        # k_max down to the unknown true support size. Measured to matter
        # at e.g. M=45, d=13, k_max=6: the full prediction is ~1.6e9
        # (over budget for full bb) while k'=2 costs
        # ~175k and still covers a 2-term Bezout-style target. No separate
        # Mk cap: this cost loop is O(k_max) regardless of Mk, so the
        # predicted-units budget above is the only gate that should apply.
        k_afford = 0
        cum = 0
        for kk in range(1, k_max + 1):
            cum += comb(Mk, kk)
            if cum * (dd ** 2) <= _BB_BUDGET:
                k_afford = kk
            else:
                break
        if k_afford >= 1:
            _, ck, _ = bb_search(Vk, dd, k_afford, max_denom, eps,
                                 sigma_estimate, Phi[:, keep], N, Mk,
                                 max_nodes=_BB_NODE_CAP)
            if ck is not None:
                c = np.zeros(M)
                c[keep] = ck
                consider(c)
            return True
        return False

    # ---- Root fast path: primary cost routing ----
    # (collect_top's enumeration regime returned above; from here on a
    # requested list is just the single winner wrapped.)
    root_zmask = np.zeros(M, dtype=bool)
    if bulk_complete(V_null, root_zmask, d):
        if best['c'] is not None:
            return [best['c']] if collect_top else best['c']
        # In-budget engine found no rational circuit: fall back to the raw
        # L1 vector.
        c_l1 = _l1_nullspace_fallback(V_null, Phi, tau_resid)
        if collect_top:
            return [c_l1] if c_l1 is not None else []
        return c_l1

    # ---- Branching corner: both exact engines over budget ----
    c_l1 = _l1_nullspace_fallback(V_null, Phi, tau_resid)
    consider(c_l1)

    nodes = 0
    stack = [(V_null, root_zmask, frozenset())]
    while stack and nodes < _UCS_NODE_CAP:
        V, Zmask, S = stack.pop()
        nodes += 1
        dd = V.shape[1]
        if dd == 0:
            continue
        row_norms = np.linalg.norm(V, axis=1)
        # A forced-nonzero coordinate whose row vanished cannot be satisfied.
        if any(row_norms[i] < 1e-12 for i in S):
            continue
        # Lower bound: any c with support >= S has R >= k >= |S| (each
        # nonzero entry costs at least one bit), so this branch cannot beat
        # the incumbent once |S| reaches its R.
        if len(S) > k_max or len(S) >= best['cost'][0]:
            continue
        if dd == 1:
            c = V[:, 0].copy()
            c[Zmask] = 0.0
            consider(c)
            continue
        if bulk_complete(V, Zmask, dd):
            continue
        undecided = [int(j) for j in np.where(~Zmask)[0] if j not in S]
        if not undecided:
            continue
        # Branch on the most informative coordinate (largest alpha-space row).
        j = max(undecided, key=lambda i: row_norms[i])
        # LIFO: push the nonzero-child first so the zero-child (which shrinks
        # d toward enumeration feasibility) is explored first.
        stack.append((V, Zmask, S | {j}))
        Cp = _complement_basis(V[j, :])
        if Cp is not None:
            Zm2 = Zmask.copy()
            Zm2[j] = True
            stack.append((V @ Cp, Zm2, S))

    UCS_SEARCH_LOG.append(dict(d=d, regime='branching', nodes=nodes,
                               node_cap=_UCS_NODE_CAP, capped=bool(stack)))
    c_win = best['c'] if best['c'] is not None else c_l1
    if collect_top:
        return [c_win] if c_win is not None else []
    return c_win


def _recover_circuit(V_null, d, Phi, N, M, k_max, max_denom, eps, sigma_estimate,
                     tau_resid, use_bb=True, collect_top=0):
    """Min-cost rational circuit of the d-dimensional nullspace spanned by
    V_null, under the one lexicographic (R, k) objective. Thin wrapper over
    `unified_circuit_search` (see its docstring for the routing/branching
    semantics, and for what collect_top returns); kept as the
    pipeline-internal seam every caller goes through.
    """
    return unified_circuit_search(V_null, d, Phi, N, M, k_max, max_denom, eps,
                                  sigma_estimate, tau_resid, use_bb,
                                  collect_top=collect_top)


def _same_poly_up_to_scalar(p1, p2):
    """True when p1 and p2 generate the same principal ideal (ratio constant)."""
    if p1 is None or p2 is None:
        return False
    try:
        ratio = cancel(p1 / p2)
        return bool(ratio.is_number or ratio.is_constant())
    except Exception:
        return p1 == p2


def _complement_basis(w):
    """Orthonormal basis (d x (d-1)) of the hyperplane orthogonal to w in R^d,
    via the SVD of the 1 x d matrix w/||w||: its only nonzero right-singular
    direction IS w's direction, so the remaining d-1 rows of Vt span exactly
    the orthogonal complement. Returns None for a (near-)zero w.

    This one operation serves BOTH sides of the circuit search: deflation
    (remove the direction alpha of an accepted circuit from the alpha-space)
    and coordinate pinning (restrict the alpha-space to {alpha : row_j.alpha
    = 0}, i.e. force coefficient j to zero) are each 'take the complement of
    one vector in alpha-space'.
    """
    nw = np.linalg.norm(w)
    if nw < 1e-300:
        return None
    _, _, Vt1 = np.linalg.svd((w / nw).reshape(1, -1))
    return Vt1[1:, :].T


def _deflate_basis(Vb, c):
    """Return an ORTHONORMAL basis of the subspace of span(Vb) orthogonal to
    the accepted circuit `c` in the standard monomial-space (ambient R^M)
    inner product.

    This must be a monomial-space orthogonal projection, not a projection in
    Vb's own (possibly oblique) coordinates: removing c's COORDINATE direction
    alpha (c = Vb @ alpha) only guarantees span{c} is gone, with no guarantee
    of preserving some OTHER genuine invariant w in span(Vb) even when w is
    exactly orthogonal to c as monomial coefficient vectors (e.g. two
    conservation laws with disjoint supports). Coordinate-space and
    monomial-space orthogonality agree only when Vb is an isometry, which
    column preconditioning deliberately makes it not be in general; measured
    directly on the entangled x0-x1/x2-x3 repro, the second invariant's
    reconstruction residual from an alpha-space-deflated oblique basis was
    ~0.08, silently unrepresentable. So: QR-orthonormalise Vb into Q (same
    span, now an isometry); c's coordinates Q.T @ (c/||c||) are then exact,
    and removing that coordinate direction (via _complement_basis, on a
    genuine isometry) preserves every monomial-space-orthogonal vector of
    span(Vb) to floating-point precision. Callers that already pass an
    orthonormal basis (the pipeline does, via _orthonormal_null_basis and the
    fact that Q @ Cperp stays orthonormal) get identical behaviour; the
    internal QR makes the function safe for ANY basis.
    """
    d = Vb.shape[1]
    if d <= 1:
        return np.zeros((Vb.shape[0], 0))
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-300:
        # Degenerate: nothing meaningful to remove -- signal the caller to
        # stop rather than silently returning an arbitrary basis.
        return np.zeros((Vb.shape[0], 0))
    Q = np.linalg.qr(Vb)[0]               # (M, d) orthonormal, same span as Vb
    q_coords = Q.T @ (c / norm_c)         # exact when c lies in span(Q)
    if np.linalg.norm(q_coords) < 1e-8:
        # c has ~no component describable by this basis -- should not happen
        # for a genuine circuit of span(Vb), but be defensive.
        return np.zeros((Vb.shape[0], 0))
    Cperp = _complement_basis(q_coords)
    if Cperp is None:
        return np.zeros((Vb.shape[0], 0))
    return Q @ Cperp                      # (M, d-1), orthonormal


def _collect_additional_invariants(V_full, c_primary, existing_polys, Phi, N, M,
                                   k_max, max_denom, eps, sigma_estimate,
                                   monomials, data, sym_vars, var_names, use_bb,
                                   max_extra=None):
    """Full-nullspace deflation: the mechanism that makes bb_search and
    enumeration genuinely complementary across an ENTIRE nullspace rather than
    at one fixed d_try.

    Starting from the nullspace basis V_full (M x d_star, d_star a generous
    nullspace-dimension estimate) and the already-accepted primary invariant
    c_primary (a vector in, or near, span(V_full)), repeatedly:
      1. project the last-accepted circuit's direction out of the working
         basis (_deflate_basis, a MONOMIAL-SPACE orthogonal projection; see
         its docstring), shrinking it by one dimension;
      2. ask `_recover_circuit` for the lowest-cost circuit of the smaller
         residual basis; it cost-routes between enumeration (cheap when the
         residual dimension is small) and bb_search (cheap when the invariant
         is sparse even though the residual dimension is large, the
         loop-invariant regime), else L1;
      3. accept the candidate only if it clears the SAME finalisation gate
         (_finalize_candidate: snap-rounding, algebraic minimality, the
         scale-relative residual gate) and the same data-vanishing residual
         gate every other candidate in this module must clear, AND is not
         already implied by the ideal generated by what has been accepted so
         far (a Groebner-reduction membership test). The membership test
         matters beyond simple duplicates: a monomial MULTIPLE of an
         already-accepted relation (e.g. x2*x4 - x3*x4, an exact consequence
         of a known x2 - x3 = 0) is a genuine exact circuit that clears
         every other gate but is not a NEW invariant; with data present,
         reduce_to_minimal_generator usually factors it back down first, but
         with data=None (prebuilt-Phi difference libraries) no reduction
         happens and only ideal membership catches it;
      4. deflate the visited direction out either way (accepted or
         redundant) and continue; stop when recovery or finalisation fails,
         or the basis is exhausted. A generous d_star is safe because these
         gates are what reject noise directions, not the dimension estimate
         itself.

    The deflation step uses the RAW circuit vector `c` (by construction
    exactly in the current working basis's span, so each step shrinks the
    basis by exactly one dimension), never the finalised/reduced c_red: a
    reduced vector's direction was typically already removed in an earlier
    step, and deflating by it again would hit _deflate_basis's degenerate
    branch and collapse the basis to nothing, silently discarding genuinely
    independent directions not yet visited. The initial deflation of
    c_primary (the snapped primary, in or near span(V_full)) is the one
    place a non-raw vector is used, which is why _deflate_basis tolerates
    vectors only approximately in span.

    This is NOT gated by the enumeration
    budget: once C(M, d-1) exceeds budget, `_recover_circuit` itself falls
    through to bb_search (or L1), so bb_search actually fires here for the
    high-d* / sparse regime instead of the collection being skipped outright.
    """
    d_star = V_full.shape[1]
    if d_star < 2:
        return []
    if max_extra is None:
        # No real benchmark in the paper needs more than a handful of
        # simultaneous invariants (the densest documented case is the 2D
        # oscillator's 4); cap generously above that so a runaway nullspace
        # dimension estimate cannot turn this into an unbounded loop, while
        # comfortably covering the sparse loop-invariant cases (d* up to ~14
        # documented for ps3 pre-preconditioning) PLUS however many redundant
        # monomial-multiple directions of each accepted relation get visited
        # and deflated along the way.
        max_extra = min(d_star - 1, 40)
    tau_resid = _tau_resid(sigma_estimate, N, Phi)
    rel_tol = 5.0 * max(sigma_estimate, 1e-6)

    known = [p for p in existing_polys if p is not None]
    Vb = _deflate_basis(V_full, c_primary)

    extra_polys = []
    for _ in range(max_extra):
        d_cur = Vb.shape[1]
        if d_cur < 1:
            break
        if d_cur == 1:
            c = Vb[:, 0]
        else:
            c = _recover_circuit(Vb, d_cur, Phi, N, M, k_max, max_denom, eps,
                                 sigma_estimate, tau_resid, use_bb)
        if c is None:
            break
        detailed = _finalize_candidate(c, Phi, monomials, data, sym_vars,
                                       sigma_estimate, eps, max_denom, N,
                                       var_names, return_details=True)
        if detailed is None:
            break
        gb_i = detailed[0]
        p_i = gb_i[0].as_expr() if hasattr(gb_i[0], 'as_expr') else gb_i[0]
        if _rel_rms_residual(p_i, data, sym_vars, Phi=Phi, monomials=monomials) > rel_tol:
            break
        # Ideal-membership redundancy test (Groebner reduction against the
        # accepted-so-far generators); scalar-ratio comparison as fallback if
        # the Groebner computation itself fails.
        try:
            known_gb = groebner([Poly(q, *sym_vars) for q in known],
                                *sym_vars, order='grevlex')
            is_redundant = (known_gb.reduce(Poly(p_i, *sym_vars))[1] == 0)
        except Exception:
            is_redundant = any(_same_poly_up_to_scalar(p_i, q) for q in known)
        if not is_redundant:
            extra_polys.append(p_i)
            known.append(p_i)
        if d_cur == 1:
            break
        Vb = _deflate_basis(Vb, c)
    return extra_polys


def progressive_nullspace_search(Phi, s, Vt, k_max, max_denom, eps,
                                  sigma_estimate, N, M, use_bb=True,
                                  d_min=1, d_max_search=None,
                                  monomials=None, data=None, sym_vars=None,
                                  var_names=None, return_details=False):
    """
    Progressive nullspace-dimension search (see paper Remark 3.3).

    Rather than committing to a single upfront estimate of the nullspace
    dimension d from the SHAPE of the singular-value spectrum -- which
    Remark 3.2 shows can fail in both directions:
      - under-estimating r / over-estimating d catastrophically when the
        spectrum decays smoothly with no genuine elbow anywhere (continuous
        ODE trajectory data at high monomial degree, e.g. Wolf2000 D=4:
        M=715, the naive heuristic reports d=714 and bb_search's cost,
        quadratic in d, becomes intractable for no real gain), and
      - over-estimating the RELEVANT d for noisy data, where several
        noise-dominated singular directions get folded into one nullspace
        estimate alongside the true invariant's direction, which breaks
        the rank-based feasibility classification bb_search relies on
        (e.g. circle_locus (feynman_polynomials.py, not a verified official
        Feynman ID) at sigma=0.02, masked this way even though the
        true 3-term support passes every check once tested in isolation)

    this tries d = 1, 2, 3, ... directly. For each d_try, the smallest
    d_try singular vectors are taken as a candidate nullspace basis, CSNP+BB
    is run on exactly that candidate (via _recover_circuit ->
    unified_circuit_search: direct enumeration when the dual cost fits its
    budget, bb_search when its predicted units fit, else the anytime
    branching corner seeded by L1), and the FULL finalization pipeline (rounding,
    algebraic minimality, the properly-scaled final residual gate) is run
    on the result before it is accepted. This last point matters: accepting
    the first d_try whose PRE-rounding candidate clears only a preliminary
    check can commit to a spurious, lower-quality candidate (e.g. a trivial
    single-variable near-fit) before the correct, higher-d candidate is ever
    tried, especially if the final residual gate is too permissive. With
    finalization performed per-candidate and the scale-relative residual gate
    applied (see the Implementation Notes), the first d_try that survives the
    FULL pipeline is genuinely the simplest correct answer, not merely the
    simplest ATTEMPTED one.

    This changes nothing about Proposition 4.1, Lemma 4.2, or Theorem 4.4,
    which are statements about what happens GIVEN a correct nullspace basis
    of dimension d -- it only changes how d and that basis are searched
    for. It is the same incremental "try simpler candidates first, accept
    the first that survives full validation" strategy Proposition 4.5 (Conditional Adaptive Exact Recovery) already
    uses for degree D (Section 3.10), now applied one level down to d.
    """
    from math import comb
    d_try_jump = None
    if d_max_search is None:
        # The paper's benchmarks never exhibit a true nullspace dimension
        # above 4 (the 2D oscillator in the Transition Invariants
        # experiments), and the Deflation Numerics limitation recommends
        # no more than 2-3 deflation steps. Cap the progressive scan at a
        # small constant so the search stays fast. For LARGE monomial
        # libraries (high degree / many variables) each candidate d is
        # expensive, and genuine invariants there are low-d, so we shrink the
        # sweep as M grows: the extra high-d probes only pay off on the small
        # libraries (e.g. Feynman degree-2, M<=15) where noise robustness
        # matters and each probe is cheap.
        if M <= 20:
            d_cap = max(6, k_max)
        elif M <= 60:
            d_cap = 4
        else:
            d_cap = 2
        # Noise-aware: the wide high-d sweep exists to handle NOISE, where the
        # true invariant's null direction gets mixed with noise-dominated
        # directions and only separates out at the correct d. For essentially
        # noiseless data the invariant is a clean exact null direction found at
        # low d, and a wide sweep merely wastes time and risks selecting a
        # spurious dense near-null direction from an overcomplete library, so
        # we keep the sweep narrow when sigma ~ 0.
        if (sigma_estimate is None) or (sigma_estimate <= 1e-9):
            d_cap = min(d_cap, 3)
        d_max_search = min(M - 1, d_cap)
        # Also honour the hybrid estimate_rank estimate (bounded extension).
        try:
            d_est = M - estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
            if 1 <= d_est <= d_max_search + 2:
                d_max_search = max(d_max_search, min(d_est, d_max_search + 2))
            elif M <= 20 and d_est > d_max_search:
                # The sigma~0 cap above assumes noiseless data has a clean,
                # LOW-d exact nullspace, which fails when monomials are
                # exactly functionally dependent (e.g. fixed-dt transition
                # pairs where x_next, v_next are exact functions of x_t, v_t):
                # the design matrix can then have a genuinely large exact
                # nullspace (d_est far above the cap) even at sigma=0. Trust
                # a rank estimate that clearly indicates a larger exact
                # nullspace rather than silently giving up, but JUMP straight
                # to d_est instead of also visiting every intermediate d_try:
                # only the full null space is guaranteed to contain the exact
                # relation in this degenerate case (any smaller subset of
                # singular vectors is an arbitrary, generally non-aligned
                # basis for part of the degenerate space), so the
                # intermediate values are both unlikely to succeed and, once
                # within budget, no longer cheap to try exhaustively.
                d_try_jump = min(d_est, M - 1)
        except Exception:
            pass
    tau_resid = _tau_resid(sigma_estimate, N, Phi)

    candidates = []  # (cost_tuple, result) collected across all d_try

    d_try_values = list(range(d_min, d_max_search + 1))
    if d_try_jump is not None and d_try_jump not in d_try_values:
        d_try_values.append(d_try_jump)

    for d_try in d_try_values:
        # (M, d_try) basis from the smallest d_try singular directions;
        # re-orthonormalised if column preconditioning left it skewed.
        V_null_try = _orthonormal_null_basis(Vt, M, d_try)

        if d_try == 1:
            cs = [V_null_try[:, 0]]
        else:
            # Cost-routed matroid-circuit recovery (enumeration / bb_search / L1).
            # collect_top: in the enumeration regime the in-search (R, k) is a
            # proxy score on raw float entries, and at noise-widened eps it can
            # rank a junk direction above the true circuit even though their
            # FINALIZED costs order the other way (see unified_circuit_search's
            # docstring for the measured algebraic_cubic_toy case). So take the top few
            # ranked candidates and let the finalized cost, the objective the
            # sweep already sorts by, adjudicate among them.
            cs = _recover_circuit(V_null_try, d_try, Phi, N, M, k_max,
                                  max_denom, eps, sigma_estimate, tau_resid,
                                  use_bb, collect_top=5)

        best_R_here, best_k_here = None, None
        for c in cs:
            if c is None:
                continue
            # Always finalize with details so we can score every surviving
            # candidate by the paper's lexicographic (R, k) cost, regardless of
            # what the caller ultimately wants returned.
            detailed = _finalize_candidate(c, Phi, monomials, data, sym_vars,
                                            sigma_estimate, eps, max_denom, N,
                                            var_names, return_details=True)
            if detailed is None:
                continue
            gb_list, c_red, S_red, R, k = detailed
            # Lexicographic cost tuple (rationality >> sparsity), the SAME order
            # bb_search / Theorem 4.4 use within a fixed d; here we extend it
            # ACROSS candidate d, mirroring how Section 3.10's MDL selection
            # picks the best degree D. A clean low-rationality-cost invariant
            # (all small integer coefficients) beats a spurious noise-fit that
            # only snap-rounds to awkward fractions.
            cost = (R if R is not None else float('inf'), k if k is not None else float('inf'))
            candidates.append((cost, detailed, d_try, V_null_try))
            if best_R_here is None or cost < (best_R_here, best_k_here):
                best_R_here, best_k_here = cost
            # Same R <= k rationale as the cross-d exit below, applied within
            # the ranked list: a finalized candidate at the 1-bit-per-
            # coefficient floor for its sparsity cannot be beaten by a later,
            # WORSE-proxy-ranked candidate except through lower k, which the
            # remaining junk-prone candidates in this list do not offer in
            # practice. Keeps the extra finalizations (and their Groebner
            # calls) off the sigma ~ 0 fast path, where the exact circuit
            # ranks first and stops the list immediately.
            if R is not None and k is not None and R <= k + 1e-9:
                break

        if best_R_here is None:
            continue

        # Early-exit HEURISTIC: R == k means every coefficient costs the
        # minimum 1 bit (simple units), the cheapest possible cost AT THIS
        # SPARSITY. It is NOT full lexicographic optimality across d: a
        # sparser candidate at a higher d with k' < k would have R' >= k'
        # and could satisfy R' < R = k, i.e. beat this one, so in principle
        # the exit can stop before discovering it. It is kept deliberately:
        # the remaining d_try values are the regime where spurious dense
        # near-null directions live (see the sigma~0 cap above), and
        # genuinely multi-invariant systems get their additional generators
        # from the deflation collection stage rather than from deeper
        # d_try values. The paper describes this stop as a heuristic, not
        # a certificate.
        if best_R_here is not None and best_R_here <= best_k_here + 1e-9:
            break

    if not candidates:
        # The capped d_try sweep (deliberately narrow for speed and noise
        # robustness on the well-conditioned benchmarks) found NOTHING at
        # all. For essentially noiseless data, fall back to the
        # full-nullspace deflation search before giving up: a single
        # generous d* estimate with the same enum/bb/L1-routed circuit
        # recovery and the same gates -- exactly the sparse/high-d regime
        # the capped sweep cannot reach, so this is where bb_search gets a
        # genuine chance to fire BY DEFAULT. Zero regression risk on any
        # case the sweep already solves (this branch only runs when sr_gb
        # would otherwise return empty), and at sigma ~ 0 the strict exact
        # gates (rel_tol ~ 5e-6) keep it from manufacturing junk. Under
        # NOISE the fallback stays off: returning empty there is the
        # published abstention behaviour, and replacing abstention with a
        # generous-d* noisy search is precisely the spurious-invariant
        # hazard measured on distance_3d (feynman_polynomials.py, not a
        # verified official Feynman ID; see the collection tail's sigma
        # gate below). Noisy callers who want it use
        # full_nullspace=True explicitly.
        if ((sigma_estimate is None or sigma_estimate <= 1e-9)
                and monomials is not None and sym_vars is not None):
            return full_nullspace_deflation_search(
                Phi, s, Vt, N, M, k_max, max_denom, eps, sigma_estimate,
                monomials, data, sym_vars, var_names, use_bb=use_bb,
                return_details=return_details)
        return None

    candidates.sort(key=lambda t: t[0])
    _, best_detailed, best_dtry, best_Vnull = candidates[0]
    gb_list, c_red, S_red, R, k = best_detailed

    # Multi-invariant recovery at the FULL nullspace dimension. The winner loop
    # may have early-exited at a SMALL best_dtry, whose low-dimensional slice of
    # the nullspace need not span the other simultaneous invariants. So run the
    # collection at the estimated full nullspace dimension d_star instead, where
    # every conservation law is a distinct pinned rational direction. The primary
    # c_red lives in best_dtry's slice, hence in the larger d_star nullspace too.
    #
    # This runs the full-nullspace DEFLATION loop (_collect_additional_
    # invariants), not a single budget-gated enumeration pass: it is NOT gated
    # on comb(M, d_collect-1) fitting the enumeration budget, because
    # _recover_circuit itself falls through to bb_search (then L1) once that
    # budget is exceeded -- exactly the sparse/high-d* (loop-invariant) regime
    # that a single enumeration pass cannot reach.
    #
    # NOISE GATE: the collection only runs for essentially noiseless data
    # (same sigma <= 1e-9 convention as the d_cap logic above). Every genuine
    # multi-invariant benchmark (loop invariants, moiety totals, the 2D
    # oscillator's difference library, entangled linear laws) is exact data;
    # under noise, by contrast, the residual direction left after deflating
    # the primary is noise-dominated, and the noise-widened tolerances
    # (eps = 3*sigma rationality windows, 5*sigma residual gate) are too weak
    # to reject it reliably; measured directly on distance_3d
    # (feynman_polynomials.py, not a verified official Feynman ID) at
    # sigma=0.02, where the leftover direction snapped to the spurious
    # d^2 - 21d/4 + 7 (no real roots) and the Groebner merge then corrupted
    # the TRUE generator. Single-invariant return under noise is the
    # validated published behaviour; callers who genuinely want noisy
    # multi-invariant deflation opt in via full_nullspace=True, where the
    # regime is explicit.
    d_star = (M - estimate_rank(s, sigma_estimate, N)) if s is not None else best_dtry
    d_collect = max(best_dtry, d_star)
    sigma_clean = (sigma_estimate is None) or (sigma_estimate <= 1e-9)
    if (sigma_clean and d_collect >= 2 and monomials is not None
            and sym_vars is not None and gb_list):
        V_collect = _orthonormal_null_basis(Vt, M, d_collect)
        base_polys = [(g.as_expr() if hasattr(g, 'as_expr') else g) for g in gb_list]
        extra_polys = _collect_additional_invariants(
            V_collect, c_red, base_polys, Phi, N, M, k_max, max_denom, eps,
            sigma_estimate, monomials, data, sym_vars, var_names, use_bb)
        if extra_polys:
            gb_list = list(groebner([Poly(p, *sym_vars)
                                     for p in base_polys + extra_polys],
                                    *sym_vars, order='grevlex'))
        best_detailed = (gb_list, c_red, S_red, R, k)

    if return_details:
        return best_detailed
    return best_detailed[0]  # gb list only

# ============================================================================
#  Unified full-nullspace deflation entry point
# ============================================================================
def full_nullspace_deflation_search(Phi, s, Vt, N, M, k_max, max_denom, eps,
                                    sigma_estimate, monomials, data, sym_vars,
                                    var_names, use_bb=True, d_star=None,
                                    return_details=False):
    """Explicit unified full-nullspace deflation pipeline: take a single
    generous nullspace-dimension estimate d*, find
    the lowest-cost circuit of the FULL nullspace via `_recover_circuit`
    (enumeration if it fits budget, else bb_search, else L1), then keep
    deflating (`_collect_additional_invariants`) until no clean circuit
    remains, and Groebner-canonicalize whatever was collected.

    This is the sparse/high-d (loop-invariant, difference-library
    deflation-benchmark) regime's own entry point, kept SEPARATE from
    `progressive_nullspace_search`'s default path rather than replacing it:
    the progressive d_try sweep + early exit is validated noise-robust
    behaviour for the well-conditioned cases (circle, sphere, Feynman) this
    module already passes, and a generous full-nullspace search is unnecessary
    extra cost and unnecessary noise-fit risk for those. Call this function
    (or `sr_gb(..., full_nullspace=True)` / `sr_gb_fixed(...,
    full_nullspace=True)`) when the target is known to be in the sparse
    high-d* regime instead -- exactly what `benchmark_loop_invariants.py` and
    `benchmark_deflation_multi_invariant.py` use rather than maintaining their
    own private `bb_search` + ad hoc deflation loops.
    """
    if d_star is None:
        d_star = M - estimate_rank(s, sigma_estimate=sigma_estimate, N=N)
    d_star = max(1, min(d_star, M - 1))
    tau_resid = _tau_resid(sigma_estimate, N, Phi)

    V_full = _orthonormal_null_basis(Vt, M, d_star)
    if d_star == 1:
        c = V_full[:, 0]
    else:
        c = _recover_circuit(V_full, d_star, Phi, N, M, k_max, max_denom, eps,
                             sigma_estimate, tau_resid, use_bb)
    if c is None:
        return None

    detailed = _finalize_candidate(c, Phi, monomials, data, sym_vars,
                                   sigma_estimate, eps, max_denom, N,
                                   var_names, return_details=True)
    if detailed is None:
        return None

    gb_list, c_red, S_red, R, k = detailed

    if d_star >= 2:
        base_polys = [(g.as_expr() if hasattr(g, 'as_expr') else g) for g in gb_list]
        extra_polys = _collect_additional_invariants(
            V_full, c_red, base_polys, Phi, N, M, k_max, max_denom, eps,
            sigma_estimate, monomials, data, sym_vars, var_names, use_bb)
        if extra_polys:
            gb_list = list(groebner([Poly(p, *sym_vars)
                                     for p in base_polys + extra_polys],
                                    *sym_vars, order='grevlex'))

    if return_details:
        return gb_list, c_red, S_red, R, k
    return gb_list

# ============================================================================
#  Fixed‑degree pipeline (with detail return)
# ============================================================================
def sr_gb_fixed(data, var_names, degree, max_denom=16, eps=1e-4,
                expected_support=None, sigma_estimate=0.0, k_max=6, use_bb=True,
                s=None, Vt=None, monomials=None, Phi=None,
                return_details=False, precondition_columns=True,
                full_nullspace=False):
    from math import comb

    if sigma_estimate is not None and sigma_estimate > 0:
        eps = max(eps, 3.0 * sigma_estimate)

    if monomials is None or Phi is None:
        sym_vars, monomials, evaluate = build_monomial_library(
            var_names, degree, min_degree=0, scale=False)
        # Errors-in-variables column correction (see evaluate's docstring):
        # only when the caller supplied a positive noise level; with
        # sigma_estimate=None the noise scale is not known until after the
        # SVD, and prebuilt-Phi callers (the else branch) own their columns.
        _ns = sigma_estimate if (sigma_estimate is not None and sigma_estimate > 0) else 0.0
        Phi, _, _ = evaluate(data, noise_sigma=_ns)
        N, M = Phi.shape
    else:
        N, M = Phi.shape
        sym_vars = symbols(var_names)

    if s is None or Vt is None:
        # Column preconditioning: normalise columns before
        # the SVD so estimate_rank and the d_try sweep see a well-conditioned
        # spectrum even when the raw monomial library spans many orders of
        # magnitude (the ps3 case). _unscale_Vt maps the resulting right-
        # singular-vectors back to real monomial coordinates, so nothing
        # downstream of this branch needs to know it happened.
        if precondition_columns and Phi.size:
            s, Vt = _column_preconditioned_svd(Phi)
        else:
            # Same undersampled-case handling as _column_preconditioned_svd.
            _, s, Vt = svd(Phi, full_matrices=Phi.shape[0] < Phi.shape[1])

    if sigma_estimate is None:
        sigma_estimate = estimate_sigma_from_svd(s, N, M)

    if expected_support is not None:
        mon_names = [str(m) for m in monomials]
        support_idx = []
        for name in expected_support:
            if name in mon_names:
                support_idx.append(mon_names.index(name))
        if len(support_idx) == 0:
            c = Vt[-1, :]
        else:
            Phi_restricted = Phi[:, support_idx]
            Ur, sr, Vtr = svd(Phi_restricted, full_matrices=False)
            r_restricted = estimate_rank(sr, sigma_estimate=sigma_estimate, N=N)
            c_restricted = Vtr[-1, :]
            c = np.zeros(M)
            for i, idx in enumerate(support_idx):
                c[idx] = c_restricted[i]
        result = _finalize_candidate(c, Phi, monomials, data, sym_vars,
                                      sigma_estimate, eps, max_denom, N,
                                      var_names, return_details=return_details)
        if result is None:
            return [] if not return_details else ([], None, None, float('inf'), 0)
        return result
    elif full_nullspace:
        # Explicit opt-in to the unified full-nullspace deflation pipeline
        # (see full_nullspace_deflation_search docstring): bypasses the
        # progressive d_try sweep entirely and searches the full generous
        # nullspace directly, which is what the sparse/high-d loop-invariant
        # and difference-library deflation-benchmark regimes need.
        result = full_nullspace_deflation_search(
            Phi, s, Vt, N, M, k_max, max_denom, eps, sigma_estimate,
            monomials, data, sym_vars, var_names, use_bb=use_bb,
            return_details=return_details)
        if result is None:
            return [] if not return_details else ([], None, None, float('inf'), 0)
        return result
    else:
        result = progressive_nullspace_search(
            Phi, s, Vt, k_max, max_denom, eps, sigma_estimate, N, M,
            use_bb=use_bb, monomials=monomials, data=data, sym_vars=sym_vars,
            var_names=var_names, return_details=return_details)
        if result is None:
            return [] if not return_details else ([], None, None, float('inf'), 0)
        return result

# ============================================================================
#  Adaptive pipeline with full MDL
# ============================================================================
def sr_gb_adaptive(data, var_names, min_degree=1, max_degree=10,
                   max_denom=16, eps=1e-4, expected_support=None,
                   sigma_estimate=0.0, k_max=6, use_bb=True,
                   max_invariants_per_degree=3, pruning_log_out=None,
                   max_consecutive_misses=3, precondition_columns=True):
    if min_degree < 1:
        min_degree = 1

    sym_vars, all_monomials, evaluate = build_monomial_library(
        var_names, max_degree, min_degree=0, scale=False)

    mon_to_deg = {}
    mon_to_idx = {}
    for idx, mon in enumerate(all_monomials):
        if mon == 1:
            deg = 0
        else:
            deg = sum(mon.as_powers_dict().values())
        mon_to_deg[mon] = deg
        mon_to_idx[mon] = idx

    # Same errors-in-variables correction as sr_gb_fixed (see evaluate's
    # docstring); every downstream consumer (QR growth, gates, MDL) reads
    # from Phi_full, so correcting it here corrects the whole sweep.
    _ns = sigma_estimate if (sigma_estimate is not None and sigma_estimate > 0) else 0.0
    Phi_full, _, _ = evaluate(data, noise_sigma=_ns)
    N, M_full = Phi_full.shape

    # Column preconditioning, threaded through the QR-
    # incremental SVD updates below. Only the SVD/QR bookkeeping operates on
    # the column-normalised copy Phi_full_qr; every residual/rationality gate
    # downstream still uses the real, unscaled Phi_active, and every nullspace
    # vector that leaves an SVD step is unscaled + renormalised back to real
    # monomial coordinates via _unscale_Vt, so nothing past that point needs
    # to know preconditioning happened.
    if precondition_columns:
        col_norms_full = np.linalg.norm(Phi_full, axis=0)
        col_norms_full = np.where(col_norms_full < 1e-300, 1.0, col_norms_full)
        Phi_full_qr = Phi_full / col_norms_full
    else:
        col_norms_full = np.ones(M_full)
        Phi_full_qr = Phi_full

    active_indices = [0]
    active_monomials = [all_monomials[0]]
    Phi_active = Phi_full[:, active_indices]
    Phi_active_qr = Phi_full_qr[:, active_indices]
    # Maintain Q, R (not a chained SVD) as the active monomial set grows.
    # progressive_nullspace_search only ever reads s and Vt, never U (the
    # large N x M orthogonal factor) -- so there is nothing downstream that
    # needs an incrementally-updated U at all. A product-form SVD update that
    # chained the reassembly of U across degrees (via a small "middle matrix"
    # SVD at each degree) is a measured source of precision loss: on one
    # noisy circle seed, the chained update's
    # smallest singular vector carried a spurious real y-component of
    # ~0.03 where a direct SVD of the identical matrix gives ~1e-5,
    # because each reassembly step introduces its own rounding error and
    # those compound. QR column-insertion (scipy's qr_insert, LAPACK-
    # backed Householder updating) is the standard stable way to grow the
    # big N x M factor; since Phi_active = Q @ R with Q orthonormal, R's
    # singular values/right singular vectors ARE Phi_active's exactly, so
    # taking a FRESH svd() of the small (M x M, independent of N) R at
    # each degree costs O(M^3) -- cheap even at CRN/glycolysis scale
    # (M~700-1000) -- and matches a full direct SVD of Phi_active to
    # machine precision every time, with no chained error and no
    # heuristic threshold to tune, PROVIDED Phi_active is genuinely full
    # column rank. That proviso can fail (see qr_reliable below).
    Q_qr, R_qr = qr(Phi_active_qr, mode='economic')
    _, s, Vt = svd(R_qr, full_matrices=R_qr.shape[0] < R_qr.shape[1])
    if precondition_columns:
        Vt = _unscale_Vt(Vt, col_norms_full[active_indices])
    # Once the active set is (numerically) exactly rank-deficient -- exact
    # linear dependencies among monomials, e.g. fixed-dt transition pairs
    # where x_next/v_next are exact deterministic functions of x_t/v_t, or
    # a MuJoCo transition dataset for a linear system (measured directly:
    # a robotics benchmark's D=2 library, M=15, has a genuine 9-dim exact
    # nullspace) -- plain (non-pivoted) qr() is not reliably rank-revealing,
    # and neither, it turns out, is qr_insert building on top of it: once
    # bad conditioning is baked into R, LATER qr_insert calls can succeed
    # without raising even though the resulting R's singular spectrum does
    # not match a direct SVD of the same Phi_active (measured: R_qr's
    # SVD reported rank 8 where a fresh svd(Phi_active_qr) correctly showed
    # rank 6, sending progressive_nullspace_search's d_try jump to the
    # wrong dimension and losing the true invariant). Once any degree hits
    # this, stop trusting svd(R_qr) for the rest of the sweep -- always
    # recompute s, Vt from a direct SVD of Phi_active_qr instead (same
    # O(N*M^2) cost as the qr() fallback already paid when it fires; Q_qr/
    # R_qr are still maintained for future qr_insert growth, which is
    # unaffected).
    qr_reliable = True

    partial_gb = []
    best_per_degree = {}
    mdl_history = []
    pruning_log = []

    # Degrees where the per-degree search comes up empty take the silent
    # `continue` paths below and never touch mdl_history, so the MDL-increase
    # stop check further down (which only compares consecutive mdl_history
    # entries) never gets a chance to fire on them: once the true invariant
    # is found, every later degree can come back empty forever with nothing
    # to stop the sweep short of max_degree. Direct measurement on the
    # circle (D_max=10) showed exactly this: the invariant is found at D=2,
    # then D=3..7 each come back empty while still paying the full search
    # cost, which grows sharply once pruning shrinks
    # the active monomial count into progressive_nullspace_search's
    # affordable-bb_search budget at larger nullspace dimensions -- D=7
    # alone took 47s for an empty result. This counter closes that gap:
    # once at least one invariant has been found, N consecutive degrees
    # with nothing new (regardless of why -- no new monomials, all pruned
    # away, or a search that found nothing) stop the sweep. It cannot mask
    # a genuine higher-degree competitor: any find resets the counter, so
    # global-MDL selection over best_per_degree still sees it.
    #
    # A second, distinct pattern needs the SAME counter, not a separate one:
    # once the true invariant is found, every higher degree's redundant
    # nullspace (multiples of that invariant) can cause the search to
    # "find" something at every single degree -- but it is the IDENTICAL
    # polynomial each time, re-derived from an ever-larger nullspace at
    # ever-increasing cost (measured directly: D=6 ~3-7s, D=7 ~70-85s for
    # one noisy circle seed, still climbing). Its MDL creeps up by only
    # ~1-2 units per degree, nowhere near the 0.5*log(N) jump threshold
    # below, so that check never fires either. Re-finding the SAME
    # invariant is not progress, so it must count as a miss too, not reset
    # the counter the way a genuinely NEW candidate should.
    consecutive_misses = 0
    last_found_poly = None
    def _same_invariant(p1, p2):
        if p1 is None or p2 is None:
            return False
        try:
            ratio = cancel(p1 / p2)
            return ratio.is_number or ratio.is_constant()
        except Exception:
            return p1 == p2
    def _register_miss():
        nonlocal consecutive_misses
        if not mdl_history:
            return False
        consecutive_misses += 1
        return consecutive_misses >= max_consecutive_misses

    for D in range(min_degree, max_degree + 1):
        new_monomials = [mon for mon, deg in mon_to_deg.items() if deg == D]
        if not new_monomials:
            if _register_miss():
                break
            continue

        gb_exprs = [g.as_expr() if hasattr(g, 'as_expr') else g for g in partial_gb]
        degree_log = []
        pruned_new = prune_active_monomials(new_monomials, gb_exprs, sym_vars=sym_vars, log=degree_log)
        if degree_log:
            pruning_log.append({'degree': D, **degree_log[0]})
        if not pruned_new:
            if _register_miss():
                break
            continue

        new_indices = [mon_to_idx[mon] for mon in pruned_new if mon in mon_to_idx]
        if not new_indices:
            if _register_miss():
                break
            continue

        new_cols_qr = Phi_full_qr[:, new_indices]
        try:
            Q_qr, R_qr = qr_insert(Q_qr, R_qr, new_cols_qr, R_qr.shape[1], which='col')
        except np.linalg.LinAlgError:
            # A new column is (numerically) exactly in the span of the
            # existing active set -- qr_insert raises rather than silently
            # degrading. This is a real case, not just a hypothetical: exact
            # linear dependencies show up for e.g. fixed-dt transition pairs
            # where x_next/v_next are exact deterministic functions of
            # x_t/v_t (see progressive_nullspace_search's docstring). Fall
            # back to a fresh QR of the extended active set -- still only
            # O(N*M^2), the same cost svd_update's old cond_R fallback paid.
            qr_reliable = False
            active_indices_extended = active_indices + new_indices
            Phi_active_qr_new = Phi_full_qr[:, active_indices_extended]
            Q_qr, R_qr = qr(Phi_active_qr_new, mode='economic')
        active_indices.extend(new_indices)
        active_monomials = [all_monomials[i] for i in active_indices]
        Phi_active = Phi_full[:, active_indices]
        Phi_active_qr = Phi_full_qr[:, active_indices]
        if qr_reliable:
            _, s, Vt = svd(R_qr, full_matrices=R_qr.shape[0] < R_qr.shape[1])
        else:
            # Once ANY degree hit the exact-rank-deficiency branch above,
            # R_qr cannot be trusted for rank/nullspace purposes even
            # at LATER degrees whose own qr_insert call raised no exception:
            # a subsequent insertion can succeed silently on top of an
            # already ill-conditioned R while the resulting spectrum still
            # diverges from a direct SVD of the same Phi_active. Measured
            # directly on a MuJoCo transition dataset (exact linear dynamics,
            # D=2 library, M=15, true 9-dim exact nullspace): the D=1 insert
            # raised and was caught, D=2's insert did NOT raise, yet
            # svd(R_qr) at D=2 still reported rank 8 (d_est=7) against the
            # true rank 6 (d_est=9) a fresh svd(Phi_active_qr) shows, sending
            # progressive_nullspace_search's d_try jump to the wrong
            # dimension and losing the true invariant. Recompute directly
            # from Phi_active_qr every degree for the rest of the sweep
            # instead (same O(N*M^2) order as the qr() fallback already
            # paid); Q_qr/R_qr themselves are untouched, so future
            # qr_insert growth attempts are unaffected.
            _, s, Vt = svd(Phi_active_qr, full_matrices=Phi_active_qr.shape[0] < Phi_active_qr.shape[1])
        if precondition_columns:
            Vt = _unscale_Vt(Vt, col_norms_full[active_indices])

        invariants_this_degree = []
        s_def, Vt_def = s, Vt
        Phi_def = Phi_active

        tau_resid = _tau_resid(sigma_estimate, N, Phi_active)

        for _ in range(max_invariants_per_degree):
            gb_res, c, S, R, k = sr_gb_fixed(
                data, var_names, degree=D,
                max_denom=max_denom, eps=eps,
                expected_support=expected_support,
                sigma_estimate=sigma_estimate,
                k_max=k_max, use_bb=use_bb,
                s=s_def, Vt=Vt_def,
                monomials=active_monomials,
                Phi=Phi_def,
                return_details=True,
                precondition_columns=precondition_columns
            )
            if not gb_res:
                break

            poly = gb_res[0].as_expr() if hasattr(gb_res[0], 'as_expr') else gb_res[0]
            poly = reduce_to_minimal_generator(poly, data, sigma_estimate, sym_vars=sym_vars)
            if poly == 0:
                break

            invariants_this_degree.append((poly, c, S, R, k, gb_res))
            partial_gb.extend(gb_res)

            # A further iteration here would re-run the search against the
            # SAME Phi_def with an SVD recomputed from the SAME matrix (no
            # projection/deflation of the just-found direction actually
            # happens below), and SVD is a deterministic decomposition, so a
            # repeat iteration is guaranteed to rediscover this identical
            # invariant rather than a genuinely new one. That was harmless
            # when each search was cheap, but once a genuinely large exact
            # nullspace pushes a single search to many seconds (a design
            # with functionally dependent monomials, e.g. fixed-dt
            # transition pairs), repeating it up to max_invariants_per_degree
            # times for what is structurally a single-invariant system is
            # pure wasted computation. Real multi-invariant deflation is
            # implemented separately (see benchmark_deflation_multi_invariant.py),
            # not via this loop, so stopping here changes nothing about what
            # this loop can actually find, only how many times it re-finds it.
            break

        if invariants_this_degree:
            # Select best by full MDL
            best_poly = None
            best_mdl = float('inf')
            best_gb_res = None
            for poly, c, S, R, k, gb_res_i in invariants_this_degree:
                # Degree cost per the paper's Eq. (mdl-cost) is the CONSTANT
                # log2(D_max) of the sweep, not log2(current D): charging the
                # current degree biased the global argmin toward lower
                # degrees relative to the stated formula.
                mdl = full_mdl_cost(poly, data, sigma_estimate, var_names,
                                    max_degree, active_monomials, Phi_active,
                                    max_denom=max_denom)
                if mdl < best_mdl:
                    best_mdl = mdl
                    best_poly = poly
                    best_gb_res = gb_res_i
            if best_poly is not None:
                best_per_degree[D] = (best_mdl, best_poly, best_gb_res)
                mdl_history.append(best_mdl)
                if _same_invariant(best_poly, last_found_poly):
                    if _register_miss():
                        break
                else:
                    consecutive_misses = 0
                last_found_poly = best_poly
                # Stopping criterion: if MDL increased by > 0.5 log2(N), stop
                # (BIC penalty in bits, matching full_mdl_cost's unit)
                if len(mdl_history) >= 2:
                    if mdl_history[-1] - mdl_history[-2] > 0.5 * np.log2(N):
                        break
            elif _register_miss():
                break
        elif _register_miss():
            break

    # Select global best by MDL
    best_global_mdl = float('inf')
    best_degree_final = None
    best_poly_global = None
    best_gb_res_global = None
    for D, (mdl, poly, gb_res_d) in best_per_degree.items():
        if mdl < best_global_mdl:
            best_global_mdl = mdl
            best_degree_final = D
            best_poly_global = poly
            best_gb_res_global = gb_res_d

    if pruning_log_out is not None:
        pruning_log_out.extend(pruning_log)

    if best_poly_global is None:
        return []

    # The winning degree's Gröbner basis was already computed above (and
    # cached in best_per_degree); reuse it instead of re-running the entire
    # search from scratch; only fall back to a fresh call in the (should not
    # happen) case where nothing was cached.
    if best_gb_res_global is not None:
        gb_final = best_gb_res_global
    else:
        gb_final = sr_gb_fixed(data, var_names, degree=best_degree_final,
                               max_denom=max_denom, eps=eps,
                               expected_support=expected_support,
                               sigma_estimate=sigma_estimate,
                               k_max=k_max, use_bb=use_bb,
                               precondition_columns=precondition_columns)

    # Algorithm 2 Step 7 / Prop. 4.5 (Conditional Adaptive Exact Recovery): the returned basis is the reduced GB
    # of the ideal generated by ALL invariants confirmed at degrees <= D*,
    # not just the winning degree's own generator. The winner was searched
    # in the PRUNED library (reduced modulo earlier-degree generators), so
    # by itself it does not imply them; returning it alone dropped
    # earlier-degree invariants from the output ideal. When only one degree
    # contributed (the common single-invariant case) this reduces to the
    # cached basis unchanged.
    contributing = {D: entry for D, entry in best_per_degree.items()
                    if D <= best_degree_final}
    if len(contributing) > 1:
        gens = []
        for D in sorted(contributing):
            for g in contributing[D][2]:
                ge = g.as_expr() if hasattr(g, 'as_expr') else g
                if ge != 0:
                    gens.append(ge)
        try:
            gb_merged = groebner(gens, *sym_vars, order='grevlex')
            gb_final = list(gb_merged)
        except Exception:
            pass  # keep the single-degree basis rather than fail the call
    return gb_final

# ============================================================================
#  Public entry point
# ============================================================================
def sr_gb_transition_difference(old_data, new_data, var_names, degree,
                                noise_sigma=None, **kwargs):
    """Recover invariants from a structured transition-difference library.

    This is the supported alternative to an unrestricted transition-pair
    dictionary when the target is a conserved state polynomial ``Q``.  It
    searches ``Q(z_t)-Q(z_{t+1})=0`` and returns polynomials in ``var_names``.
    ``degree`` is fixed because prebuilt libraries are not supported by the
    adaptive degree sweep.  Remaining keyword arguments are forwarded to
    :func:`sr_gb`, including ``sigma_estimate`` and ``full_nullspace``.

    ``noise_sigma`` (the Phi-column errors-in-variables correction, see
    ``build_monomial_library``'s ``evaluate``) defaults to the forwarded
    ``sigma_estimate`` when not given explicitly, so the two do not silently
    drift apart: elsewhere in the pipeline (``sr_gb_fixed``) they are always
    the same value by construction, and a caller here who set one without
    the other would get a Phi built for the wrong noise level relative to
    the gates that judge it. Pass ``noise_sigma`` explicitly to override.
    """
    if noise_sigma is None:
        noise_sigma = kwargs.get('sigma_estimate') or 0.0
    _, monomials, Phi = build_transition_difference_library(
        old_data, new_data, var_names, degree, noise_sigma=noise_sigma)
    return sr_gb(None, var_names, degree=degree, monomials=monomials,
                 Phi=Phi, **kwargs)


def sr_gb(data, var_names, degree=None, D_max=10, max_denom=16, eps=1e-4,
          expected_support=None, sigma_estimate=0.0, k_max=6, use_bb=True,
          min_degree=1, pruning_log_out=None, max_consecutive_misses=3,
          precondition_columns=True, full_nullspace=False,
          monomials=None, Phi=None):
    """Public entry point.

    full_nullspace=True routes the fixed-degree path through
    `full_nullspace_deflation_search` instead of the progressive d_try sweep;
    see that function's docstring. This is the sparse/high-d* (loop-invariant)
    entry point, and requires a fixed `degree`.

    monomials= / Phi= (the "prebuilt-Phi entry"): pass a
    pre-built monomial list and design matrix instead of having sr_gb build
    its own library from `data`. This is what lets a difference-library
    caller (e.g. benchmark_deflation_multi_invariant.py, which feeds
    Phi_old - Phi_new so a conserved quantity Q satisfies Q(old)-Q(new)=0 on
    every sample) route through sr_gb() instead of reimplementing its own
    bb_search/nullspace pipeline. `data` may then be None: the residual and
    term-scale gates are computed from Phi directly (see _finalize_candidate
    and _rel_rms_residual), since a conserved quantity does not vanish
    pointwise on state data. Only supported with a fixed `degree`
    (sr_gb_adaptive always builds its own per-degree library from `data`).
    """
    if degree is None:
        if full_nullspace:
            raise ValueError(
                "full_nullspace=True requires a fixed `degree`; the adaptive "
                "sweep manages its own per-degree search.")
        if Phi is not None or monomials is not None:
            raise ValueError(
                "monomials=/Phi= (prebuilt library) require a fixed `degree`; "
                "sr_gb_adaptive always builds its own per-degree library from data.")
        return sr_gb_adaptive(data, var_names, min_degree=min_degree,
                              max_degree=D_max, max_denom=max_denom,
                              eps=eps, expected_support=expected_support,
                              sigma_estimate=sigma_estimate,
                              k_max=k_max, use_bb=use_bb,
                              pruning_log_out=pruning_log_out,
                              max_consecutive_misses=max_consecutive_misses,
                              precondition_columns=precondition_columns)
    else:
        return sr_gb_fixed(data, var_names, degree=degree,
                           max_denom=max_denom, eps=eps,
                           expected_support=expected_support,
                           sigma_estimate=sigma_estimate,
                           k_max=k_max, use_bb=use_bb,
                           precondition_columns=precondition_columns,
                           full_nullspace=full_nullspace,
                           monomials=monomials, Phi=Phi)

# ============================================================================
#  Helper: exact recovery check
# ============================================================================
def exact_recovery(gb_result, true_poly_expr):
    if not gb_result:
        return False
    for p in gb_result:
        p_expr = p.as_expr() if hasattr(p, 'as_expr') else p
        try:
            ratio = cancel(p_expr / true_poly_expr)
            if ratio.is_number or ratio.is_constant():
                return True
        except Exception:
            pass
    return False

# ============================================================================
#  Helper: reduce to minimal generator (kept from original)
# ============================================================================
def _check_residual(poly_expr, data, sigma_estimate, sym_vars=None, full_vars=None):
    """
    full_vars is the FULL, data-column-ordered variable list (data[:, j]
    corresponds to full_vars[j]). sym_vars may be an arbitrary subset/
    reordering of full_vars (e.g. only the variables poly_expr actually
    depends on); each symbol's data column is looked up by its position
    in full_vars, not positionally within sym_vars -- using range(len(sym_vars))
    against data's columns directly is only correct when sym_vars happens to
    be an exact column-order prefix of full_vars, which silently fails (wrong
    columns fed to poly_expr, sometimes without even an exception) once a
    polynomial drops a variable that isn't last in the column order.
    """
    tau_floor = max(1e-4, sigma_estimate * 3.0)
    if sym_vars is None:
        sym_vars = sorted(poly_expr.free_symbols, key=lambda s: s.name)
    if full_vars is None:
        full_vars = sym_vars
    try:
        col_indices = [full_vars.index(v) for v in sym_vars]
        f = lambdify(sym_vars, poly_expr, modules='numpy')
        vals = np.atleast_1d(np.asarray(
            f(*[data[:, j] for j in col_indices]), dtype=float))
        N = len(vals)
        grad_norm = 1.0
        if sigma_estimate > 0 and N > 0:
            grad_exprs = [poly_expr.diff(v) for v in sym_vars]
            grad_cols = []
            for g in grad_exprs:
                gf = lambdify(sym_vars, g, modules='numpy')
                gv = gf(*[data[:, j] for j in col_indices])
                grad_cols.append(np.broadcast_to(np.asarray(gv, dtype=float), (N,)))
            grad_vals = np.stack(grad_cols, axis=1)
            grad_norm = max(1.0, float(np.median(np.linalg.norm(grad_vals, axis=1))))
        extreme_factor = np.sqrt(2.0 * np.log(max(N, 2)))
        tau = max(tau_floor, 3.0 * sigma_estimate * grad_norm * extreme_factor)
        return np.max(np.abs(vals)) < tau
    except Exception:
        return False

def reduce_to_minimal_generator(poly_expr, data, sigma_estimate, sym_vars=None, eps=1e-3, full_vars=None):
    if poly_expr == 0:
        return poly_expr
    if full_vars is None:
        # Only true on the initial (non-recursive) call: sym_vars here is
        # the caller's full, data-column-ordered variable list. Every
        # recursive call below passes full_vars explicitly so this
        # original column mapping is never lost to the vars_in_poly
        # subsetting immediately below.
        full_vars = sym_vars
    if sym_vars is None:
        vars_in_poly = sorted(poly_expr.free_symbols, key=lambda s: s.name)
    else:
        vars_in_poly = [v for v in sym_vars if v in poly_expr.free_symbols]
        if not vars_in_poly:
            vars_in_poly = sorted(poly_expr.free_symbols, key=lambda s: s.name)
    if not vars_in_poly:
        return poly_expr
    if full_vars is None:
        full_vars = vars_in_poly
    p_poly = Poly(poly_expr, *vars_in_poly, domain='QQ')
    int_poly = p_poly.clear_denoms()[1]
    factored = factor(int_poly.as_expr())
    factors = []
    if factored.is_Mul:
        for f in factored.args:
            if f.is_Number:
                continue
            if f.is_Pow:
                base = f.base
            else:
                base = f
            if base not in factors:
                factors.append(base)
    elif factored.is_Pow:
        factors = [factored.base]
    else:
        factors = [factored]
    for f in factors:
        if f == poly_expr:
            continue
        if _check_residual(f, data, sigma_estimate, sym_vars=vars_in_poly, full_vars=full_vars):
            return reduce_to_minimal_generator(f, data, sigma_estimate, sym_vars=vars_in_poly, full_vars=full_vars)
    for var in vars_in_poly:
        if poly_expr.has(var):
            quotient, remainder = div(poly_expr, var, domain='QQ')
            if remainder == 0 and quotient != 0 and quotient != 1:
                # Symmetric with the factor-branch above: verify the
                # quotient actually still explains the data before
                # recursing on it, rather than trusting exact symbolic
                # divisibility (var*quotient == poly_expr as polynomials)
                # to imply quotient itself is small on the real, noisy data.
                if _check_residual(quotient, data, sigma_estimate, sym_vars=vars_in_poly, full_vars=full_vars):
                    return reduce_to_minimal_generator(quotient, data, sigma_estimate, sym_vars=vars_in_poly, full_vars=full_vars)
    return poly_expr

# ============================================================================
#  Normal-form-safe pruning of monomials using partial Gröbner basis
#  (paper Lemma 4.3, "Normal-Form Pruning": deg(NF_G(p)) <= deg(p) only
#  holds under a GRADED order, so the leading monomial of each confirmed
#  generator must be taken under grevlex -- the order used everywhere else
#  in the pipeline (build_monomial_library, groebner(..., order='grevlex'))
#  -- not sympy's Poly.LM() default (lex), and gens must be the pipeline's
#  canonical sym_vars rather than an arbitrarily-ordered free_symbols set,
#  since ties under grevlex are broken by generator order.
#
#  Poly.LM() returns a sympy.polys.monomials.Monomial object, not an Expr.
#  div() (called below to test divisibility) cannot polify a Monomial and
#  raises PolificationFailed on every call, so the Monomial must be converted
#  to an Expr via .as_expr() before it reaches div(). Without that conversion
#  every divisibility test raises, is swallowed by the bare except/continue
#  in the divisibility loop, `divisible` is never set True, and this function
#  prunes nothing for any input. The grevlex/sym_vars requirement above and
#  this .as_expr() conversion are both needed: the former only fixes the
#  leading-monomial computation, and does nothing on its own without the latter.
# ============================================================================
def prune_active_monomials(active_monomials, gb_basis, sym_vars=None, log=None):
    if not gb_basis:
        return active_monomials[:]
    leading_monomials = []
    for g in gb_basis:
        if hasattr(g, 'as_expr'):
            g_expr = g.as_expr()
        else:
            g_expr = g
        if g_expr == 0:
            continue
        try:
            gens = sym_vars if sym_vars is not None else sorted(g_expr.free_symbols, key=str)
            poly = Poly(g_expr, *gens)
            lm = poly.LM(order='grevlex').as_expr()
            leading_monomials.append(lm)
        except Exception:
            continue
    if not leading_monomials:
        return active_monomials[:]
    pruned = []
    for mon in active_monomials:
        if mon == 1:
            pruned.append(mon)
            continue
        divisible = False
        for lm in leading_monomials:
            try:
                _, rem = div(mon, lm, domain='QQ')
                if rem == 0:
                    divisible = True
                    break
            except Exception:
                continue
        if not divisible:
            pruned.append(mon)
    if log is not None:
        log.append({
            'n_active': len(active_monomials),
            'n_pruned': len(active_monomials) - len(pruned),
            'n_kept': len(pruned),
        })
    return pruned
