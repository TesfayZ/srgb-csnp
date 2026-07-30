"""
avi_baselines.py - Approximate Vanishing Ideal (AVI) / Approximate Border
Basis (ABM) baseline.

Implements the numerical Buchberger-Moller / border-basis-by-least-squares
construction of Heldt, Kreuzer, Pokutta & Poulisse (2009), "Approximate
computation of zero-dimensional polynomial ideals" (see also Kreuzer &
Robbiano 2010, cited in the paper as Heldt2009 / Kreuzer2010). The paper
discusses AVI in Adaptive_CSNP/Adaptive-CSNP.tex (Section "Approximate
vanishing ideals and variety learning" and Appendix "Comparison with
Approximate Vanishing Ideals"); this module runs it, over the
monomial-evaluation machinery in sr_gb.py (`build_monomial_library`), so it
can be benchmarked directly against SR-GB+CSNP.

Algorithm (matches the sketch in the paper's AVI appendix):

    O <- {1}                     order ideal of "standard" monomials
    G <- {}                      border basis (approximate vanishing polys)
    for d = 1, 2, ..., max_degree:
        border_d <- { x_i * m : m in O, deg(m) == d-1 } \\ (already resolved)
        for each t in border_d (fixed order):
            least-squares project eval(t) onto span{ eval(m) : m in O }
            if the relative residual ||eval(t) - proj|| / ||eval(t)|| < eps:
                t is (approximately) dependent on O -> record the relation
                g_t = t - sum_i lambda_i * m_i as an approximate vanishing
                polynomial; t itself is NOT added to O (it becomes a
                "leading"/non-standard monomial and its border is not
                expanded further)
            else:
                t is (approximately) independent -> add t to O (it becomes
                a new standard monomial and its own children extend the
                border at degree d+1)
    return G

This is a genuine border-basis construction (BFS over the border graph of
O), not a single dense degree-d SVD nullspace: O only grows one monomial
at a time and the border at degree d is generated from whichever
monomials of degree d-1 actually made it into O, exactly the distinction
the paper draws between AVI's border basis and the "Dense SVD+GB"
per-degree ablation.

The least-squares membership test here plays the same role that
`estimate_rank`'s singular-value threshold plays in the main SR-GB-CSNP
pipeline (sr_gb.py): both are numerical-rank/near-dependency decisions
under noise, just phrased as a relative-residual test on a single column
here (AVI processes monomials one at a time) rather than a spectral gap
over the whole matrix at once (SR-GB-CSNP's SVD nullspace estimate).

Output convention: each element of the returned border basis is snap-
rounded with sr_gb.snap_round to an exact-rational-coefficient sympy
expression, exactly like SR-GB+CSNP's own output, so that elements are
directly comparable up to scalar (e.g. via sr_gb.exact_recovery) with
SR-GB+CSNP's recovered generators. This does NOT further minimize or
canonicalize the set: the whole point of comparing against AVI is that it
returns the full border basis rather than a single minimal generator, so
no Groebner reduction / redundancy removal is applied here.
"""

import numpy as np
from sympy import Poly, Rational, symbols

from sr_gb import build_monomial_library, snap_round

# Hand-set, not fit to any particular benchmark. Sensitivity across a range
# of multipliers is characterized in ablation_avi_eps_multiplier.py: the
# per-system step to 100% correctness sits at a different k for circle vs.
# sphere, and 5.0 clears both with margin and no added over-merging cost up
# to k=7.0. Changing this value invalidates avi_baseline_results.csv /
# avi_baseline_summary.csv and Table tab:avi in Adaptive-CSNP.tex.
_AVI_EPS_MULTIPLIER = 5.0


def _monomial_exponents(sym_vars, monomials_sym):
    """Exponent tuple for each monomial in monomials_sym, same order."""
    n = len(sym_vars)
    zero_exp = tuple([0] * n)
    exps = []
    for mon in monomials_sym:
        if mon == 1:
            exps.append(zero_exp)
            continue
        p = Poly(mon, *sym_vars)
        exps.append(p.monoms()[0])
    return exps


def avi_border_basis(data, var_names, max_degree, eps=None, sigma_estimate=0.0):
    """
    Approximate Border Basis / AVI baseline (Heldt, Kreuzer, Pokutta,
    Poulisse 2009).

    Parameters
    ----------
    data : (N, n_vars) array
        Sample points (rows = observations, columns = variables, in the
        order given by var_names).
    var_names : list[str]
    max_degree : int
        Maximum total monomial degree considered. Bounds the border
        expansion; the construction still stops growing earlier than this
        if the order ideal stabilizes (no monomial at some degree is
        found independent, so its children are never generated).
    eps : float or None
        Relative-residual tolerance for the least-squares membership test,
        ||eval(t) - proj_O(eval(t))|| / ||eval(t)||. If None, derived from
        sigma_estimate: eps = max(1e-6, _AVI_EPS_MULTIPLIER * sigma_estimate).
        This is the AVI analogue of the absolute singular-value threshold
        `estimate_rank` uses for the SVD nullspace test in sr_gb.py: a
        relative (rather than absolute) tolerance because a single
        monomial's evaluation-vector scale varies enormously across
        degree and variable range, exactly the scale-dependence problem
        sr_gb.py's own gates (`_finalize_candidate`) solve with
        scale-relative thresholds. See the module-level
        `_AVI_EPS_MULTIPLIER` comment for its value and the ablation behind
        it.
    sigma_estimate : float
        Estimated per-coordinate noise standard deviation; used only when
        `eps` is None, and passed through to `snap_round` for the same
        reason it is throughout sr_gb.py (wider snap tolerance at higher
        noise).

    Returns
    -------
    G : list[sympy.Expr]
        The approximate vanishing polynomials making up the border basis,
        in discovery order. Comparable, up to scalar, with SR-GB+CSNP's
        output via `sr_gb.exact_recovery(G, target_expr)`.
    """
    data = np.asarray(data, dtype=float)
    N, n_vars = data.shape
    if N == 0:
        return []

    if eps is None:
        eps = max(1e-6, _AVI_EPS_MULTIPLIER * sigma_estimate)

    sym_vars, monomials_sym, evaluate = build_monomial_library(
        var_names, max_degree, min_degree=0)
    Phi, monomials_sym, _ = evaluate(data, noise_sigma=sigma_estimate)
    exps = _monomial_exponents(sym_vars, monomials_sym)
    col_of = {exp: i for i, exp in enumerate(exps)}
    expr_of = dict(zip(exps, monomials_sym))

    def degree_of(exp):
        return sum(exp)

    zero_exp = tuple([0] * n_vars)
    O = [zero_exp]                       # order ideal (standard monomials)
    O_cols = [Phi[:, col_of[zero_exp]]]  # matching evaluation vectors
    G = []                               # border basis (vanishing polys)
    processed = {zero_exp}

    for d in range(1, max_degree + 1):
        # Border at degree d: children of degree-(d-1) elements of O that
        # have not already been resolved (into either O or G).
        border_d = set()
        for exp in O:
            if degree_of(exp) != d - 1:
                continue
            for i in range(n_vars):
                child = list(exp)
                child[i] += 1
                child = tuple(child)
                if child in processed or child not in col_of:
                    continue
                border_d.add(child)
        if not border_d:
            # O has stopped growing at degree d-1 -> no further border to
            # test at any higher degree either; keep looping in case a
            # later degree's monomials aren't representable at all (they
            # simply won't appear in border_d), but nothing more happens.
            continue

        for exp in sorted(border_d):
            if exp in processed:
                continue
            processed.add(exp)

            t_col = Phi[:, col_of[exp]]
            A = np.stack(O_cols, axis=1)
            coef, _, _, _ = np.linalg.lstsq(A, t_col, rcond=None)
            resid = t_col - A @ coef
            t_norm = np.linalg.norm(t_col)
            rel_resid = np.linalg.norm(resid) / (t_norm + 1e-300)

            if rel_resid < eps:
                # t is (approximately) in the span of O's evaluation
                # vectors -> record the approximate vanishing relation
                # g = t - sum_i coef_i * O_i. t itself does NOT join O
                # (its children are therefore never generated as border
                # monomials, so multiples of this relation are not
                # redundantly re-tested).
                v = np.concatenate([[1.0], -np.asarray(coef, dtype=float)])
                rounded = snap_round(v, sigma_estimate=sigma_estimate)
                terms = [expr_of[exp]] + [expr_of[o] for o in O]
                g_expr = sum(Rational(c) * m for c, m in zip(rounded, terms)
                             if c != 0)
                if g_expr != 0:
                    G.append(g_expr)
            else:
                # t is (approximately) independent -> becomes a new
                # standard monomial, extending the order ideal (and hence
                # the border at degree d+1).
                O.append(exp)
                O_cols.append(t_col)

    return G


if __name__ == "__main__":
    # Quick self-test: circle x^2 + y^2 - 1, noiseless.
    from data_generator import generate_variety_data
    from sr_gb import exact_recovery
    from sympy import parse_expr

    X = generate_variety_data("x**2 + y**2 - 1", ["x", "y"],
                               {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
                               N=1000, sigma=0.0, seed=0)
    G = avi_border_basis(X, ["x", "y"], max_degree=4, sigma_estimate=0.0)
    true_expr = parse_expr("x**2 + y**2 - 1")
    print(f"Border basis cardinality: {len(G)}")
    for g in G:
        print(" ", g)
    print("Contains true invariant (up to scalar):",
          exact_recovery(G, true_expr))
