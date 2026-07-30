"""
verification.py – SOS and SMT verification of polynomial invariants.
Optional dependencies: SOSTOOLS (via MATLAB engine), Z3.
"""

import numpy as np
from sympy import lambdify, symbols, Poly, expand
import warnings
warnings.filterwarnings('ignore')

def sos_feasibility(poly_expr, constraints, var_names, solver='sostools'):
    """
    Check if poly_expr >= 0 on variety defined by constraints.
    Currently a stub – requires SOSTOOLS or CVXPY.
    """
    # Placeholder: use sampling as fallback
    return sos_feasibility_sampling(poly_expr, constraints, var_names)

def sos_feasibility_sampling(poly_expr, constraints, var_names, n_samples=10000,
                             seed=0, constraint_tol=1e-2):
    """
    Sampling-based check: if any negative value of poly_expr is found at a
    sampled point, return False.

    `constraints` (a list of expressions that vanish on the variety) restricts
    where nonnegativity is tested: samples are drawn from a uniform box and
    kept only at near-variety points (all |constraint| <= constraint_tol)
    before testing nonnegativity. Testing on the whole box instead would ask a
    different, stricter question than "nonnegative on the variety". Sampling is
    seeded for reproducibility. If no sample lands near the variety, returns
    None (inconclusive) rather than a misleading True/False.
    """
    syms = symbols(var_names)
    f = lambdify(syms, poly_expr, modules='numpy')
    rng = np.random.RandomState(seed)
    try:
        points = rng.uniform(-5, 5, (n_samples, len(var_names)))
        if constraints:
            mask = np.ones(n_samples, dtype=bool)
            for con in constraints:
                g = lambdify(syms, con, modules='numpy')
                gv = np.asarray(g(*[points[:, i] for i in range(len(var_names))]),
                                dtype=float)
                if gv.ndim == 0:
                    gv = np.full(n_samples, float(gv))
                mask &= np.abs(gv) <= constraint_tol
            points = points[mask]
            if len(points) == 0:
                return None  # inconclusive: no sample near the variety
        vals = np.asarray(f(*[points[:, i] for i in range(len(var_names))]),
                          dtype=float)
        return bool(np.min(vals) >= -1e-6)
    except Exception:
        return False

def verify_invariant_smt(poly_expr, true_expr, var_names, tolerance=1e-6):
    """
    Verify using Z3 that poly_expr is identically equal to true_expr (up to
    an overall sign), i.e. that (poly_expr - true_expr) or
    (poly_expr + true_expr) vanishes for every assignment of the variables.
    poly_expr is a candidate that vanishes only on a variety (e.g.
    x**2+y**2-1), not an expression that is identically zero itself, so
    checking poly_expr alone against zero is the wrong question -- this
    checks equivalence to the known true invariant instead.
    Returns (verified, counterexample, z3_available). verified is True if the
    two are identical up to sign, False with a counterexample point if they
    disagree, or None if the check is inconclusive: either because Z3
    answered "unknown" for both sign checks, or because Z3 is not installed,
    or because the sympy-to-z3 conversion itself raised (e.g. an unsupported
    expression form). z3_available is False only in the "not installed" case;
    callers that need to distinguish "verified" from "not actually checked"
    should gate on z3_available, not just on verified being truthy.
    """
    try:
        from z3 import Real, Solver, sat, unsat, RealVal
    except ImportError:
        print("Z3 not installed; skipping SMT verification.")
        return None, None, False

    syms = [Real(v) for v in var_names]
    # Convert sympy expression to Z3
    def to_z3(expr):
        if expr.is_Number:
            return RealVal(float(expr))
        elif expr.is_Symbol:
            return syms[var_names.index(str(expr))]
        elif expr.is_Add:
            return sum(to_z3(arg) for arg in expr.args)
        elif expr.is_Mul:
            prod = 1
            for arg in expr.args:
                prod *= to_z3(arg)
            return prod
        elif expr.is_Pow:
            base = to_z3(expr.base)
            exp = float(expr.exp)
            if exp == 1:
                return base
            elif exp == 2:
                return base * base
            else:
                return base ** RealVal(exp)
        else:
            raise ValueError(f"Unsupported expression: {expr}")

    try:
        diff_minus = expand(poly_expr - true_expr)
        diff_plus = expand(poly_expr + true_expr)
        z3_diff_minus = to_z3(diff_minus)
        z3_diff_plus = to_z3(diff_plus)
    except Exception as e:
        print(f"Z3 conversion failed: {e}")
        return None, None, True

    # poly_expr matches true_expr up to an overall sign iff either
    # difference is identically zero, i.e. its negation check is UNSAT.
    counterexample = None
    saw_sat = False
    for z3_diff in (z3_diff_minus, z3_diff_plus):
        solver = Solver()
        solver.add(z3_diff != 0)
        result = solver.check()
        if result == unsat:
            return True, None, True
        elif result == sat:
            saw_sat = True
            if counterexample is None:
                model = solver.model()
                counterexample = {v: float(model[s].as_fraction())
                                  for v, s in zip(var_names, syms)}
    if not saw_sat:
        return None, None, True  # both checks came back "unknown": inconclusive
    return False, counterexample, True

# ============================================================================
#  Ideal membership test
# ============================================================================
def ideal_membership(poly_expr, ideal_gens, var_names):
    """
    Check if poly_expr ∈ <ideal_gens>.

    Division by an arbitrary generator list is NOT a membership test: a
    nonzero remainder disproves membership only when dividing by a Gröbner
    basis (and the result was order-dependent besides). Compute the GB of
    the generators first, then reduce; this matches how sr_gb.py's
    deflation loop already tests membership internally.
    """
    from sympy import symbols, Poly, groebner, reduced
    syms = symbols(var_names)
    gb = groebner([Poly(g, *syms, domain='QQ') for g in ideal_gens],
                  *syms, order='grevlex')
    _, rem = reduced(Poly(poly_expr, *syms, domain='QQ').as_expr(),
                     list(gb), *syms, order='grevlex')
    return rem == 0

def algebraic_independence(polys, var_names):
    """
    Check if given polynomials are algebraically independent (Jacobian rank = n).
    """
    from sympy import Matrix, symbols, diff
    syms = symbols(var_names)
    J = Matrix([[diff(p, v) for v in syms] for p in polys])
    rank = J.rank()
    return rank == len(polys)