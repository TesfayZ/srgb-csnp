"""
test_safe_pruning_counterexample.py - Regression test for the Normal-Form
Pruning Lemma (paper Lemma 4.3, Section 4.3) and for two bugs the review
process found in prune_active_monomials (sr_gb.py):

1. The pruning check silently never removed anything: Poly.LM() returns a
   sympy Monomial object, not an Expr, and passing that into div() always
   raised PolificationFailed, swallowed by a bare except/continue. Any
   input to prune_active_monomials that should prune at least one monomial
   is therefore also a regression guard against this becoming a no-op again.
2. The leading monomial was taken under sympy's default (lex) order via an
   arbitrarily-ordered free_symbols set, rather than the grevlex order used
   everywhere else in the pipeline. Lemma 4.3's degree bound
   deg(NF_G(p)) <= deg(p) only holds for a GRADED order, so a mixed-degree
   generator (E - h*nu, Feynman-style) needs its grevlex leading monomial,
   not its lex one.

The paper's own counterexample to the *original*, incorrect Safe Pruning
Lemma is also checked directly: I_G = <x^2 - y>, I = <x^2, y>, p = x^2 has
its entire support divisible by LM(x^2-y) = x^2, so the original lemma's
claim (some monomial of p always survives) is false. The corrected lemma
claims this instead about NF_G(p) = y, which is what pruning must expose.
"""

from sympy import symbols, div
from sr_gb import prune_active_monomials


def test_pruning_is_not_a_no_op():
    x, y = symbols('x y')
    g = x**2 - y
    monomials = [1, x, y, x**2, x*y, y**2]
    log = []
    pruned = prune_active_monomials(monomials, [g], sym_vars=[x, y], log=log)
    assert log[0]['n_pruned'] > 0, (
        "prune_active_monomials removed nothing; a Monomial/Expr type mismatch "
        "would make pruning a silent no-op"
    )
    print("Pruning is not a no-op: removed", log[0]['n_pruned'], "of", log[0]['n_active'])


def test_paper_counterexample_x2_minus_y():
    x, y = symbols('x y')
    g = x**2 - y
    monomials = [1, x, y, x**2, x*y, y**2]
    pruned = prune_active_monomials(monomials, [g], sym_vars=[x, y])

    assert x**2 not in pruned, "x^2 should be pruned: it is LM(x^2-y) itself"
    assert y in pruned, "y must survive pruning: it is NF_G(x^2), the corrected lemma's witness"

    # NF_G(p) = y is exactly what division by g leaves as remainder.
    _, remainder = div(x**2, g, x, y, domain='QQ')
    assert remainder == y
    print("Counterexample reproduced: x^2 pruned, y survives, matches NF_G(x^2) = y.")


def test_grevlex_not_lex_leading_monomial():
    # E - h*nu is degree 1 in E, degree 2 in h*nu. Under grevlex (used by
    # the rest of the pipeline), LM = h*nu since higher total degree wins.
    # Under lex with E listed first among the generators, LM = E instead,
    # since lex compares the exponent of the first generator only.
    E, h, nu = symbols('E h nu')
    g = E - h * nu
    monomials = [1, E, h, nu, E**2, h*nu, h**2, nu**2]

    pruned = prune_active_monomials(monomials, [g], sym_vars=[E, h, nu])

    assert h*nu not in pruned, "h*nu is LM(E - h*nu) under grevlex and must be pruned"
    assert E in pruned, (
        "E must survive: it is not divisible by the grevlex leading monomial h*nu "
        "(a lex-ordered computation with E listed first would wrongly prune E instead)"
    )
    print("grevlex leading monomial correctly identified as h*nu, not E.")


def test_degree_bound_lets_true_invariant_survive_to_its_degree():
    # A minimal end-to-end check that a degree-3 true invariant's support
    # is not wiped out by pruning against a confirmed degree-2 generator,
    # since Lemma 4.3(iv) guarantees deg(NF_G(p)) <= deg(p).
    x, y, z = symbols('x y z')
    confirmed = x**2 - y  # degree 2, LM = x^2 under grevlex
    degree3_monomials = [x**3, x**2*y, x*y*z, y*z**2, z**3, x*z**2]
    pruned = prune_active_monomials(degree3_monomials, [confirmed], sym_vars=[x, y, z])

    assert x**3 not in pruned, "x^3 = x*x^2 is divisible by LM(x^2-y) and should be pruned"
    assert x**2*y not in pruned, "x^2*y is divisible by LM(x^2-y) and should be pruned"
    assert x*y*z in pruned, "x*y*z is not divisible by x^2 and must survive"
    print("Degree-3 monomials not divisible by the confirmed LM survive pruning, as required.")


if __name__ == "__main__":
    test_pruning_is_not_a_no_op()
    test_paper_counterexample_x2_minus_y()
    test_grevlex_not_lex_leading_monomial()
    test_degree_bound_lets_true_invariant_survive_to_its_degree()
    print("All safe-pruning counterexample tests passed.")
