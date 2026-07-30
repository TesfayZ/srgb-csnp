# DIG baseline: install notes and findings

Cloning [DIG](https://github.com/dynaroars/dig) (dynaroars/dig,
Nguyen et al.) as an empirical baseline against SR-GB-CSNP. 

## Install

- Cloned `https://github.com/dynaroars/dig.git` at commit `d08160438c8ef6a389add5ff4a23220247e2d501`
  (2026-07-04). The repo has moved on from the historical Sage dependency: its
  `pyproject.toml` lists only `sympy`, `z3-solver`, `beartype`, `pycparser`, `numpy`
  as runtime dependencies (plus optional `pytest` / `anthropic` for tests / the
  experimental LLM mode). No Sage install was needed at any point.
- `pyproject.toml` declares `requires-python = ">=3.14"`, but only Python 3.12 is
  available in this sandbox. DIG is "run as flat scripts from `./src`... not
  installed as a package" (comment in `pyproject.toml`), so the `>=3.14` constraint
  is packaging metadata, not an enforced runtime check: a plain venv with Python
  3.12 plus `pip install sympy z3-solver beartype pycparser numpy` was sufficient,
  and `python -O src/dig.py ...` ran without any 3.14-only syntax/stdlib issues.
- Confirmed working end-to-end on DIG's own bundled example first:
  `python -O dig.py ../examples/traces/cohendiv.csv -log 3` reproduced the same
  kind of output shown in DIG's README (17 invariants at `vtrace1`, 8 at
  `vtrace2`, ~55s wall time including inequality/congruence inference), and
  `python -O dig.py ../examples/traces/ex_float.csv -log 3` correctly recovered
  `2*x - y + 3 == 0` from a float-typed trace file for the linear relation
  `y = 2x + 3`. Toolchain confirmed functional before touching any of this
  repo's benchmark systems.

No Docker/Sage was needed; total install time was a few minutes.

## How DIG's equation engine actually works

Traced through `src/helpers/miscs.py` (`Miscs.solve_eqts`, `Miscs._null_space_fast`)
and `src/infer/inv.py` (`Inv.test_single_trace`):

- DIG instantiates a monomial template up to `-maxdeg`, picks `1.5x` as many
  traces as unknowns (`settings.EQT_RATE = 1.5`), and solves for an **exact**
  rational nullspace of the resulting linear system (`sympy.Matrix.nullspace()`,
  with a NumPy SVD used only as a fast full-rank pre-check, never as a
  tolerance/threshold on the residual).
- Every surviving candidate is then tested against **every** trace at that
  location via `bool(self.inv.xreplace(trace.mydict))` — i.e. the candidate
  polynomial must evaluate to **exactly** 0 as a `sympy.Rational`, for every
  single trace. There is no `epsilon`/tolerance anywhere in this path.
- Trace values are read either as type `D` (`sympy.Rational(float(v))`, the
  exact binary64 double the string denotes) or type `F`/`I`
  (`sympy.Rational(str(v))`, an exact decimal fraction). Either way, a trace
  value is turned into an exact rational with no "closeness" semantics.

Consequence: DIG's nonlinear-equality inference is designed for **exact**
program traces (integer loop counters, or floats that satisfy a relation to
machine precision by construction, e.g. C doubles printed as `%.17g` from a
program whose invariant is enforced by integer/rational arithmetic). It has no
mechanism analogous to SR-GB-CSNP's SVD-nullspace-with-noise-threshold +
snap-rounding pipeline for recovering a relation that only holds
*approximately* on the data.

## What was run on this repo's benchmark systems

This paper's adjacent script (not committed; see `Results/dig_baseline_results.csv`
for outputs) generated N=300-point traces at sigma in `{0.00, 0.02, 0.05}`
(matching `benchmark_feyman.py`'s own noise grid) for:

- `circle` (`x**2+y**2-1`), `sphere` (`x**2+y**2+z**2-1`), `cubic_x3y3`
  (`x**3+y**3-1`, degree-3 implicit variety) via this repo's
  `data_generator.generate_variety_data`
- `kepler_angular_momentum`, reusing `benchmark_kepler_angular_momentum.py`'s
  exact `kepler_analytic` generator and its transition invariant
  `x_t*vy_t - y_t*vx_t - x_next*vy_next + y_next*vx_next`
- `harmonic_oscillator_2d`, reusing `benchmark_harmonic_oscillator_vs_sindy.py`'s
  `generate_harmonic_trajectory` and its transition invariant
  `x_t**2 + v_t**2 - x_next**2 - v_next**2`
- 4 equations from `feynman_polynomials.py` spanning degree 2 and 3
  and 3-5 variables: `circle_locus`, `angular_momentum_2d`, `I.12.2`, `I.39.22`
  (labels as of the 2026-07-22 ID-audit rename; at the time this baseline was
  run they were named `I.9.18`, `I.18.4`, `I.12.3`, `II.11.28`, but the
  underlying equations are unchanged)

Each trace was written in DIG's trace CSV format with type `D` (exact double,
`repr(float(x))` per value) so DIG sees exactly the same floating-point numbers
SR-GB-CSNP consumes, with no extra decimal-rounding error introduced by the
harness. DIG was run as:
`python -O dig.py <trace.csv> -noieqs -nocongruences -nominmaxplus -maxdeg <deg> -log 3`
(inequality/octagon/congruence inference disabled since only the polynomial-
equality output is comparable to SR-GB-CSNP).

### Result: 0/27 runs recovered any equality invariant, at every sigma including 0.00

Every one of the 9 systems x 3 sigma levels (27 runs) produced
`NO EQTS RESULTS, reducing deg to ...` all the way down to degree 0, i.e. DIG
found **zero** candidate equalities, not merely a wrong or non-minimal one, at
**every** noise level, including nominal `sigma=0.0`. Runtimes were 2-10s per run.

This is not simply "DIG can't handle Gaussian noise" — even `sigma=0.0`
trajectories fail, because `data_generator.generate_variety_data` and the
Kepler/harmonic generators evaluate trig/sqrt functions in IEEE-754 double
arithmetic, so e.g. `x**2+y**2-1` is never *exactly* 0 as a rational, only
`~1e-16`-close to it (verified directly: `max(abs(x**2+y**2-1))` over 300
circle points was `2.22e-16`, not `0`). DIG's exact-rational vanishing check
rejects this residual outright, exactly as the traced-through code predicts.

### Positive control (confirms the negative result is real, not a harness bug)

To rule out "the harness is broken" as an alternative explanation, one circle
trace was built from an **exact rational parametrization**
(`x,y = (1-t^2)/(1+t^2), 2t/(1+t^2)` for rational `t = i/7`, `i=1..59`, i.e.
Pythagorean-triple-like points), so `x**2+y**2-1` vanishes exactly as a sympy
Rational by construction. On this control trace DIG correctly and immediately
recovered `x**2 + y**2 - 1 == 0` (see the `CONTROL_circle_exact_rational` row
in `Results/dig_baseline_results.csv`). This confirms the DIG install, the
trace format, and the harness are all correct, and that the 27/27 failures
above are a genuine property of DIG's exact-equality inference applied to
floating-point physical trajectory data, not a setup mistake.

## For the paper

DIG is a related prior work (SVD-nullspace + exact linear-algebra
equality inference over traces, the same high-level idea as
SR-GB-CSNP's monomial-lifting + nullspace step), but its verification step
requires **exact rational vanishing** of the candidate polynomial on every
trace, with no residual-tolerance / thresholding mechanism analogous to
SR-GB-CSNP's SVD-based estimation + MDL-driven snap-rounding. It is built for
discrete/integer program-invariant traces (loop counters, array indices) or
floating traces that are exact by construction, not continuous noisy physical
measurements. On every one of this paper's benchmark systems, at every noise
level tested (including nominal zero added noise, where only IEEE-754 rounding
is present), DIG recovered zero invariants; the positive control shows this is
DIG behaving exactly as designed, not an integration failure on our part.

## Reproducing

The DIG authors own github repo as presented above 
