# SR-GB+CSNP: Canonical Minimal Polynomial Invariants from Noisy Data via Adaptive Sparse Nullspace Recovery and Gröbner Basis

**Document(paper preprint):** [PDF](https://zenodo.org/records/21836794)
**Package:** Initial release available at [`polyinv`](https://github.com/TesfayZ/polyinv) — install via `pip install polyinv`.


**SR-GB-CSNP** is a Python framework for discovering **minimal, canonical polynomial invariants** from noisy observational data.

Unlike conventional symbolic regression, which returns a single expression, **SR-GB-CSNP** recovers an **algebraic equivalence class** represented by a **reduced Gröbner basis**. This produces a unique, irredundant representation of the inferred polynomial ideal, removing redundant equations while improving interpretability.

The framework combines:

1. **Monomial lifting** into a polynomial feature space.
2. **Nullspace estimation** using Singular Value Decomposition (SVD).
3. **Combinatorial Sparse Nullspace Pursuit (CSNP)**, which searches over monomial supports while enforcing the exact rank-deficiency condition (Proposition 4.1) to identify geometrically valid candidates.
4. **Lexicographic Minimum Description Length (MDL) selection**, choosing among valid candidates by:

   * smallest **rationality score** (coefficient simplicity),
   * then smallest **sparsity** (number of active monomials),
   * then smallest **geometric uniqueness** (`s_min`) as a tie-breaker.
5. **Noise-tolerant rational snap-rounding** to recover exact rational coefficients.
6. **Reduced Gröbner basis computation** over **ℚ** to obtain a canonical generating set of the inferred polynomial ideal.

The resulting representation is suitable for scientific discovery, symbolic model reduction, system identification, and constraint validation.

---

# Features

* ✅ Exact recovery of polynomial invariants from noisy data (100% up to **σ = 0.14** on the circle benchmark, partial recovery to σ = 0.20)
* ✅ Sparse nullspace recovery via **CSNP**

  * Fast mode (`d = 2`)
  * General combinatorial mode (`d ≥ 3`)
  * Automatic **L1 fallback**
* ✅ Lexicographic MDL model selection without manually tuned weights
* ✅ Rational coefficient reconstruction via snap-rounding
* ✅ Canonicalization using reduced Gröbner bases (grevlex order)
* ✅ Transition invariant discovery for dynamical systems
* ✅ Comprehensive benchmark suite
* ✅ Comparisons against SINDy-null (KRONIC), SINDy-ST, SINDy-FD/AD, Dense SVD+GB, OMP, AVI, and PySR baselines

---

# Installation

## Requirements

* Python 3.9+
* NumPy
* SciPy
* SymPy
* scikit-learn
* pandas
* matplotlib

Optional:

* PySR (for baseline comparisons)
* Z3 (for SMT-based invariant verification)
* MuJoCo (for robotics trajectory benchmarks)

## Install from Source

```bash
git clone https://github.com/TesfayZ/srgb-csnp.git
cd srgb-csnp
pip install -r requirements.txt
```

---

# Repository Structure

```
SR-GB-CSNP/
│
├── sr_gb.py                 # core pipeline
├── data_generator.py        # synthetic variety sampling
├── feynman_polynomials.py   # Feynman benchmark equations
├── sindy_baselines.py       # SINDy-family baselines
├── avi_baselines.py         # AVI / approximate border-basis baseline
├── verification.py          # SOS/SMT verification helpers
├── robot_data.py            # MuJoCo trajectory generation
├── vortex_data.py           # point-vortex (Kirchhoff) trajectory generation
├── utils_stats.py           # Wilson interval / stats helpers
│
├── benchmark_*.py
├── ablation_*.py
├── test_*.py
├── _*.py                    # underscore-prefixed probe/diagnostic scripts,
│                            # registered in run_all_experiments.py alongside
│                            # the benchmarks they back
│
├── bio_chem_attempts.py     # real biochem attempts (glycolysis + CRN), not in paper
├── crn_data/                # chemical-reaction-network (CRN) model data
├── glycolysis_data/         # Wolf2000 / Bier2000 glycolysis model data
│
├── generate_result_tables.py  # Results/*.csv -> LaTeX tables
├── run_all_experiments.py     # orchestrates benchmarks/ablations/tests
├── Results/                    # CSV outputs + generated LaTeX tables
└── Adaptive_CSNP/               # paper source (Adaptive-CSNP.tex)
```

The `SCRIPTS` dict in `run_all_experiments.py` is the authoritative, current list of every benchmark/ablation/test script — prefer it over this README if the two ever disagree. The paper itself names no scripts or CSV files in its main text; every experimental claim there is backed by one of the scripts below, and this file is the map from claim to script to output file.

---

# Scripts and Paper Mapping

Paper references below point to content (a table's subject, or a section name) rather than a table number, since numbering shifts as the paper is revised; cross-check against the caption text in `Adaptive_CSNP/Adaptive-CSNP.tex` or `Results/all_paper_tables.tex`.

## Core Implementation

| Script                   | Purpose                                                                                                          |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `sr_gb.py`               | Main implementation of SR-GB+CSNP (monomial lifting, SVD, CSNP, branch-and-bound, snap-rounding, Gröbner basis). |
| `data_generator.py`      | Synthetic data generation (`generate_variety_data`) for most benchmark problems.                                 |
| `feynman_polynomials.py` | Polynomial Feynman benchmark equations (26 total).                                                               |
| `sindy_baselines.py`     | SINDy-null (KRONIC) / SINDy-ST / SINDy-FD / SINDy-AD baseline implementations used across benchmarks.            |
| `avi_baselines.py`       | Approximate Vanishing Ideal / approximate border-basis baseline (Heldt–Kreuzer–Pokutta–Poulisse construction).   |
| `verification.py`        | SOS feasibility (sampling-based) and Z3-based SMT verification of recovered invariants.                          |
| `robot_data.py`          | MuJoCo trajectory generation for `benchmark_robotics.py` / `benchmark_gravity_pendulum.py` (no dummy-data fallback: these scripts abort loudly if MuJoCo is unavailable). |
| `vortex_data.py`         | Point-vortex (Kirchhoff equations) trajectory generation for `benchmark_vortex.py`.                               |
| `utils_stats.py`         | `wilson_interval` and other statistical utilities used to report recovery rates everywhere.                       |

---

## Benchmarks

| Script                                      | Produces (`Results/*.csv`)                                          | Paper reference |
| -------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------- |
| `benchmark_redundancy_elimination.py`       | `redundancy_elimination_full.csv`, `redundancy_elimination_summary.csv` | Synthetic Benchmarks: Redundancy Elimination |
| `benchmark_harmonic_oscillator_vs_sindy.py` | `harmonic_oscillator_sindy_comparison.csv`                            | Transition Invariants |
| `benchmark_kepler_angular_momentum.py`      | `kepler_angular_momentum_results.csv`                                 | Transition Invariants |
| `benchmark_feyman.py`                       | `feynman_results_overall.csv`, `feynman_results_full.csv`, `feynman_results_summary.csv` | Feynman Polynomial Benchmark (overall table, baseline comparison, and the full per-equation appendix table) |
| `benchmark_linear_holonomic.py`, `benchmark_nonlinear_holonomic.py` | `linear_holonomic_equality_results.csv`, `linear_holonomic_equality_results_summary.csv`, `nonlinear_holonomic_results.csv`, `nonlinear_holonomic_results_summary.csv` | Holonomic Equality Constraints |
| `benchmark_loop_invariants.py`              | `loop_invariants_full.csv`, `loop_invariants_summary.csv`             | Extended Results: Loop Invariants and DIG (appendix) |
| `benchmark_omp_nullspace.py`                | `omp_nullspace_results.csv`, `omp_nullspace_summary.csv`              | Comparison with OMP on Nullspace |
| `benchmark_overlifting_circle.py`           | `overlifting_circle_results.csv`, `overlifting_circle_summary.csv`    | Degree Lifting Robustness |
| `benchmark_runtime_breakdown.py`            | `runtime_breakdown.csv`, `runtime_breakdown_summary.csv`              | Computational Complexity (the cost breakdown is stated in prose, not a generated table) |
| `benchmark_pysr_gb_implicit_vs_explicit.py` | `pysr_gb_circle_results.csv`, `pysr_gb_fma_results.csv`, `pysr_gb_combined_results.csv` | PySR-GB Baseline |
| `benchmark_deflation_multi_invariant.py`    | `deflation_multi_invariant_results_fixed.csv`                        | Multiple Invariants via Nullspace Deflation / Transition Invariants |
| `benchmark_vortex.py`                       | `vortex_results.csv`, `vortex_summary.csv`                            | Transition Invariants (point-vortex multi-invariant table) |
| `benchmark_gb_scalability.py`               | `gb_scalability.csv`                                                  | Gröbner Basis Scalability |
| `benchmark_sos_sensitivity.py`              | `sos_sensitivity_results.csv`                                        | SOS sensitivity to redundant generating sets |
| `benchmark_robotics.py`                     | `benchmark_robotics.csv`                                              | robotics positive control, cited in Limitations |
| `benchmark_gravity_pendulum.py`             | `benchmark_gravity_pendulum.csv`                                      | robotics negative control (transcendental energy), cited in Limitations |
| `benchmark_smt_verification.py`             | `benchmark_smt_verification.csv`                                     | Z3 ideal-equality check mentioned in the Experiments introduction |
| `benchmark_dt_sweep_modified_equation.py`   | `dt_sweep_exact_gap.csv`, `dt_sweep_direct_projection_results.csv`, `dt_sweep_direct_projection_summary.csv`, `dt_sweep_degeneracy_probe.csv` | Modified-Equation Separation (main text and the closed-form-derivation appendix) |
| `benchmark_difference_dictionary_generality.py` | `difference_dictionary_generality_results.csv`, `difference_dictionary_generality_rates.csv` | Modified-Equation Separation: What Remains Open |
| `benchmark_avi_baseline.py`                 | `avi_baseline_results.csv`, `avi_baseline_summary.csv`                | AVI Baseline (main text) and the AVI appendix table |
| `benchmark_bb_search_stats.py`              | `bb_search_stats_results.csv`, `bb_search_stats_summary.csv`          | Global Optimality of Branch-and-Bound ("Search efficiency, measured directly") |
| `benchmark_dt_discriminator.py`             | `dt_discriminator_results.csv`, `dt_discriminator_summary.csv`        | Fixed-$dt$ versus Variable-$dt$: The Discriminator Table |
| `benchmark_unit_scale_sensitivity.py`       | `unit_scale_sensitivity_results.csv`, `unit_scale_sensitivity_summary.csv` | Unit-Dependence of the Rationality Prior |
| `benchmark_oracle_misclassification.py`    | `oracle_misclassification_full.csv`, `oracle_misclassification_summary.csv` | Global Optimality of Branch-and-Bound ("The oracle hypothesis can fail") |
| `_noise_ceiling_probe.py`                   | `noise_ceiling_probe.csv`, `noise_ceiling_probe_summary.csv`          | Noise Ceiling for Higher-Degree Invariants |
| `_verify_bootstrap_reduction.py`            | `bootstrap_reduction_verification.csv`                                | supports the SINDy-null redundancy claim in Synthetic Benchmarks |
| `_feynman_timing_probe.py`                  | `feynman_timing_probe.csv`, `feynman_timing_probe_full.csv`           | the compute-bill paragraph in the Feynman Polynomial Benchmark section |

A further script, `bio_chem_attempts.py`, is an **exploratory attempt** (not a paper benchmark and not wired into `run_all_experiments.py`) on five real biochemical systems: the Wolf2000/Bier2000 glycolytic oscillators (biology) and the Gardner/Kholodenko/Markevich CRN signaling models (chemistry). Per system it runs SR-GB+CSNP and the four nullspace/vanishing-ideal baselines on cached trajectories. SR-GB+CSNP recovers no new polynomial invariant on any of them (root cause: independent-IC perturbation breaks the pooled linear totals, and the Wolf2000 model eliminates ADP/NADH so the moiety SVD lacks a constant column), and the baselines fare no better; the script is kept so the attempt stays reproducible and honestly recorded. It supersedes the earlier `benchmark_crn.py`, `benchmark_glycolysis.py`, and `benchmark_real_system_baselines.py`.

DIG (Nguyen et al.) is compared against directly by running the external `dynaroars/dig` tool rather than a script in this repository; see `DIG_INSTALL_NOTES.md` and the paper's "Extended Results: Loop Invariants and DIG" appendix.

---

## Ablation Studies

| Script                                      | Produces (`Results/*.csv`)                                    | Paper reference |
| -------------------------------------------- | ---------------------------------------------------------------- | ---------------- |
| `ablation_degree_circle.py`                 | `ablation_adaptive_degree_circle.csv`, `ablation_fixed_degree_results_full.csv`, `ablation_fixed_degree_summary.csv` | Adaptive degree discovery vs. fixed-degree recovery (appendix tables) |
| `ablation_noise_circle_snap_vs_original.py` | `noise_ablation_circle_full.csv`, `noise_ablation_circle_rates.csv` | Noise Sensitivity (panel (a): naïve rounding vs. snap-rounding) |
| `ablation_noise_ar1_circle.py`              | `ablation_noise_ar1_circle_results.csv`, `ablation_noise_ar1_circle_rates.csv` | Noise Sensitivity (panel (b): AR(1)-correlated noise) |
| `ablation_sample_size_circle.py`            | `sample_size_ablation.csv`                                     | Sample Size and Degree Sensitivity |
| `ablation_mdl_stopping.py`                  | `ablation_mdl_stopping.csv`                                     | MDL Stopping and Global Degree Selection (stop-vs-fixed-degree-oracle check) |
| `ablation_qmax_tolerance.py`                | `ablation_qmax_recovery.csv`, `ablation_qmax_recovery_summary.csv`, `ablation_qmax_falsepos.csv`, `ablation_qmax_falsepos_summary.csv`, `ablation_eps_sensitivity.csv`, `ablation_eps_sensitivity_summary.csv` | `sec:qmax-ablation` (`tab:qmax-recovery`, `tab:qmax-falsepos`, `tab:eps-sensitivity`) |
| `ablation_noisy_rank_guard.py`              | `ablation_noisy_rank_guard_*.csv`                             | Feasibility Classification / noisy-rank guard sensitivity (`sec:bb`, `sec:global-opt`) |
| `ablation_avi_eps_multiplier.py`            | `ablation_avi_eps_multiplier*.csv`                            | AVI Baseline eps-multiplier calibration (`sec:avi-baseline`) |
| `ablation_sparsity_vs_rationality_harmonic.py` | `ablation_sparsity_vs_rationality_harmonic_*.csv`          | Sparsity-first vs. rationality-first selection, fixed-$dt$ harmonic oscillator (`sec:transition`) |
| `ablation_sindy_snap_gb_hybrid.py`          | `sindy_snap_gb_hybrid_results.csv`, `sindy_snap_gb_hybrid_summary.csv` | SINDy-null candidates + snap-rounding + one shared Gröbner-basis call, on the over-lifted circle (`app:baseline-rationale`) |

---

## Validation Tests

| Script                                         | Purpose                                                                                        |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `test_auto_invariants.py`                      | Automatic invariant discovery (no `expected_support`) on harmonic oscillator and Kepler problems. |
| `test_l1_fallback_4sphere.py`                  | L1 fallback stress test (4-sphere invariant, M=35, over combinatorial budget).                  |
| `test_rationality_cost.py`                     | Rationality-cost function unit tests.                                                           |
| `test_bb_global_optimality.py`                 | Branch-and-bound global-optimality verification against brute force.                            |
| `test_rationality_resolves_dt_ambiguity.py`    | Rationality cost resolves the dt-induced nullspace ambiguity (fixed-dt harmonic oscillator).    |
| `test_algebraic_minimality.py`                 | Reduction of a non-minimal generator (e.g. `y*(x^2+y^2-1)`) to its minimal form.                |
| `test_degree_minimality.py`                    | Degree-minimality check verification.                                                           |
| `test_safe_pruning_counterexample.py`          | The Normal-Form Pruning Lemma's counterexample (Section "Algebra-Guided Pruning").              |
| `test_unified_search_optimality.py`            | `unified_circuit_search`'s routing agrees with a brute-force reference on small cases.          |
| `validate_sr_gb.py`                            | PASS/FAIL smoke tests (preconditioning, full-nullspace deflation, prebuilt-Phi); run this after any change to `sr_gb.py`. |
| `verify_claims.py`                             | Reproduces the paper's closed-form derivations and worked examples (the rationality-cost floor, the cos(0.1) worked example, the denominator-ceiling normalisation convention) in under a second, against `sr_gb.py`'s actual functions rather than a re-derivation. Complements `benchmark_dt_sweep_modified_equation.py` (Q_dt closed forms) and `test_safe_pruning_counterexample.py` (Normal-Form Pruning Lemma), which already cover the other closed-form claims. |

---

## Experiment Orchestration

| Script                       | Purpose                                                                                        |
| ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| `run_all_experiments.py`     | Runs benchmarks/ablations/tests (`--category`, `--quick`, `--skip-tests`, `--generate-tables`). |
| `generate_result_tables.py`  | Builds LaTeX tables from `Results/*.csv` (`Results/all_paper_tables.tex`); several paper tables are hand-extended beyond what it emits (see `CLAUDE.md`). |

---

# Usage

Run an individual benchmark:

```bash
python benchmark_redundancy_elimination.py

python benchmark_feyman.py

python benchmark_kepler_angular_momentum.py
```

Run every experiment:

```bash
python run_all_experiments.py
```

Run a faster smoke test:

```bash
python run_all_experiments.py --quick
```

Run one category only, and regenerate LaTeX tables afterward:

```bash
python run_all_experiments.py --category benchmarks --generate-tables
```

---

# Practitioner's Guide

Operational guidance for running the pipeline on new data. The paper's
"A Practitioner's Protocol" appendix gives the condensed version; this
section is the fuller reference.

## Choosing N

* Single-invariant (d = 1) problems at moderate noise: N >= 100 already
  achieves 100% exact recovery on the unit circle at sigma = 0.01
  (`ablation_sample_size_circle.py`). This is not a general guarantee,
  but it is the right order of magnitude to start from.
* Transition-invariant and multi-dimensional-nullspace problems need
  substantially more. Every such benchmark in the paper uses N = 5000;
  smaller samples degrade recovery on the same systems even where the
  underlying invariant is unchanged.

## Reading an abstention

A run returning no invariant has at least three distinct causes, not
mutually exclusive. The pipeline does not distinguish them
automatically, so the checks below are manual:

1. **No polynomial invariant exists at the searched degrees.**
   Diagnostic: the MDL history is flat or monotonically increasing
   across the whole swept degree range, with no interior minimum.
2. **The coordinates are in the wrong units** (the rationality prior is
   not scale-invariant). Diagnostic: rescale each variable by its own
   empirical standard deviation and rerun; if the outcome changes, units
   were the cause. This probe diagnoses the problem but does not
   reliably fix it (see the paper's unit-invariance stress test and
   `benchmark_unit_scale_sensitivity.py`).
3. **Noise exceeds the degree-dependent ceiling.** Diagnostic: a
   candidate is found and scored but rejected at the final residual
   gate, rather than never found at all; the two cases look identical
   from the return value alone but differ in the internal search log.

## Reading an L1-fallback answer

When both exact search engines exceed their budgets and the pipeline
falls back to the LASSO-seeded anytime search, the returned candidate
carries no optimality certificate, unlike every branch-and-bound
result in the paper. Treat it as a hypothesis for external
verification, for example an exact algebraic check against
independently collected data (`verification.py` provides Z3-based
checks), not as equivalent to a certified branch-and-bound output.

---

# Benchmarks Included

The repository contains experiments covering:

* Polynomial invariant recovery
* Algebraic redundancy elimination
* Noise robustness
* Sample complexity
* Runtime analysis
* Sparse nullspace recovery
* Harmonic oscillator invariants
* Kepler angular momentum
* Polynomial Feynman equations
* Linear and nonlinear holonomic constraints
* Program loop invariants
* Real biochemical systems (glycolytic oscillators + CRN signaling) — exploratory attempts, see `bio_chem_attempts.py`
* MuJoCo robotics trajectories
* SOS and SMT (Z3) verification of recovered invariants
* PySR comparisons
* OMP comparisons

---
## Citation

If you use the paper or this software in your research, please cite both separately.

### Paper citation

```bibtex
@misc{Gebrekidan2026srgb_csnp,
  author       = {Tesfay Zemuy Gebrekidan},
  title        = {Canonical Minimal Polynomial Invariants from Noisy Data via Adaptive Sparse Nullspace Recovery and Gröbner Basis},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21836794},
  url          = {https://doi.org/10.5281/zenodo.21836794},
}
```

### Software citation

```bibtex
@software{Gebrekidan2026polyinv,
  author       = {Tesfay Zemuy Gebrekidan},
  title        = {polyinv: Canonical Minimal Polynomial Invariant Computation Package - based on the paper - Canonical Minimal Polynomial Invariants from Noisy Data via Adaptive Sparse Nullspace Recovery and Gröbner Basis},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.21838053},
  url          = {https://doi.org/10.5281/zenodo.21838053},
}
```

---
# License

The code in this repository is released under the **MIT license** (see
`LICENSE`). 
