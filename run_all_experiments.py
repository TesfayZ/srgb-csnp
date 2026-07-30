#!/usr/bin/env python3
"""
run_all_experiments.py – Orchestrate benchmarks, ablations, validation, and tests.

Usage:
    python run_all_experiments.py                     # Full run
    python run_all_experiments.py --quick             # Quick mode
    python run_all_experiments.py --category benchmarks  # Benchmarks only
    python run_all_experiments.py --category ablations   # Ablations only
    python run_all_experiments.py --category tests       # Unit tests only
    python run_all_experiments.py --category validation_and_verification  # Integration validation + verification only
    python run_all_experiments.py --generate-tables      # Generate LaTeX tables after
"""

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

SCRIPTS = {
    "benchmarks": [
        # Table 3: Redundancy elimination
        "benchmark_redundancy_elimination.py",
        # Table 5: Transition invariants
        "benchmark_harmonic_oscillator_vs_sindy.py",
        "benchmark_kepler_angular_momentum.py",
        # Table 7: Feynman
        "benchmark_feyman.py",
        # Table 8: Holonomic
        "benchmark_linear_holonomic.py",
        "benchmark_nonlinear_holonomic.py",
        "benchmark_loop_invariants.py",
        # Table 9: OMP vs CSNP
        "benchmark_omp_nullspace.py",
        # Table 4: Degree overlifting on circle
        "benchmark_overlifting_circle.py",
        # PySR-GB comparison
        "benchmark_pysr_gb_implicit_vs_explicit.py",
        # Deflation multi-invariant benchmark
        "benchmark_deflation_multi_invariant.py",
        # Point-vortex physical multi-invariant benchmark (full-nullspace deflation)
        "benchmark_vortex.py",
        # Groebner-stage scaling table (tab:gb-scalability)
        "benchmark_gb_scalability.py",
        "benchmark_robotics.py",
        # AVI border-basis baseline
        "benchmark_avi_baseline.py",
        # Fixed-dt vs variable-dt discriminator (Section "Fixed-dt versus Variable-dt")
        "benchmark_dt_discriminator.py",
    ],
    "ablations": [
        "ablation_degree_circle.py",
        "ablation_noise_circle_snap_vs_original.py",
        # Correlated-noise robustness: AR(1) measurement noise on the circle
        # (phi = 0 iid control), answering the "all noise is iid Gaussian"
        # reviewer question.
        "ablation_noise_ar1_circle.py",
        "ablation_sample_size_circle.py",
        "ablation_mdl_stopping.py",
        # Qmax (rational-denominator cap) and eps (snap tolerance) sensitivity:
        # recovery-vs-Qmax ceiling, false-positive-vs-Qmax, and eps robustness.
        "ablation_qmax_tolerance.py",
        # Sensitivity of the noisy-rank dominance-pruning guard's multiplier
        # and the AVI baseline's eps multiplier, both hand-set constants.
        "ablation_noisy_rank_guard.py",
        "ablation_avi_eps_multiplier.py",
        # Sparsity-first vs rationality-first selection on the fixed-dt
        # harmonic oscillator's transition-pair dictionary (isolates
        # sparsity vs. rationality as the only variable in SR-GB+CSNP's own
        # search).
        "ablation_sparsity_vs_rationality_harmonic.py",
        # Runs the "SINDy + snap-rounding + GB" hybrid that
        # app:baseline-rationale argues against but never executes: SINDy-
        # null's candidate list fed jointly into one Groebner-basis call,
        # on the over-lifted circle.
        "ablation_sindy_snap_gb_hybrid.py",
    ],
    "tests": [
        # Reproduces the paper's closed-form derivations and worked
        # examples (rationality-cost floor, the cos(0.1) worked example,
        # the Normal-Form Pruning Lemma counterexample, the denominator-
        # ceiling normalisation convention) in well under a minute, calling
        # sr_gb.py's actual functions rather than re-deriving the formulas,
        # so it also catches paper/code drift.
        "verify_claims.py",
        "test_auto_invariants.py",
        "test_l1_fallback_4sphere.py",
        "test_rationality_cost.py",
        "test_bb_global_optimality.py",
        "test_rationality_resolves_dt_ambiguity.py",
        "test_algebraic_minimality.py",
        "test_degree_minimality.py",
        "test_safe_pruning_counterexample.py",
        "test_unified_search_optimality.py",
    ],
    # End-to-end integration checks, correctness/soundness verification of
    # specific pipeline claims, and diagnostic/instrumentation probes. Kept
    # separate from "benchmarks" (which compares recovery rates against
    # baselines and backs numbered paper tables) and from the short
    # assert-based unit tests in "tests".
    "validation_and_verification": [
        # End-to-end integration/regression smoke tests; these deliberately
        # exercise high-dimensional BB paths.
        "validate_sr_gb.py",
        # Pipeline-stage wall-clock profiling.
        "benchmark_runtime_breakdown.py",
        # SOS solver sensitivity to redundant constraints (preliminary
        # illustration of the reduced-Groebner-basis feasibility check).
        "benchmark_sos_sensitivity.py",
        # Negative counterpart to benchmark_robotics.py: real gravity
        # pendulum, transcendental (non-polynomial) energy, verifies the
        # pipeline correctly abstains rather than reporting a false positive.
        "benchmark_gravity_pendulum.py",
        # SMT-based verification of discovered invariants (Z3 if available).
        "benchmark_smt_verification.py",
        # Modified-equation separation gate (exact, symbolic).
        "benchmark_dt_sweep_modified_equation.py",
        # Difference-dictionary generality probe: sigma>0 + Stormer-Verlet
        # (upgrades sec:modified-equation-future's sigma=0 single-integrator
        # evidence to a measured claim).
        "benchmark_difference_dictionary_generality.py",
        # BB search node statistics (nodes popped/evaluated/pruned).
        "benchmark_bb_search_stats.py",
        # Unit-invariance stress test (Limitations: Unit-Dependence of the Rationality Prior).
        "benchmark_unit_scale_sensitivity.py",
        # Feasibility-oracle misclassification diagnostic (Theorem 4.4 soundness).
        "benchmark_oracle_misclassification.py",
        # Probe/diagnostic scripts (underscore-prefixed). These back paper
        # claims; they are registered here so they run with the rest of the suite.
        # Noise-ceiling probe feeds Table~\ref{tab:noise-ceiling-probe}
        # (degree-2 vs degree-3 nullspace L2 error and gap ratio).
        "_noise_ceiling_probe.py",
        # Bootstrap-reduction verification for the SINDy-null redundancy claim.
        "_verify_bootstrap_reduction.py",
        # Per-cell timing diagnostic for the Feynman benchmark (30 seeds).
        "_feynman_timing_probe.py",
    ],
}

# Every script in SCRIPTS accepts --quick (either it genuinely reduces
# seeds/N/sweep ranges, or -- for scripts with no argparse of their own --
# the flag is simply present in sys.argv and ignored, which is harmless).
#
# Full-mode (non---quick) timeouts (seconds). Set from real measured
# full-scale runtimes (30 seeds, N=5000 where applicable) on a local CPU,
# with roughly 3x headroom, not guessed -- a blanket 3600s/1h default was
# both too tight for a few genuinely slow scripts and wildly wasteful for
# the many that finish in seconds.
#
# The full runs may execute on a slower, more variable CPU (e.g. a hosted
# notebook) than the local machine these numbers were measured on, where wall
# time can run ~1.2-1.7x the local figure and is not uniform across scripts:
# search-heavy cells (the two circle ablations, avi_baseline, Feynman) blow up
# per-seed under noise more than the rest. Search-heavy scripts therefore get
# a wider margin above their measured local runtime than the roughly 3x
# default. Measured local runtimes (reference for the ceilings below):
#   redundancy_elimination 178s, harmonic_oscillator_vs_sindy 487s,
#   kepler_angular_momentum 28s, linear_holonomic 31s, nonlinear_holonomic 60s,
#   loop_invariants 889s, omp_nullspace 521s, overlifting_circle 8s,
#   runtime_breakdown 3s, deflation_multi_invariant 29s, sos_sensitivity 3s,
#   robotics 5s (with a working MuJoCo build; 30/30 exact recovery of the
#     harmonic-oscillator energy invariant). 1800s is kept as headroom for a
#     slower/noisier MuJoCo environment; this benchmark is local-only where
#     MuJoCo is unavailable,
#   smt_verification 5s, ablation_degree_circle 353s (search-heavy under noise;
#     ceiling set well above local),
#   noise_circle_snap_vs_original 82s (likewise search-heavy),
#   ablation_mdl_stopping 8.7s for the full 30-seed run, 100%/100% recovery
#     (stop vs. fixed-degree oracle); 300s keeps generous headroom,
#   ablation_sample_size_circle 10s,
#   avi_baseline: full 30-seed/N=5000 headline config (10 systems x 3 sigma x
#     30 seeds) measures ~3060s (51 min) wall clock; most systems' AVI +
#     sr_gb() calls are well under 100ms each and only a handful of noisy
#     cells (Kepler, harmonic oscillator, I.12.2, kinematics_position) are
#     expensive.
#     7200s (2h) keeps ~2.3x headroom,
#   dt_sweep_modified_equation: Part 1 (symbolic, no data) plus Part 2a
#     (direct least-squares projection, no search, 30 seeds) plus one
#     illustrative Part 2 degeneracy-probe call together measure 25.66s total.
#     300s keeps ~11.7x headroom; this script does not sweep the full pipeline
#     across dt (see the module docstring for why that is descoped).
#   pysr_gb: single real N=5000/niterations=20 fit measures 100s;
#     2 experiments x 30 seeds x ~100s ~= 1.7h, so 3h keeps ~1.5-2x headroom.
#   feynman: one full seed across all 26 eqs x 3 sigma (78 cells) totals 242s
#     local (SR-GB 150s, SINDy-ST 70s, KRONIC 22s), so 30 seeds ~= 7260s
#     (~2.0h) local; on a ~2x-slower hosted CPU ~12000s (~3.3h). 21600s (6h)
#     keeps ~1.8x headroom. Per-cell timings: Results/feynman_timing_probe.csv.
TIMEOUTS = {
    "benchmark_redundancy_elimination.py": 900,
    "benchmark_harmonic_oscillator_vs_sindy.py": 1800,
    "benchmark_kepler_angular_momentum.py": 300,
    "benchmark_linear_holonomic.py": 300,
    "benchmark_nonlinear_holonomic.py": 300,
    "benchmark_loop_invariants.py": 2700,
    "benchmark_omp_nullspace.py": 1800,
    "benchmark_overlifting_circle.py": 300,
    "ablation_sindy_snap_gb_hybrid.py": 300,
    "benchmark_runtime_breakdown.py": 300,
    "benchmark_pysr_gb_implicit_vs_explicit.py": 10800,
    # deflation_multi_invariant: includes a SINDy-FD arm (one trajectory +
    # finite-difference call per seed) alongside SR-GB+CSNP and SINDy-null,
    # and now also a SINDy-AD arm (spline-based derivative proxy on the
    # same trajectory, each call wrapped in its own 30s timeout). FD-only
    # measurement at full scale (30 seeds): 45.3s. The AD arm has not yet
    # been re-measured at full scale here (only a 3-seed quick smoke test,
    # which ran effectively instantly with 0 timeouts); dt_discriminator's
    # own AD arm added no measurable overhead at full scale (see below), so
    # 300s (~6.6x the FD-only figure) is expected to stay generous, but
    # that has not been directly confirmed for this script yet.
    "benchmark_deflation_multi_invariant.py": 300,
    # vortex: physical multi-invariant recovery on pooled point-vortex
    # trajectories (n=3,4,5), degree-2 full-nullspace deflation. Per-seed cost
    # measured locally at n=3 ~10s, n=4 ~50s, n=5 ~11s (~71s/seed summed); the
    # full 30-seed run is ~2130s local, ~3500s on a ~1.65x-slower hosted CPU.
    # 5400s keeps ~1.5x headroom over the slower estimate.
    "benchmark_vortex.py": 5400,
    # gb_scalability: 7 configs x 3 instances, each instance SIGALRM-capped
    # at 120s inside the script (random interacting ideals are bimodal:
    # typically sub-second, occasionally doubly-exponential blowups that the
    # cap absorbs as recorded timeouts). Worst case ~ 21 x 120s = 2520s.
    "benchmark_gb_scalability.py": 3600,
    "benchmark_sos_sensitivity.py": 300,
    "benchmark_robotics.py": 1800,
    # Same harness as benchmark_robotics.py (measured ~4s locally); 1800s for
    # the same MuJoCo-environment headroom, also local-only where MuJoCo is
    # unavailable.
    "benchmark_gravity_pendulum.py": 1800,
    "benchmark_smt_verification.py": 300,
    "benchmark_feyman.py": 25200,
    "ablation_degree_circle.py": 3600,
    "ablation_noise_circle_snap_vs_original.py": 3600,
    "ablation_sample_size_circle.py": 300,
    # ar1_circle: 3-seed sample measured ~0.03s/call locally; full run
    # (9 cells x 30 seeds) ~30s local. 600s covers hosted-CPU slowdown plus a
    # search-heavy worse seed by a wide margin.
    "ablation_noise_ar1_circle.py": 600,
    "ablation_mdl_stopping.py": 300,
    # qmax_tolerance: fixed-degree-2 recovery/abstention calls at N=5000.
    # Arm A 6 systems x 6 Qmax + Arm B 2 systems x 6 Qmax + Arm C 4 eps,
    # all x 30 seeds ~= 1740 calls. Local degree-2 conic calls are sub-second;
    # 7200s covers hosted-CPU slowdown and search-heavy negatives by a wide margin.
    "ablation_qmax_tolerance.py": 7200,
    # noisy_rank_guard: Arm A is cheap (SVD + evaluate_support calls, no
    # search); Arm B is 2 subsystems x 2 sigma x 10 k x 30 seeds = 1200 full
    # sr_gb(full_nullspace=True) calls on an over-lifted d*~13 library.
    # Measured 163s locally; 3600s covers hosted-CPU slowdown by a wide margin.
    "ablation_noisy_rank_guard.py": 3600,
    # avi_eps_multiplier: 3 systems x 3 sigma x 10 k x default 10 seeds =
    # 900 avi_border_basis calls at N=1000, smaller/faster than
    # benchmark_avi_baseline.py's own N=5000 calls. 3600s is generous headroom.
    "ablation_avi_eps_multiplier.py": 3600,
    # sparsity_vs_rationality_harmonic: direct support enumeration (k=1..6)
    # over a 15-monomial dictionary, 11 dt values x 30 seeds = 330 trials at
    # sigma=0. Measured directly at full scale: 1308s (~21.8 min). 3600s is
    # ~2.75x, matching the other search-heavy ablations' wider margin rather
    # than this file's ~20% convention.
    "ablation_sparsity_vs_rationality_harmonic.py": 3600,
    "benchmark_avi_baseline.py": 14400,
    "benchmark_dt_sweep_modified_equation.py": 300,
    # difference_dictionary_generality: 2-seed sample measured ~0.03s/trial
    # locally (M=5 difference library); full run (16 cells x 30 seeds) ~15s
    # local. 600s covers hosted-CPU slowdown by a wide margin.
    "benchmark_difference_dictionary_generality.py": 600,
    # The five entries below use a tighter ~10-20% margin over the measured
    # or extrapolated full-scale cost, rather than the ~3x convention used
    # elsewhere in this file: large multiples waste allotted time on scripts
    # that already finished, while a ~10-20% pad is enough to absorb normal
    # run-to-run variance without either wasting a large block of unused
    # headroom or risking a near-complete run getting killed right before it
    # finishes.
    # bb_search_stats: measured directly at full scale (Kepler + harmonic
    # oscillator, 2 sigma levels, 10 seeds each, N=5000) at 46s. 55s is ~20%.
    "benchmark_bb_search_stats.py": 1200,
    # dt_discriminator: includes SINDy-FD and SINDy-AD arms (the latter a
    # spline-based derivative proxy, each call wrapped in its own 30s
    # timeout) on the two fixed-dt modes. Re-measured directly at full
    # scale (30 seeds x 3 modes, N=5000) with both arms included:
    # 23.7s + 46.7s + 17.7s = 88.1s (90.5s wall via `time`), 0 SINDy-AD
    # timeouts, so adding the AD arm did not meaningfully change the
    # FD-only figure this measured before (93.2s). 900s is ~10x, kept
    # generous rather than this block's usual ~20% margin since neither
    # arm has yet been exercised on a hosted CPU.
    "benchmark_dt_discriminator.py": 900,
    # unit_scale_sensitivity: 2-seed sample (4 benchmarks x 3 conditions)
    # measured 150s; linear extrapolation to 30 seeds ~2250s. 2700s is ~20%.
    "benchmark_unit_scale_sensitivity.py": 3600,
    # oracle_misclassification: full 30-seed run measured at ~7s locally, so
    # this cap only needs to cover the fast run with headroom for a slower
    # hosted CPU. 300s is deliberately generous (>40x the local time).
    "benchmark_oracle_misclassification.py": 300,
    # validate_sr_gb: smoke/integration suite with high-dimensional BB
    # fallback cases.  It has no --quick mode, and measured local runs can
    # exceed several minutes on a constrained CPU.  Thirty minutes provides
    # generous hosted-notebook headroom without masking a genuine hang.
    "validate_sr_gb.py": 1800,
    # noise_ceiling_probe: pure-SVD probe (2 systems x 5 sigma x 10 seeds),
    # no full pipeline; measured well under a minute locally. 300s is ample.
    "_noise_ceiling_probe.py": 300,
    # verify_bootstrap_reduction: 26 Feynman eqs x 2 sigma x 5 seeds, each
    # running bootstrapped SINDy-null and SINDy-ST-ensemble twice (reduced and
    # full settings). Cost is driven by the SINDy dictionary size C(n+d, d),
    # not degree alone. The two 56-monomial cases (5-var degree-3, e.g.
    # polarization_correction) run ~967s each while a 35-monomial degree-3
    # case (I.43.31) is only ~313s and the 14 degree-2 cases are ~20-40s
    # each. A size^2.4 fit
    # through the two measured degree-3 anchors puts the full 26-equation local
    # sweep at ~5500s; on a ~1.65x slower hosted CPU that is ~9000s (~2.5h).
    # 14400s (4h) keeps ~1.6x headroom over the slower estimate. The script
    # flushes its CSV after every equation, so even a timeout past this
    # preserves all completed equations. (This is the slowest probe in the
    # suite for a footnote-level claim; dropping it from 5 to 3 seeds would cut
    # it by ~40% if a shorter cap is wanted.)
    "_verify_bootstrap_reduction.py": 14400,
    # feynman_timing_probe: 30-seed per-cell timing diagnostic over all 26
    # equations x 3 sigma. It backs the paper's SINDy-ST / KRONIC baseline-cost
    # ratios (per-cell timing varies with seed, and the worst-cell figure is a
    # max, so a single seed is too noisy for a cited number) and also anchors
    # benchmark_feyman.py's timeout. The pipeline times ~204s of
    # method-call wall per seed (Results/feynman_timing_probe.csv), so ~6100s
    # local for 30 seeds and ~10100s on a slower hosted CPU.
    # 14400s (4h) covers the slow end plus search-heavy worse seeds under noise.
    "_feynman_timing_probe.py": 14400,
}
# Fallback for anything not listed above (currently just the test_*.py
# scripts, all of which run in a few seconds even at full scale) -- kept
# short so an unexpected hang on an untuned script doesn't silently burn an
# hour of compute.
DEFAULT_TIMEOUT = 600

# The blanket 300s quick-mode cap below is too tight for two different
# reasons:
#  - PySR's juliacall dependency downloads and precompiles its own managed
#    Julia install on first use, which alone can take well over 5 minutes
#    on a fresh machine -- unrelated to how few seeds/iters --quick requests.
#  - --quick only reduces seed count (1-3 seeds), not N/sigma grids or
#    equation subsets (see each script's own --quick help text), so a
#    script whose *full* run is dominated by many (equation, sigma) cells
#    rather than by seeds can still take several minutes in quick mode.
#    benchmark_feyman.py --quick (1 seed, but still all 26 equations x 3
#    sigma levels x full N=5000) measures ~1370s local via the 1-seed timing
#    probe at the full baseline ensembles (n_bootstrap=15, n_alphas=10),
#    ~1550-2260s on a slower hosted CPU, with SINDy-ST dominating. A
#    per-equation Branch-and-Bound blowup for an unlucky noise realization
#    (individual BB calls that normally take a few seconds occasionally run
#    20-120s+) can push a single cell far above the point estimate, so the
#    cap stays generous: 5000s is ~2.2x the slow estimate, and run_script()'s
#    TimeoutExpired handler preserves partial stdout/stderr, so a timeout
#    will show which combo it was on;
#    benchmark_avi_baseline.py --quick (2 seeds, full N=1000 and
#    all 3 sigma levels, including the Kepler/harmonic-oscillator systems
#    whose degree-2 dictionary has nullspace dimension 9) is a smaller but
#    similar case.
QUICK_TIMEOUT_OVERRIDES = {
    "benchmark_pysr_gb_implicit_vs_explicit.py": 1800,
    "benchmark_feyman.py": 5000,
    "benchmark_avi_baseline.py": 900,
    # validate_sr_gb.py ignores --quick, so do not silently cut its
    # integration-validation budget to the blanket 300-second quick cap.
    "validate_sr_gb.py": 1800,
}


def run_script(script_path, quick=False, timeout=None):
    cmd = [sys.executable, script_path]
    if quick:
        cmd.append("--quick")
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        runtime = time.time() - start_time
        return result.returncode == 0, result.stdout, result.stderr, runtime
    except subprocess.TimeoutExpired as e:
        runtime = time.time() - start_time
        # e.stdout/e.stderr hold whatever the subprocess had already flushed
        # before being killed -- capturing it is the only way to see which
        # combo (equation/sigma/seed) a script was stuck on when it timed
        # out, rather than losing that progress log entirely.
        partial_stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        partial_stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return False, partial_stdout, partial_stderr + f"\nTimeout after {timeout}s", runtime
    except Exception as e:
        runtime = time.time() - start_time
        return False, "", str(e), runtime


def run_category(category_name, scripts, quick=False, verbose=True):
    print("\n" + "=" * 80)
    print(f"RUNNING CATEGORY: {category_name.upper()}")
    print("=" * 80)
    results = {"category": category_name, "total": len(scripts), "passed": 0, "failed": 0, "skipped": 0, "details": [], "total_time": 0.0}

    for i, script in enumerate(scripts, 1):
        script_path = Path(script)
        if not script_path.exists():
            print(f"\n[{i}/{len(scripts)}] ⚠️  SKIPPED: {script} (file not found)")
            results["skipped"] += 1
            results["details"].append({"script": script, "status": "skipped", "reason": "file not found", "runtime": 0.0})
            continue

        timeout = TIMEOUTS.get(script, DEFAULT_TIMEOUT)
        if quick:
            timeout = min(timeout, QUICK_TIMEOUT_OVERRIDES.get(script, 300))

        print(f"\n[{i}/{len(scripts)}] Running: {script} (timeout: {timeout//60} min)...")
        sys.stdout.flush()

        success, stdout, stderr, runtime = run_script(script, quick=quick, timeout=timeout)
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  Status: {status} (runtime: {runtime:.2f}s)")

        if verbose and stdout:
            lines = stdout.strip().split('\n')
            if len(lines) > 10:
                print("  Output (first 10 lines):")
                for line in lines[:10]:
                    print(f"    {line}")
                print(f"    ... ({len(lines)-10} more lines)")
            else:
                for line in lines:
                    print(f"    {line}")

        if stderr and not success:
            # Tracebacks put the actual exception message at the end, not
            # the start, so show the tail -- the full text is also always
            # saved to Results/logs/ regardless of this preview length.
            err_lines = stderr.strip().split('\n')
            print(f"  Error output (last {min(15, len(err_lines))} lines):")
            for line in err_lines[-15:]:
                print(f"    {line}")

        os.makedirs("Results/logs", exist_ok=True)
        log_stem = Path(script).stem
        with open(f"Results/logs/{log_stem}.stdout.log", "w") as f:
            f.write(stdout)
        with open(f"Results/logs/{log_stem}.stderr.log", "w") as f:
            f.write(stderr)

        results["details"].append({"script": script, "status": "passed" if success else "failed", "runtime": runtime})
        if success:
            results["passed"] += 1
        else:
            results["failed"] += 1
        results["total_time"] += runtime
        time.sleep(1)

    return results


def generate_tables():
    print("\n" + "=" * 80)
    print("GENERATING LATEX TABLES")
    print("=" * 80)
    table_script = "generate_result_tables.py"
    if not Path(table_script).exists():
        print(f"⚠️  {table_script} not found. Skipping.")
        return False
    try:
        result = subprocess.run([sys.executable, table_script], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Tables generated successfully.")
            print(result.stdout)
            return True
        else:
            print("❌ Table generation failed.")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Table generation error: {e}")
        return False


def print_summary(all_results):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    total_passed, total_failed, total_skipped, total_time = 0, 0, 0, 0.0
    for category, results in all_results.items():
        print(f"\n📁 {category.upper()}")
        print(f"  Total:  {results['total']}")
        print(f"  Passed: {results['passed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Skipped: {results['skipped']}")
        print(f"  Time:   {results['total_time']:.2f}s")
        total_passed += results['passed']
        total_failed += results['failed']
        total_skipped += results['skipped']
        total_time += results['total_time']

    failed_scripts = []
    for category, results in all_results.items():
        for detail in results["details"]:
            if detail["status"] == "failed":
                failed_scripts.append(f"  [{category}] {detail['script']}")
    if failed_scripts:
        print("\n❌ FAILED SCRIPTS:")
        for script in failed_scripts:
            print(script)

    total_scripts = total_passed + total_failed + total_skipped
    print("\n" + "-" * 80)
    print(f"OVERALL: {total_passed}/{total_scripts} passed ({(total_passed/total_scripts*100):.1f}%)")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("=" * 80)

    os.makedirs("Results", exist_ok=True)
    with open("Results/experiment_summary.txt", "w") as f:
        f.write(f"Experiment Run Summary\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n\n")
        for category, results in all_results.items():
            f.write(f"\n📁 {category.upper()}\n")
            f.write(f"  Total:  {results['total']}\n")
            f.write(f"  Passed: {results['passed']}\n")
            f.write(f"  Failed: {results['failed']}\n")
            f.write(f"  Skipped: {results['skipped']}\n")
            f.write(f"  Time:   {results['total_time']:.2f}s\n")
        f.write(f"\nOVERALL: {total_passed}/{total_scripts} passed\nTotal time: {total_time:.2f}s\n")
    print("\n📊 Summary saved to: Results/experiment_summary.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="all",
                        choices=["all", "benchmarks", "ablations", "tests", "validation_and_verification"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the validation_and_verification category in an all-categories run.")
    parser.add_argument("--generate-tables", action="store_true")
    parser.add_argument("--skip", action="append", default=[], metavar="SCRIPT",
                        help="Script filename to skip (repeatable), e.g. "
                             "--skip benchmark_feyman.py, for excluding a "
                             "script that already ran or is being debugged "
                             "separately.")
    args = parser.parse_args()

    if args.quick and args.generate_tables:
        # --quick writes to the exact same Results/*.csv paths as a full run
        # (only ~1 of ~34 registered scripts distinguishes its output path by
        # tag), so generating tables immediately afterwards would silently
        # paste quick-mode (1-3 seed) numbers into the LaTeX tables in place
        # of the full-scale committed results. Both flags are legitimate on
        # their own; only the combination is unsafe.
        print("ERROR: --quick and --generate-tables cannot be combined -- "
              "quick-mode results would overwrite the full-scale Results/*.csv "
              "that --generate-tables reads from. Run the full suite, or run "
              "--generate-tables separately afterwards.")
        sys.exit(1)

    known_scripts = {s for script_list in SCRIPTS.values() for s in script_list}
    unknown_skips = [s for s in args.skip if s not in known_scripts]
    if unknown_skips:
        print(f"WARNING: --skip names not in SCRIPTS (typo?): {unknown_skips}")

    scripts_to_run = {}
    if args.category == "all":
        categories = ["benchmarks", "ablations"]
        if not args.skip_validation:
            categories.append("validation_and_verification")
        if not args.skip_tests:
            categories.append("tests")
    else:
        categories = [args.category]

    skip_set = set(args.skip)
    for cat in categories:
        if cat in SCRIPTS:
            scripts_to_run[cat] = [s for s in SCRIPTS[cat] if s not in skip_set]

    if not scripts_to_run:
        print("No scripts to run.")
        sys.exit(1)

    print("=" * 80)
    print("SR-GB EXPERIMENT ORCHESTRATOR")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick mode: {args.quick}")
    print(f"Verbose: {args.verbose}")
    print(f"Categories: {', '.join(scripts_to_run.keys())}")
    if args.skip:
        print(f"Skipped scripts: {', '.join(args.skip)}")
    print("=" * 80)

    os.makedirs("Results", exist_ok=True)

    all_results = {}
    for category, scripts in scripts_to_run.items():
        all_results[category] = run_category(category, scripts, quick=args.quick, verbose=args.verbose)

    print_summary(all_results)

    if args.generate_tables:
        generate_tables()

    total_failed = sum(r["failed"] for r in all_results.values())
    total_skipped = sum(r["skipped"] for r in all_results.values())
    sys.exit(0 if total_failed == 0 and total_skipped == 0 else 1)


if __name__ == "__main__":
    main()
