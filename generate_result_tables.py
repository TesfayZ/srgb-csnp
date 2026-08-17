#!/usr/bin/env python3
"""
generate_result_tables.py – Generate LaTeX tables from CSV files.
includes:
  - Table: Adaptive degree discovery (from ablation_adaptive_degree_circle.csv)
  - Table: Fixed‑degree recovery rates (from ablation_fixed_degree_summary.csv)
  - Wilson confidence intervals added to all tables that lacked them
"""

import os
import sys
import pandas as pd
import numpy as np
from utils_stats import wilson_interval, mcnemar_exact

# ----------------------------------------------------------------------
# Helper function for formatting rates with CI
# ----------------------------------------------------------------------

def _fmt_rate_with_ci(rate, n, decimals=0):
    """Format a rate with its Wilson CI, given number of trials n."""
    if n <= 0:
        return "---"
    k = int(round(rate * n))
    lo, hi = wilson_interval(k, n)
    if decimals == 0:
        return f"{rate*100:.0f}\\% [{lo*100:.0f}\\%, {hi*100:.0f}\\%]"
    else:
        return f"{rate*100:.{decimals}f}\\% [{lo*100:.{decimals}f}\\%, {hi*100:.{decimals}f}\\%]"

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def load_csv(filename):
    path = os.path.join("Results", filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found.", file=sys.stderr)
        return None
    df = pd.read_csv(path)
    print(f"Loaded {filename}: columns = {list(df.columns)}")
    return df

def format_ci(rate, ci_low, ci_high, decimals=1):
    return f"{rate*100:.{decimals}f}\\% [{ci_low*100:.0f}\\% , {ci_high*100:.0f}\\%]"

def format_ci_compact(rate, ci_low, ci_high, decimals=1):
    """Compact CI format for side-by-side minipage panels: '73.1\\%
    [70.0,76.0]', percent sign once, no space after the comma. Used where
    two tables are merged into (a)/(b) panels and column width is tight."""
    def _fmt(x):
        pct = x * 100
        return f"{pct:.0f}" if float(pct).is_integer() else f"{pct:.{decimals}f}"
    return f"{_fmt(rate)}\\% [{_fmt(ci_low)},{_fmt(ci_high)}]"

def format_mean_std(mean, std, decimals=1):
    if pd.isna(mean) or pd.isna(std):
        return "---"
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"

def safe_mean(df, col):
    if col in df.columns:
        return df[col].mean()
    return np.nan

def safe_get(df, problem, col):
    if col not in df.columns:
        return np.nan
    try:
        val = df[df["problem"] == problem][col].values[0]
        return val
    except IndexError:
        return np.nan

def write_table_header(out, caption, label, columns, fontsize=None):
    out.write("\\begin{table}[ht]\n")
    out.write("\\centering\n")
    out.write(f"\\caption{{{caption}}}\n")
    out.write(f"\\label{{{label}}}\n")
    if fontsize:
        out.write(f"\\{fontsize}\n")
    out.write("\\begin{tabular}{" + columns + "}\n")
    out.write("\\toprule\n")

def write_table_footer(out):
    out.write("\\bottomrule\n")
    out.write("\\end{tabular}\n")
    out.write("\\end{table}\n\n")

def write_rows_collapsing_ranges(out, sigmas, cell_lists, fmt="{:.2f}"):
    """Write one row per run of consecutive sigmas sharing identical cell
    content, printing a 'lo--hi' range label when a run spans more than
    one sigma. Matches the paper's convention of merging rows that repeat
    the same result across a parameter sweep instead of listing each one,
    e.g. sigma=0.04..0.14 all reading 0%/100% collapses to one row."""
    i = 0
    n = len(sigmas)
    while i < n:
        j = i
        while j + 1 < n and cell_lists[j + 1] == cell_lists[i]:
            j += 1
        label = fmt.format(sigmas[i])
        if j > i:
            label += "--" + fmt.format(sigmas[j])
        out.write(f"{label} & " + " & ".join(cell_lists[i]) + " \\\\\n")
        i = j + 1

# ----------------------------------------------------------------------
# Redundancy elimination (tab:synthetic)
# ----------------------------------------------------------------------

def table_redundancy(out):
    df = load_csv("redundancy_elimination_summary.csv")
    if df is None:
        print("WARNING: redundancy_elimination_summary.csv missing – redundancy table skipped.")
        return

    methods = [
        ("SINDy-null (KRONIC)", "null"),
        ("SINDy-ST", "st"),
        ("SINDy-AD (spline-denoised)", "ad"),   # note: column names still 'ad' from earlier runs
        ("Dense SVD+GB", "dense"),
        ("SR-GB+CSNP (ours)", "srgb")
    ]

    overall_rec = {}
    overall_fdr = {}
    for label, prefix in methods:
        overall_rec[label] = safe_mean(df, f"{prefix}_rec_mean")
        overall_fdr[label] = safe_mean(df, f"{prefix}_fdr_mean")

    if all(np.isnan(v) for v in overall_rec.values()):
        print("ERROR: No recognition columns found. Please regenerate CSV with updated benchmark.")
        return

    write_table_header(out,
        "Redundancy elimination results over 30 seeds ($\\sigma=0.0$). "
        "SINDy-AD requires time-series data for its spline-based denoising and is not applicable to these static varieties.",
        "tab:synthetic",
        "lccccr")
    out.write("\\textbf{Method} & \\textbf{Red. Circle} & \\textbf{Red. Sphere} & \\textbf{Red. Cubic} & \\textbf{Rec. error} & \\textbf{FDR} \\\\\n")
    out.write("\\midrule\n")

    for label, prefix in methods:
        circle = format_mean_std(safe_get(df, "Circle", f"{prefix}_red_mean"),
                                 safe_get(df, "Circle", f"{prefix}_red_std"))
        sphere = format_mean_std(safe_get(df, "Sphere", f"{prefix}_red_mean"),
                                 safe_get(df, "Sphere", f"{prefix}_red_std"))
        cubic = format_mean_std(safe_get(df, "Cubic", f"{prefix}_red_mean"),
                                safe_get(df, "Cubic", f"{prefix}_red_std"))
        rec = f"{overall_rec[label]:.2f}" if not np.isnan(overall_rec[label]) else "---"
        fdr = f"{overall_fdr[label]:.2f}" if not np.isnan(overall_fdr[label]) else "---"
        out.write(f"{label} & {circle} & {sphere} & {cubic} & {rec} & {fdr} \\\\\n")

    write_table_footer(out)

# ----------------------------------------------------------------------
# Over-lifting robustness on SINDy-null and Dense SVD+GB (tab:overlifting)
# ----------------------------------------------------------------------

def table_overlifting(out):
    df = load_csv("overlifting_circle_summary.csv")
    if df is None:
        print("WARNING: overlifting_circle_summary.csv missing – Table 4 skipped.")
        return
    row = df.iloc[0]

    # Rate and CI are separate table cells: the header declares three
    # columns, so a single combined format_ci cell would leave every data row
    # one column short.
    def _cells(rate, lo, hi, bold=False):
        r = f"{rate*100:.1f}\\%"
        if bold:
            r = f"\\textbf{{{r}}}"
        return f"{r} & [{lo*100:.0f}\\%, {hi*100:.0f}\\%]"

    write_table_header(out,
        "Exact recovery on the unit circle with over‑lifting ($D=3$). "
        "SINDy‑null returns a higher‑degree multiple in most trials; Dense SVD+GB reduces it via algebraic minimality.",
        "tab:overlifting",
        "lcc")
    out.write("\\textbf{Method} & \\textbf{Exact recovery} & \\textbf{95\\% Wilson CI} \\\\\n")
    out.write("\\midrule\n")
    out.write(f"SINDy‑null (KRONIC) & {_cells(row['sindy_rate'], row['sindy_ci_low'], row['sindy_ci_high'])} \\\\\n")
    out.write(f"Dense SVD+GB & {_cells(row['dense_rate'], row['dense_ci_low'], row['dense_ci_high'], bold=True)} \\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Transition invariants (tab:transition)
# ----------------------------------------------------------------------

def table_transition(out):
    df_ho = load_csv("harmonic_oscillator_sindy_comparison.csv")
    ho_rates = {}
    if df_ho is not None:
        col_map = {
            "srgb_exact": "SR-GB+CSNP (ours)",
            "sindy_null_exact": "SINDy-null (KRONIC)",
            "sindy_st_exact": "SINDy-ST",
            # Dual FD scoring: state-energy target and the induced
            # transition-invariant target reported separately.
            "sindy_fd_state_exact": "SINDy-FD (finite diff., state target)",
            "sindy_fd_transition_exact": "SINDy-FD (finite diff., transition target)",
            # Backward compatibility with the older single-column CSV.
            "sindy_fd_exact": "SINDy-FD (finite diff.)"
        }
        for col, label in col_map.items():
            if col not in df_ho.columns:
                continue
            k = df_ho[col].sum()
            n = len(df_ho)
            rate = k / n
            ci = wilson_interval(k, n)
            ho_rates[label] = (rate, ci[0], ci[1])

    df_kep = load_csv("kepler_angular_momentum_results.csv")
    if df_kep is not None and "exact" in df_kep.columns:
        k = df_kep["exact"].sum()
        n = len(df_kep)
        kep_rate = k / n
        kep_ci = wilson_interval(k, n)
    else:
        kep_rate = kep_ci = None

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Transition invariant recovery rates ($\\sigma=0.0$). "
               "SINDy-FD uses finite differences; it is not the published "
               "SINDy-AD method.}\n")
    out.write("\\label{tab:transition}\n")
    out.write("\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n")
    out.write("\\begin{tabular}{@{}p{5.0cm}p{4.9cm}cc@{}}\n\\toprule\n")
    out.write("\\textbf{System} & \\textbf{Mode} & \\textbf{Exact recovery} & \\textbf{Redundancy} \\\\\n")
    out.write("\\midrule\n")

    if "SR-GB+CSNP (ours)" in ho_rates:
        r, lo, hi = ho_rates["SR-GB+CSNP (ours)"]
        out.write(f"Harmonic oscillator, fixed $dt$ & BB & \\textbf{{{format_ci(r, lo, hi)}}} & 1.0 \\\\\n")
    else:
        print("WARNING: harmonic_oscillator_sindy_comparison.csv missing or lacks 'srgb_exact'; skipping row")

    for label in ["SINDy-null (KRONIC)", "SINDy-ST",
                  "SINDy-FD (finite diff., state target)",
                  "SINDy-FD (finite diff., transition target)",
                  "SINDy-FD (finite diff.)"]:
        if label in ho_rates:
            r, lo, hi = ho_rates[label]
            out.write(f"Harmonic oscillator, fixed $dt$ & {label} & {format_ci(r, lo, hi)} & --- \\\\\n")

    if kep_rate is not None and not np.isnan(kep_rate):
        out.write(f"Kepler angular momentum & BB & \\textbf{{{format_ci(kep_rate, kep_ci[0], kep_ci[1])}}} & 1.0 \\\\\n")
    else:
        print("WARNING: kepler_angular_momentum_results.csv missing or lacks 'exact'; skipping row")

    write_table_footer(out)

# ----------------------------------------------------------------------
# Noise sensitivity (tab:noise)
# ----------------------------------------------------------------------

def table_noise(out):
    """Merged panel table: (a) circle noise sweep, na\\"ive vs snap-rounding;
    (b) AR(1)-correlated-noise circle ablation ($\\phi=0$ reproduces the iid
    control). Matches the paper's single merged table under \\label{tab:noise},
    which absorbs the former standalone tab:ar1-noise as panel (b)."""
    df = load_csv("noise_ablation_circle_rates.csv")
    if df is None:
        print("WARNING: noise_ablation_circle_rates.csv missing – Table (tab:noise) skipped.")
        return

    required = ["sigma", "new_rate", "new_ci_low", "new_ci_high", "orig_rate", "orig_ci_low", "orig_ci_high"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing. Please regenerate ablation script.")
            return

    df = df[df["sigma"] <= 0.30]

    df_ar1 = load_csv("ablation_noise_ar1_circle_rates.csv")
    ar1_required = ["phi", "sigma", "exact_rate", "ci_low", "ci_high"]
    ar1_ok = df_ar1 is not None and all(c in df_ar1.columns for c in ar1_required)
    if df_ar1 is not None and not ar1_ok:
        print("ERROR: ablation_noise_ar1_circle_rates.csv missing required columns – "
              "panel (b) skipped.")

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Circle noise sensitivity, $N=5000$, 30 seeds, 95\\% Wilson CI. "
               "\\textbf{(a)} Exact recovery vs.\\ noise level $\\sigma$ under i.i.d.\\ "
               "noise, na\\\"{i}ve rounding vs.\\ snap-rounding. \\textbf{(b)} Exact "
               "recovery under AR(1)-correlated noise at three correlation strengths "
               "$\\phi$, isolating whether the SVD's noise-averaging is disrupted by "
               "correlation rather than just magnitude.}\n")
    out.write("\\label{tab:noise}\n")

    # Panel (a)
    out.write("\\begin{minipage}[t]{0.56\\textwidth}\n\\centering\n\\footnotesize\n"
               "\\textbf{(a)}\\\\[2pt]\n")
    out.write("\\begin{tabular}{ccc}\n\\toprule\n")
    out.write("$\\sigma$ & Na\\\"{i}ve & Snap-round.\\\\\n\\midrule\n")
    sigmas, cells = [], []
    for _, row in df.iterrows():
        sigmas.append(row["sigma"])
        naive = format_ci(row["orig_rate"], row["orig_ci_low"], row["orig_ci_high"])
        snap = format_ci(row["new_rate"], row["new_ci_low"], row["new_ci_high"])
        cells.append((naive, f"\\textbf{{{snap}}}"))
    write_rows_collapsing_ranges(out, sigmas, cells)
    out.write("\\bottomrule\n\\end{tabular}\n\\end{minipage}%\n\\hfill\n")

    # Panel (b)
    out.write("\\begin{minipage}[t]{0.42\\textwidth}\n\\centering\n\\scriptsize\n"
               "\\textbf{(b)}\\\\[2pt]\n")
    if ar1_ok:
        phis = sorted(df_ar1["phi"].unique())
        sigmas = sorted(df_ar1["sigma"].unique())
        out.write("\\begin{tabular}{" + "c" * (len(phis) + 1) + "}\n\\toprule\n")
        header = "$\\sigma$ & " + " & ".join(f"$\\phi{{=}}{p:g}$" for p in phis)
        out.write(header + "\\\\\n\\midrule\n")
        for sigma in sigmas:
            cells = []
            for p in phis:
                row = df_ar1[(df_ar1["phi"] == p) & (df_ar1["sigma"] == sigma)]
                if len(row) != 1:
                    cells.append("---")
                    continue
                cells.append(f"{row.iloc[0]['exact_rate']*100:.3g}\\%")
            out.write(f"{sigma:.2f} & " + " & ".join(cells) + " \\\\\n")
        out.write("\\bottomrule\n\\end{tabular}\n")
        out.write("\\\\[3pt]\n{\\tiny 95\\% Wilson CI for each rate here matches panel "
                   "(a) at the same point estimate (n=30 in both).}\n")
    else:
        out.write("\\emph{AR(1) ablation data unavailable.}\n")
    out.write("\\end{minipage}\n\\end{table}\n\n")

# ----------------------------------------------------------------------
# Feynman overall (tab:feynman)
# ----------------------------------------------------------------------

def table_feynman_overall(out):
    """Merged panel table: (a) Feynman exact-recovery rates by degree
    subset; (b) SR-GB+CSNP's All-26 row from (a) against KRONIC
    (SINDy-null) and the all-targets SINDy-ST ensemble, same sigma grid.
    Matches the paper's single merged table under \\label{tab:feynman} /
    \\label{tab:feynman-baselines} (the latter absorbed as panel (b))."""
    df_overall = load_csv("feynman_results_overall.csv")
    if df_overall is None:
        print("WARNING: feynman_results_overall.csv missing – Feynman table skipped.")
        return

    required = ["sigma", "srgb_rate", "srgb_ci_low", "srgb_ci_high", "n_eqs"]
    for col in required:
        if col not in df_overall.columns:
            print(f"ERROR: column '{col}' missing in feynman_results_overall.csv")
            return

    df_full = load_csv("feynman_results_full.csv")
    # Compute degree‑specific summaries from df_full if available
    deg2_summary = None
    deg3_summary = None
    if df_full is not None:
        deg_stats = df_full.groupby(["sigma", "degree"]).agg(
            rate=("srgb_exact", "mean"),
            k=("srgb_exact", "sum"),
            n=("srgb_exact", "count")
        ).reset_index()
        deg2_summary = deg_stats[deg_stats["degree"] == 2][["sigma", "rate", "k", "n"]]
        deg3_summary = deg_stats[deg_stats["degree"] == 3][["sigma", "rate", "k", "n"]]

    n_all = df_overall["n_eqs"].iloc[0] if not df_overall.empty else 26
    # n = number of EQUATIONS per degree class, not trials: dividing by
    # trials-per-sigma instead would print 140/120 in the paper's n column.
    n_deg2 = df_full[df_full["degree"] == 2]["name"].nunique() if df_full is not None else 14
    n_deg3 = df_full[df_full["degree"] == 3]["name"].nunique() if df_full is not None else 12

    baseline_cols = ["null_rate", "null_ci_low", "null_ci_high",
                      "sindy_st_rate", "sindy_st_ci_low", "sindy_st_ci_high"]
    have_baselines = all(c in df_overall.columns for c in baseline_cols)
    if not have_baselines:
        print("WARNING: KRONIC/SINDy-ST columns missing in feynman_results_overall.csv – "
              "panel (b) skipped.")

    # Paired McNemar exact test, srgb vs each baseline, per sigma. Uses
    # feynman_results_full.csv's per-trial columns directly (same rows the
    # rates above are aggregated from), not a new experiment. sigma=0 is
    # skipped: all three methods already agree at 100%, so it's degenerate
    # (b=c=0, p=1) and uninformative to print.
    if df_full is not None and all(c in df_full.columns for c in
                                     ["srgb_exact", "null_exact", "sindy_st_exact"]):
        print("\nMcNemar exact test (SR-GB+CSNP vs baseline, paired per-trial outcomes):")
        for sigma in [0.02, 0.05]:
            sub = df_full[df_full["sigma"] == sigma]
            if sub.empty:
                continue
            b_n, c_n, p_n = mcnemar_exact(sub["srgb_exact"], sub["null_exact"])
            b_s, c_s, p_s = mcnemar_exact(sub["srgb_exact"], sub["sindy_st_exact"])
            print(f"  sigma={sigma}: vs KRONIC   b={b_n} c={c_n} p={p_n:.3g}")
            print(f"  sigma={sigma}: vs SINDy-ST b={b_s} c={c_s} p={p_s:.3g}")

    out.write("\\begin{table}[ht]\n\\centering\n")
    out.write("\\caption{Feynman benchmark, all 26 equations, 30 seeds each. "
               "\\textbf{(a)} Exact recovery rates by degree subset. \\textbf{(b)} "
               "\\SRGBCSNP's All-26 row from (a) against \\KRONIC (\\SINDy-null) and "
               "the all-targets \\SINDy-ST ensemble, same $\\sigma$ grid.}\n")
    out.write("\\label{tab:feynman}\n\\label{tab:feynman-baselines}\n")

    # Panel (a)
    out.write("\\begin{minipage}[t]{0.58\\textwidth}\n\\centering\n\\footnotesize\n"
               "\\setlength{\\tabcolsep}{3pt}\n\\textbf{(a)}\\\\[2pt]\n")
    out.write("\\begin{tabular}{lcccc}\n\\toprule\n")
    out.write("\\textbf{Eq.\\ type} & $\\sigma{=}0.00$ & $\\sigma{=}0.02$ & "
               "$\\sigma{=}0.05$ & \\# \\\\\n\\midrule\n")

    def panel_a_cells(summary_or_overall, is_overall):
        cells = []
        for sigma in [0.00, 0.02, 0.05]:
            if is_overall:
                row = summary_or_overall[summary_or_overall["sigma"] == sigma]
                if row.empty:
                    cells.append("---")
                    continue
                r, lo, hi = (row["srgb_rate"].iloc[0], row["srgb_ci_low"].iloc[0],
                             row["srgb_ci_high"].iloc[0])
            else:
                if summary_or_overall is None:
                    cells.append("---")
                    continue
                row = summary_or_overall[summary_or_overall["sigma"] == sigma]
                if row.empty:
                    cells.append("---")
                    continue
                k, n = row["k"].iloc[0], row["n"].iloc[0]
                r = row["rate"].iloc[0]
                lo, hi = wilson_interval(k, n)
            cells.append(format_ci_compact(r, lo, hi))
        return cells

    all_str = panel_a_cells(df_overall, True)
    out.write(f"All 26 & \\textbf{{{all_str[0]}}} & {all_str[1]} & {all_str[2]} & {n_all} \\\\\n")
    deg2_str = panel_a_cells(deg2_summary, False)
    out.write(f"Degree-2 & \\textbf{{{deg2_str[0]}}} & {deg2_str[1]} & {deg2_str[2]} & {n_deg2} \\\\\n")
    deg3_str = panel_a_cells(deg3_summary, False)
    out.write(f"Degree-3 & \\textbf{{{deg3_str[0]}}} & {deg3_str[1]} & {deg3_str[2]} & {n_deg3} \\\\\n")
    out.write("\\bottomrule\n\\end{tabular}\n\\end{minipage}%\n\\hfill\n")

    # Panel (b)
    out.write("\\begin{minipage}[t]{0.38\\textwidth}\n\\centering\n\\footnotesize\n"
               "\\setlength{\\tabcolsep}{3pt}\n\\textbf{(b)}\\\\[2pt]\n")
    if have_baselines:
        out.write("\\begin{tabular}{lcc}\n\\toprule\n")
        out.write("$\\sigma$ & \\KRONIC & \\SINDy-ST\\\\\n\\midrule\n")
        srgb_vals = []
        for sigma in [0.00, 0.02, 0.05]:
            row = df_overall[df_overall["sigma"] == sigma]
            if row.empty:
                out.write(f"{sigma:.2f} & --- & --- \\\\\n")
                continue
            r = row.iloc[0]
            null = format_ci_compact(r["null_rate"], r["null_ci_low"], r["null_ci_high"])
            st = format_ci_compact(r["sindy_st_rate"], r["sindy_st_ci_low"], r["sindy_st_ci_high"])
            out.write(f"{sigma:.2f} & {null} & {st} \\\\\n")
            srgb_vals.append(f"{r['srgb_rate']*100:.3g}\\%")
        out.write("\\bottomrule\n\\end{tabular}\n")
        out.write("\\\\[3pt]\n{\\tiny \\SRGBCSNP at these $\\sigma$: "
                   + ", ".join(srgb_vals) + " (panel (a), All 26).}\n")
    else:
        out.write("\\emph{Baseline data unavailable.}\n")
    out.write("\\end{minipage}\n\\end{table}\n\n")

# ----------------------------------------------------------------------
# Groebner-basis scalability timings (tab:gb-scalability)
# ----------------------------------------------------------------------

def table_gb_scalability(out):
    """Emit tab:gb-scalability from gb_scalability.csv.

    Generating the table from the CSV keeps the median timings in sync with the
    data and the table reproducible. Times at or above 0.1 s are printed to 2
    decimals and smaller ones to 1 significant figure, matching the paper's
    rounding."""
    df = load_csv("gb_scalability.csv")
    if df is None:
        print("WARNING: gb_scalability.csv missing – GB scalability table skipped.")
        return

    required = ["n", "D", "M", "generators", "n_timeout", "instance_cap_s",
                "time_median_s"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in gb_scalability.csv "
                  "– GB scalability table skipped.")
            return

    cap = int(df["instance_cap_s"].iloc[0])
    n_inst = int(df["n_instances"].iloc[0]) if "n_instances" in df.columns else 3

    write_table_header(out,
        "Gr\\\"{o}bner basis computation timings (SymPy, grevlex over "
        "$\\mathbb{Q}$) for random sparse polynomial ideals with the stated "
        f"number of generators. Median over {n_inst} seeded instances per row, "
        f"with a {cap}\\,s per-instance cap; the timeout column counts "
        "instances hitting the cap (excluded from the median). Produced from "
        "\\texttt{Results/gb\\_scalability.csv}.",
        "tab:gb-scalability",
        "cccccc")
    out.write("$n$ & $D$ & $M$ & Generators & Median time & "
              f"Timeouts\\,/\\,{n_inst}\\\\\n")
    out.write("\\midrule\n")

    for _, r in df.iterrows():
        t = float(r["time_median_s"])
        tstr = f"{t:.2f}" if t >= 0.1 else f"{t:.1g}"
        out.write(f"{int(r['n'])} & {int(r['D'])} & {int(r['M'])} & "
                  f"{int(r['generators'])} & ${tstr}$\\,s & "
                  f"{int(r['n_timeout'])}\\\\\n")

    write_table_footer(out)

# ----------------------------------------------------------------------
# Holonomic constraints (tab:holonomic)
# ----------------------------------------------------------------------

def table_holonomic(out):
    df_lin = load_csv("linear_holonomic_equality_results_summary.csv")
    df_nonlin = load_csv("nonlinear_holonomic_results_summary.csv")
    if df_lin is None or df_nonlin is None:
        print("WARNING: holonomic summary CSVs missing – Table 7 skipped.")
        return
    required = ["sigma", "srgb_rate", "srgb_ci_low", "srgb_ci_high"]
    for df, name in [(df_lin, "linear"), (df_nonlin, "nonlinear")]:
        for col in required:
            if col not in df.columns:
                print(f"ERROR: column '{col}' missing in {name} summary CSV.")
                return

    merged = pd.merge(df_lin[["sigma", "srgb_rate", "srgb_ci_low", "srgb_ci_high"]],
                      df_nonlin[["sigma", "srgb_rate", "srgb_ci_low", "srgb_ci_high"]],
                      on="sigma", suffixes=("_lin", "_nonlin"))

    write_table_header(out,
        "Holonomic equality constraint recovery (30 seeds, 95\\% Wilson CI).",
        "tab:holonomic",
        "ccc")
    out.write("$\\sigma$ & \\textbf{Linear ($q_1-q_2=0$)} & \\textbf{Nonlinear (distance equality)} \\\\\n")
    out.write("\\midrule\n")

    sigmas, cells = [], []
    for _, row in merged.iterrows():
        sigmas.append(row["sigma"])
        lin = format_ci(row["srgb_rate_lin"], row["srgb_ci_low_lin"], row["srgb_ci_high_lin"])
        nonlin = format_ci(row["srgb_rate_nonlin"], row["srgb_ci_low_nonlin"], row["srgb_ci_high_nonlin"])
        cells.append((lin, nonlin))
    write_rows_collapsing_ranges(out, sigmas, cells)

    write_table_footer(out)

# ----------------------------------------------------------------------
# OMP vs CSNP (tab:omp)
# ----------------------------------------------------------------------

def table_omp(out):
    df = load_csv("omp_nullspace_summary.csv")
    if df is None:
        print("WARNING: omp_nullspace_summary.csv missing – Table 8 skipped.")
        return

    required = ["problem", "csnp_rate", "csnp_ci_low", "csnp_ci_high", "omp_rate", "omp_ci_low", "omp_ci_high"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in omp_nullspace_summary.csv")
            return

    write_table_header(out,
        "OMP vs CSNP exact recovery (30 seeds, 95\\% Wilson CI).",
        "tab:omp",
        "lcc")
    out.write("\\textbf{Problem} & \\textbf{CSNP (ours)} & \\textbf{OMP baseline} \\\\\n")
    out.write("\\midrule\n")

    for _, row in df.iterrows():
        problem = row["problem"].replace("_", "\\_")
        csnp = format_ci(row["csnp_rate"], row["csnp_ci_low"], row["csnp_ci_high"])
        omp = format_ci(row["omp_rate"], row["omp_ci_low"], row["omp_ci_high"])
        out.write(f"{problem} & \\textbf{{{csnp}}} & {omp} \\\\\n")

    write_table_footer(out)

# ----------------------------------------------------------------------
# Deflation multi-invariant recovery (tab:deflation)
# ----------------------------------------------------------------------

def table_deflation(out):
    df = load_csv("deflation_multi_invariant_results_fixed.csv")
    if df is None:
        print("WARNING: deflation_multi_invariant_results_fixed.csv missing – Table 9 skipped.")
        return

    required = ["energy_recovered", "ang_mom_recovered", "full_set_recovered"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in deflation CSV.")
            return

    if not any(c.startswith("sindy_fd_") for c in df.columns):
        print("WARNING: deflation CSV has no sindy_fd_* columns -- SINDy-FD "
              "cells will all render '---', which is indistinguishable from "
              "the caption's genuine structural-unreachability case. Rerun "
              "benchmark_deflation_multi_invariant.py to populate them.")
    if not any(c.startswith("sindy_ad_") for c in df.columns):
        print("WARNING: deflation CSV has no sindy_ad_* columns -- SINDy-AD "
              "cells will all render '---', which is indistinguishable from "
              "the caption's genuine structural-unreachability case. Rerun "
              "benchmark_deflation_multi_invariant.py to populate them.")
    if not any(c.startswith("rref_") for c in df.columns):
        print("WARNING: deflation CSV has no rref_* columns -- RREF cells "
              "will all render '---'. Rerun "
              "benchmark_deflation_multi_invariant.py to populate them.")

    n = len(df)

    def rate_ci(col):
        if col not in df.columns:
            return None
        k = int(df[col].sum())
        rate = k / n
        lo, hi = wilson_interval(k, n)
        return rate, lo, hi

    def cell(stats, bold=False):
        if stats is None:
            return "---"
        rate, lo, hi = stats
        s = f"{rate:.0%} [{lo:.0%}, {hi:.0%}]".replace("%", "\\%")
        return f"\\textbf{{{s}}}" if bold else s

    write_table_header(out,
        "Multi-invariant recovery via nullspace deflation (2D harmonic "
        "oscillator, 4 simultaneous generators, 30 seeds). \\SINDy-null, "
        "RREF, and \\SRGBCSNP search the identical difference dictionary; "
        "\\SINDy-FD and \\SINDy-AD search a smaller, restricted state-only "
        "dictionary and each return a single candidate per seed, so the "
        "full 4-generator set is structurally unreachable for either "
        "(marked ---). \\SINDy-AD differs from \\SINDy-FD only in its "
        "derivative estimator (a smoothing spline in place of a finite "
        "difference); it is a lightweight proxy for Kaheman et al.'s joint "
        "denoising/noise-model method, not a reimplementation of it, which "
        "is a fair comparison at this table's $\\sigma=0$ since there is "
        "no noise for the joint noise model to characterize; see the "
        "module docstring of "
        "\\texttt{benchmark\\_deflation\\_multi\\_invariant.py}. "
        "RREF~\\cite{OellerichEmelianenko2024} rows are read directly off "
        "the row-echelon-reduced nullspace basis, with no further "
        "Gr\\\"{o}bner reduction, matching what that method actually "
        "delivers as output.",
        "tab:deflation",
        "lccccc",
        fontsize="footnotesize")
    out.write("\\textbf{Invariant} & \\SRGBCSNP & \\SINDy-null & RREF & \\SINDy-FD & \\SINDy-AD \\\\\n")
    out.write("\\midrule\n")
    rows = [
        ("Energy (per-axis)", "energy_recovered", "sindy_null_energy_recovered", "rref_energy_recovered", "sindy_fd_energy_recovered", "sindy_ad_energy_recovered"),
        ("Cross term", "cross_term_recovered", "sindy_null_cross_recovered", "rref_cross_recovered", "sindy_fd_cross_recovered", "sindy_ad_cross_recovered"),
        ("Angular momentum", "ang_mom_recovered", "sindy_null_ang_mom_recovered", "rref_ang_mom_recovered", "sindy_fd_ang_mom_recovered", "sindy_ad_ang_mom_recovered"),
        ("Full 4-generator set", "full_set_recovered", "sindy_null_full_set_recovered", "rref_full_set_recovered", None, None),
    ]
    for label, csnp_col, null_col, rref_col, fd_col, ad_col in rows:
        csnp_stats = rate_ci(csnp_col)
        null_stats = rate_ci(null_col)
        rref_stats = rate_ci(rref_col)
        fd_stats = rate_ci(fd_col) if fd_col else None
        ad_stats = rate_ci(ad_col) if ad_col else None
        out.write(f"{label} & {cell(csnp_stats, bold=csnp_stats and csnp_stats[0]>0)} & "
                  f"{cell(null_stats)} & {cell(rref_stats, bold=rref_stats and rref_stats[0]>0)} & "
                  f"{cell(fd_stats)} & {cell(ad_stats)} \\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# SOS sensitivity (tab:sos)
# ----------------------------------------------------------------------

def table_sos(out):
    df = load_csv("sos_sensitivity_results.csv")
    if df is None:
        print("WARNING: sos_sensitivity_results.csv missing – Table 10 skipped.")
        return

    required = ["feasible_reduced", "feasible_redundant"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in SOS CSV.")
            return

    n = len(df)
    reduced_k = df["feasible_reduced"].sum()
    redundant_k = df["feasible_redundant"].sum()

    reduced_rate = reduced_k / n
    redundant_rate = redundant_k / n
    failure_rate = 1 - redundant_rate

    ci_reduced = wilson_interval(reduced_k, n)
    ci_redundant = wilson_interval(redundant_k, n)

    write_table_header(out,
        "SOS solver sensitivity to redundant constraints (30 random test polynomials).",
        "tab:sos",
        "lcc")
    out.write("\\textbf{Constraint set} & \\textbf{Feasibility rate} & \\textbf{95\\% Wilson CI} \\\\\n")
    out.write("\\midrule\n")
    # Double braces escape the literal braces in the f-string
    out.write(f"Reduced Gr\\\"{{o}}bner basis & {reduced_rate:.0%} & [{ci_reduced[0]:.0%}, {ci_reduced[1]:.0%}] \\\\\n")
    out.write(f"Redundant generating set & {redundant_rate:.0%} & [{ci_redundant[0]:.0%}, {ci_redundant[1]:.0%}] \\\\\n")
    out.write(f"Failure rate due to redundancy & {failure_rate:.0%} & --- \\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Appendix: Feynman per-equation
# ----------------------------------------------------------------------

# The merged appendix table (equation list + per-equation rates; labels
# tab:feynman-full and tab:feynman-per-eq on one table) needs the human-
# readable invariant form of each equation, which is a typesetting choice
# rather than data, so the LaTeX form strings live here. Row order is the
# paper's (natural equation-number order within each degree block); Vars
# and Deg. are read from feynman_polynomials.py, rates from the CSV.
FEYNMAN_FORM_LATEX = [
    # degree-2 block
    ("I.8.14",   "$(x_2-x_1)^2+(y_2-y_1)^2-d^2$"),
    ("circle_locus", "$x^2+y^2-r^2$"),
    ("newtons_second_law", "$F-ma$"),
    ("momentum_conservation_1d", "$m_1v_1+m_2v_2-P$"),
    ("work_energy", "$W-Fd$"),
    ("momentum_definition",  "$p-mv$"),
    ("distance_3d", "$x^2+y^2+z^2-d^2$"),
    ("angular_momentum_2d", "$xv_y-yv_x-L$"),
    ("power_definition", "$P-Fv$"),
    ("frequency_shift_linear", "$f-f_0-vf$"),
    ("wavenumber_dispersion",  "$\\omega-\\omega_0-vk$"),
    ("I.34.27",  "$E-h\\nu$"),
    ("planck_relation_duplicate", "$h\\nu-E$"),
    ("I.39.22", "$pV-nT$"),
    # degree-3 block
    ("algebraic_cubic_toy",   "$x^3+2xy+y^2$"),
    ("I.12.2",   "$Fr^2-kq_1q_2$"),
    ("centripetal_force",   "$rF-mv^2$"),
    ("charge_force_toy",   "$F-q_1q_2r$"),
    ("I.13.4",   "$KE-\\tfrac12mv^2$"),
    ("kinetic_energy_scaled",  "$2\\,KE-mv^2$"),
    ("I.14.3",   "$U-mgh$"),
    ("kinematics_position",   "$x-ut-\\tfrac12at^2$"),
    ("parallel_axis_theorem",   "$I-I_c-md^2$"),
    ("gravitational_pe_orbit",  "$Ur+Gm_1m_2$"),
    ("I.43.31",  "$D-kT\\mu$"),
    ("polarization_correction", "$n\\,kT-n_0\\,kT-n_0pE_f$"),
]


def table_feynman_per_eq(out):
    """Merged appendix table: equation list (invariant form, Vars, Deg.)
    plus per-equation exact recovery rates, exactly as the paper's single
    merged table carrying both \\label{tab:feynman-full} and
    \\label{tab:feynman-per-eq}."""
    df_sum = load_csv("feynman_results_summary.csv")
    if df_sum is None:
        print("WARNING: feynman_results_summary.csv missing – appendix skipped.")
        return

    required = ["name", "sigma", "srgb_rate"]
    for col in required:
        if col not in df_sum.columns:
            print(f"ERROR: column '{col}' missing in feynman_results_summary.csv")
            return

    rates = {}
    for _, row in df_sum.iterrows():
        rates.setdefault(row["name"], {})[round(float(row["sigma"]), 2)] = row["srgb_rate"]

    import sympy as sp
    from feynman_polynomials import feynman_polynomials
    meta = {}
    for fid, expr_str, var_names, _ranges in feynman_polynomials:
        expr = sp.expand(sp.sympify(expr_str))
        meta[fid] = (len(var_names), sp.total_degree(expr))

    form_ids = {fid for fid, _ in FEYNMAN_FORM_LATEX}
    if form_ids != set(rates.keys()) or form_ids != set(meta.keys()):
        print("ERROR: FEYNMAN_FORM_LATEX out of sync with CSV/feynman_polynomials:"
              f" only-in-forms={sorted(form_ids - set(rates))},"
              f" only-in-csv={sorted(set(rates) - form_ids)}")
        return

    out.write("\\begin{table}[ht]\n")
    out.write("\\centering\n")
    out.write("\\caption{Feynman benchmark: the 26 polynomial equations in the implicit\n"
              "polynomial form actually used by \\texttt{feynman\\_polynomials.py} (not\n"
              "the original, sometimes non-polynomial, physics formula each is derived\n"
              "from; Vars and Deg.\\ are read from the generating source), with\n"
              "per-equation exact recovery rates over 30 seeds per condition.}\n")
    out.write("\\label{tab:feynman-full}\n")
    out.write("\\label{tab:feynman-per-eq}\n")
    out.write("\\small\n")
    out.write("\\begin{tabular}{llccccc}\n")
    out.write("\\toprule\n")
    out.write("\\textbf{ID} & \\textbf{Invariant form ($p(x)=0$)} & \\textbf{Vars} &\n"
              "  \\textbf{Deg.} & $\\sigma=0.00$ & $\\sigma=0.02$ & $\\sigma=0.05$\\\\\n")
    out.write("\\midrule\n")

    prev_deg = None
    for fid, form in FEYNMAN_FORM_LATEX:
        n_vars, deg = meta[fid]
        if prev_deg is not None and deg != prev_deg:
            out.write("\\addlinespace\n")
        prev_deg = deg
        r = rates[fid]
        cells = " & ".join(f"{r[s]:.2f}" for s in (0.00, 0.02, 0.05))
        out.write(f"{fid} & {form} & {n_vars} & {deg} & {cells}\\\\\n")

    out.write("\\bottomrule\n")
    out.write("\\end{tabular}\n")
    out.write("\\end{table}\n\n")


# ----------------------------------------------------------------------
# Table: Adaptive degree discovery
# ----------------------------------------------------------------------

def table_adaptive_degree(out):
    df = load_csv("ablation_adaptive_degree_circle.csv")
    if df is None:
        print("WARNING: ablation_adaptive_degree_circle.csv missing – adaptive degree table skipped.")
        return
    n = len(df)
    exact_k = df["exact"].sum()
    deg_ok_k = (df["recovered_degree"] == 2).sum()
    exact_rate = exact_k / n
    deg_rate = deg_ok_k / n
    ci_exact = wilson_interval(exact_k, n)
    ci_deg = wilson_interval(deg_ok_k, n)
    
    write_table_header(out,
        "Adaptive degree discovery on the unit circle ($\\sigma=0.01$, $D_{\\max}=4$).",
        "tab:adaptive-degree",
        "lcc")
    out.write("\\textbf{Metric} & \\textbf{Rate} & \\textbf{95\\% Wilson CI} \\\\\n")
    out.write("\\midrule\n")
    out.write(f"Exact recovery & {exact_rate:.0%} & [{ci_exact[0]:.0%}, {ci_exact[1]:.0%}] \\\\\n")
    out.write(f"Correct degree (2) selected & {deg_rate:.0%} & [{ci_deg[0]:.0%}, {ci_deg[1]:.0%}] \\\\\n")
    write_table_footer(out)


# ----------------------------------------------------------------------
# Table: Fixed‑degree recovery rates
# ----------------------------------------------------------------------

def table_fixed_degree(out):
    """New table: exact recovery for each fixed degree (1..Dmax)."""
    df = load_csv("ablation_fixed_degree_summary.csv")
    if df is None:
        print("WARNING: ablation_fixed_degree_summary.csv missing – fixed-degree table skipped.")
        return
    required = ["degree", "rate", "ci_low", "ci_high"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in fixed-degree summary CSV.")
            return

    write_table_header(out,
        "Fixed‑degree exact recovery rates on the unit circle ($\\sigma=0.01$, 30 seeds).",
        "tab:fixed-degree",
        "ccc")
    out.write("\\textbf{Degree} & \\textbf{Recovery rate} & \\textbf{95\\% Wilson CI} \\\\\n")
    out.write("\\midrule\n")
    for _, row in df.iterrows():
        D = int(row["degree"])
        # Rate and CI as separate cells to match the three-column header;
        # a single combined format_ci cell would leave rows a column short.
        out.write(f"{D} & {row['rate']*100:.1f}\\% & "
                  f"[{row['ci_low']*100:.0f}\\%, {row['ci_high']*100:.0f}\\%] \\\\\n")
    write_table_footer(out)



def table_loop_invariants(out):
    df = load_csv("loop_invariants_summary.csv")
    if df is None:
        print("WARNING: loop_invariants_summary.csv missing – loop invariants table skipped.")
        return

    def cell(row, meth):
        return (f"{row[f'{meth}_rate']*100:.0f}\\% "
                f"[{row[f'{meth}_ci_low']*100:.0f}\\%, {row[f'{meth}_ci_high']*100:.0f}\\%]")

    # The four nullspace-reading methods (SR-GB+CSNP from floating point,
    # the three exact-arithmetic baselines) have returned identical rates on
    # every benchmark; the paper's table therefore collapses them into one
    # column, with SINDy-ST (all-targets implicit STLSQ ensemble, the same
    # standard configuration as every other benchmark) reported separately.
    # Fall back to one column per method if they ever disagree.
    nullspace_methods = ["SR-GB+CSNP", "SINDy-null (exact)", "OMP (exact)",
                         "Dense SVD+GB (exact)"]
    collapsed = all(
        np.allclose(df[f"{m}_rate"], df[f"{nullspace_methods[0]}_rate"])
        for m in nullspace_methods
    )

    if collapsed:
        write_table_header(out,
            "Loop invariant recovery rates (30 seeds, 95\\% Wilson CI). The "
            "three exact-arithmetic baselines (SINDy-null, OMP, Dense SVD+GB) "
            "read the same exact integer nullspace, while SR-GB+CSNP runs its "
            "full floating-point pipeline on the raw traces; all four return "
            "identical results on every benchmark, so one column reports all "
            "four. SINDy-ST runs the standard all-targets implicit STLSQ "
            "ensemble and is reported separately, since it regresses implicit "
            "target columns rather than reading the nullspace.",
            "tab:loop-invariants",
            "lcc")
        out.write("\\textbf{Benchmark} & \\textbf{Nullspace methods} & \\textbf{SINDy-ST}\\\\\n")
        out.write("\\midrule\n")
        for _, row in df.iterrows():
            out.write(f"{row['benchmark']} & {cell(row, 'SR-GB+CSNP')} & "
                      f"{cell(row, 'SINDy-ST')}\\\\\n")
        write_table_footer(out)
        return

    methods = nullspace_methods[:2] + ["SINDy-ST"] + nullspace_methods[2:]
    write_table_header(out,
        "Loop invariant recovery rates (30 seeds, 95\\% Wilson CI). Baselines use exact integer arithmetic; SR-GB+CSNP runs the full floating-point pipeline.",
        "tab:loop-invariants",
        "l" + "c"*len(methods))
    col_headers = " & ".join([f"\\textbf{{{m.split()[0]}}}" for m in methods])
    out.write(f"\\textbf{{Benchmark}} & {col_headers} \\\\\n")
    out.write("\\midrule\n")
    for _, row in df.iterrows():
        entries = [cell(row, meth) for meth in methods]
        out.write(f"{row['benchmark']} & " + " & ".join(entries) + " \\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: AVI border-basis cardinality vs SR-GB+CSNP
# ----------------------------------------------------------------------

def table_avi(out):
    df = load_csv("avi_baseline_summary.csv")
    if df is None:
        print("WARNING: avi_baseline_summary.csv missing – AVI table skipped.")
        return
    # (csv system key, display label) in the paper's row order.
    rows = [
        ("circle", "Circle"),
        ("sphere", "Sphere"),
        ("cubic_algebraic_toy", "Cubic (algebraic toy)"),
        ("kepler_angular_momentum", "Kepler angular momentum"),
        ("harmonic_oscillator_2d", "Harmonic oscillator (transition)"),
        ("feynman_circle_locus", "Feynman circle locus"),
        ("feynman_angular_momentum_2d", "Feynman angular momentum (2D)"),
        ("feynman_newtons_second_law", "Feynman Newton's second law"),
        ("feynman_I.12.2", "Feynman I.12.2"),
        ("feynman_kinematics_position", "Feynman kinematics (position)"),
    ]
    sigmas = [0.0, 0.01, 0.02]

    def card(v):
        # Sub-unity cardinalities carry a second decimal in the paper.
        return f"{v:.2f}" if v < 1.0 else f"{v:.1f}"

    # Determine n_seeds
    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell_exact(rate):
        return _fmt_rate_with_ci(rate, n_seeds)

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{AVI border-basis cardinality vs \\SRGBCSNP, mean over 30 seeds "
              "($N=5000$). A single true generator underlies every system; AVI's "
              "cardinality is the size of the border basis it returns.}\n")
    out.write("\\label{tab:avi}\n\\small\n")
    out.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    out.write("& \\multicolumn{3}{c}{\\textbf{AVI cardinality}} & "
              "\\multicolumn{3}{c}{\\textbf{\\SRGBCSNP exact}}\\\\\n")
    out.write("\\textbf{System} & $\\sigma{=}0$ & $\\sigma{=}0.01$ & $\\sigma{=}0.02$ & "
              "$\\sigma{=}0$ & $\\sigma{=}0.01$ & $\\sigma{=}0.02$\\\\\n\\midrule\n")
    row_data = []
    for key, label in rows:
        sub = df[df["system"] == key]
        cards, exacts = [], []
        for s in sigmas:
            r = sub[np.isclose(sub["sigma"], s)]
            if r.empty:
                cards.append("---"); exacts.append("---"); continue
            cards.append(card(float(r["mean_avi_cardinality"].iloc[0])))
            rate = float(r["srgb_exact_rate"].iloc[0])
            exacts.append(cell_exact(rate))
        row_data.append((label, tuple(cards + exacts)))

    # Merge consecutive rows with identical values (labels joined by ", "),
    # matching the paper's convention of not repeating a result row-for-row.
    # A shared "Feynman " prefix across every label in the merge is written
    # once, e.g. "Feynman circle locus, angular momentum (2D)" rather than
    # repeating the prefix.
    i = 0
    while i < len(row_data):
        j = i
        while j + 1 < len(row_data) and row_data[j + 1][1] == row_data[i][1]:
            j += 1
        labels = [row_data[k][0] for k in range(i, j + 1)]
        if len(labels) > 1 and all(l.startswith("Feynman ") for l in labels):
            merged_label = "Feynman " + ", ".join(l[len("Feynman "):] for l in labels)
        else:
            merged_label = ", ".join(labels)
        out.write(f"{merged_label} & " + " & ".join(row_data[i][1]) + "\\\\\n")
        i = j + 1
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: fixed-dt vs variable-dt discriminator
# ----------------------------------------------------------------------

def table_dt_discriminator(out):
    df = load_csv("dt_discriminator_summary.csv")
    if df is None:
        print("WARNING: dt_discriminator_summary.csv missing – discriminator table skipped.")
        return
    rows = [
        ("exact_flow", "Exact rotation flow"),
        ("fixed_dt_symplectic_euler", "Symplectic Euler, fixed $dt$"),
        ("variable_dt", "Symplectic Euler, variable $dt$"),
    ]

    def cell(rate, n, bold=False):
        return _fmt_rate_with_ci(rate, n) if not bold else f"\\textbf{{{_fmt_rate_with_ci(rate, n)}}}"

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Fixed-$dt$ vs.\\ variable-$dt$ discriminator: exact recovery "
              "over 30 seeds ($N=5000$, $\\sigma=0$). \\SRGBCSNP, \\OMP, and \\SINDy-null "
              "search the full fifteen-monomial transition-pair dictionary and are scored "
              "against $H$ directly. \\SINDy-FD and \\SINDy-AD search a different, "
              "restricted state-only dictionary $(x_t,v_t)$, have no entry on variable "
              "$dt$ (no single fixed step for their derivative estimators), and are each "
              "scored two ways: against the state energy directly, and against $H$ via the "
              "induced form $p(x_t,v_t)-p(x_{\\mathrm{next}},v_{\\mathrm{next}})$. "
              "\\SINDy-AD differs from \\SINDy-FD only in its derivative estimator (a "
              "smoothing spline in place of a finite difference); it is a lightweight "
              "proxy for Kaheman et al.'s joint denoising/noise-model method, not a "
              "reimplementation, which is a fair comparison at this table's $\\sigma=0$ "
              "since there is no noise for the joint noise model to characterize.}\n")
    out.write("\\label{tab:edisc}\n")
    out.write("\\begin{tabular}{lcccccccc}\n\\toprule\n")
    out.write("\\textbf{Regime} & \\SRGBCSNP & \\OMP & \\SINDy-null & "
              "\\SINDy-FD (state) & \\SINDy-FD (transition) & "
              "\\SINDy-AD (state) & \\SINDy-AD (transition)\\\\\n\\midrule\n")
    for key, label in rows:
        r = df[df["mode"] == key]
        if r.empty:
            continue
        n = int(r["n_seeds"].iloc[0])
        csnp = float(r["csnp_rate"].iloc[0])
        omp = float(r["omp_rate"].iloc[0])
        snull = float(r["sindy_null_rate"].iloc[0])
        fd_state = r["sindy_fd_state_rate"].iloc[0] if "sindy_fd_state_rate" in r.columns else float("nan")
        fd_trans = r["sindy_fd_transition_rate"].iloc[0] if "sindy_fd_transition_rate" in r.columns else float("nan")
        ad_state = r["sindy_ad_state_rate"].iloc[0] if "sindy_ad_state_rate" in r.columns else float("nan")
        ad_trans = r["sindy_ad_transition_rate"].iloc[0] if "sindy_ad_transition_rate" in r.columns else float("nan")
        fd_state_s = cell(float(fd_state), n, bold=float(fd_state) > 0) if pd.notna(fd_state) else "---"
        fd_trans_s = cell(float(fd_trans), n, bold=float(fd_trans) > 0) if pd.notna(fd_trans) else "---"
        ad_state_s = cell(float(ad_state), n, bold=float(ad_state) > 0) if pd.notna(ad_state) else "---"
        ad_trans_s = cell(float(ad_trans), n, bold=float(ad_trans) > 0) if pd.notna(ad_trans) else "---"
        out.write(f"{label} & {cell(csnp, n, bold=csnp>0)} & "
                  f"{cell(omp, n)} & {cell(snull, n)} & {fd_state_s} & {fd_trans_s} & "
                  f"{ad_state_s} & {ad_trans_s}\\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: sparsity-first vs rationality-first selection, fixed-dt harmonic
# oscillator
# ----------------------------------------------------------------------

def table_sparsity_vs_rationality(out):
    df = load_csv("ablation_sparsity_vs_rationality_harmonic_summary.csv")
    if df is None:
        print("WARNING: ablation_sparsity_vs_rationality_harmonic_summary.csv missing "
              "- sparsity-vs-rationality table skipped.")
        return
    required = ["dt", "sparsity_rate", "rationality_rate", "n_seeds"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in ablation_sparsity_vs_rationality_harmonic_summary.csv")
            return

    def cell(rate, n, bold=False):
        return _fmt_rate_with_ci(rate, n) if not bold else f"\\textbf{{{_fmt_rate_with_ci(rate, n)}}}"

    write_table_header(out,
        "Sparsity-first vs.\\ rationality-first selection on the fixed-$dt$ "
        "harmonic oscillator's transition-pair dictionary ($N=5000$, "
        "$\\sigma=0$, 30 seeds per $dt$). Both arms search the identical "
        "candidate pool; only the final selection key differs.",
        "tab:sparsity-vs-rationality",
        "c c c")
    out.write("$dt$ & \\textbf{Sparsity-first} & \\textbf{Rationality-first}\\\\\n\\midrule\n")
    for _, row in df.sort_values("dt").iterrows():
        n = int(row["n_seeds"])
        sp = float(row["sparsity_rate"])
        ra = float(row["rationality_rate"])
        out.write(f"{row['dt']:.2f} & {cell(sp, n)} & {cell(ra, n, bold=ra>0)}\\\\\n")
    out.write("\\midrule\n")
    n_all = int(df["n_seeds"].sum())
    sp_all = float((df["sparsity_rate"] * df["n_seeds"]).sum() / n_all)
    ra_all = float((df["rationality_rate"] * df["n_seeds"]).sum() / n_all)
    out.write(f"Pooled ($n={n_all}$) & {cell(sp_all, n_all)} & {cell(ra_all, n_all, bold=ra_all>0)}\\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: unit-invariance stress test
# ----------------------------------------------------------------------

def table_unit_scale(out):
    df = load_csv("unit_scale_sensitivity_summary.csv")
    if df is None:
        print("WARNING: unit_scale_sensitivity_summary.csv missing – unit-scale table skipped.")
        return
    rows = [
        ("circle", "Circle"),
        ("oscillator_fixed_dt", "Oscillator, fixed $dt$"),
        ("kepler_angular_momentum", "Kepler angular momentum"),
        ("feynman_momentum_conservation_1d", "Feynman momentum conservation (1D)"),
    ]
    conditions = ["baseline", "rescaled", "standardized"]

    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell(rate, emphasize=False):
        s = _fmt_rate_with_ci(rate, n_seeds)
        return f"\\textbf{{{s}}}" if (emphasize and rate >= 1.0) else s

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Unit-invariance stress test: exact recovery under random per-quantity "
              "rescaling (30 seeds each). ``Baseline'' is the unmodified benchmark already "
              "reported elsewhere in this paper.}\n")
    out.write("\\label{tab:eunit}\n")
    out.write("\\begin{tabular}{lccc}\n\\toprule\n")
    out.write("\\textbf{Benchmark} & \\textbf{Baseline} & \\textbf{Rescaled} & "
              "\\textbf{Standardized}\\\\\n\\midrule\n")
    for key, label in rows:
        sub = df[df["benchmark"] == key]
        cells = []
        for cond in conditions:
            r = sub[sub["condition"] == cond]
            if r.empty:
                cells.append("---"); continue
            rate = float(r["rate"].iloc[0])
            cells.append(cell(rate, emphasize=(cond != "baseline")))
        out.write(f"{label} & " + " & ".join(cells) + "\\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: point-vortex multi-invariant recovery
# ----------------------------------------------------------------------

def table_vortex(out):
    df = load_csv("vortex_summary.csv")
    if df is None:
        print("WARNING: vortex_summary.csv missing – vortex table skipped.")
        return
    rows = [("vortex3", "3 vortices"),
            ("vortex4", "4 vortices"),
            ("vortex5", "5 vortices")]

    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell(rate):
        return _fmt_rate_with_ci(rate, n_seeds)

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Point-vortex multi-invariant recovery via full-nullspace "
              "deflation, degree-2 lift over the pooled state, mean over the seed "
              "grid ($\\sigma=0$). $P=\\sum_j x_j$ and $Q=\\sum_j y_j$ are the "
              "degree-1 linear impulses; $I=\\sum_j(x_j^2+y_j^2)-I_0$ is the degree-2 "
              "angular impulse. \\SRGBCSNP's $P$/$Q$/$I$ columns and ``Ideal match'' "
              "test ideal membership of the true generator against \\SRGBCSNP's own "
              "canonicalized (reduced Gr\\\"{o}bner basis) output. The "
              "RREF~\\cite{OellerichEmelianenko2024} columns instead test a direct "
              "scalar-multiple match against each raw generator, with no further "
              "reduction, since that is what row-echelon disambiguation actually "
              "delivers; an ideal-equality reading of RREF's raw, non-minimal "
              "nullspace basis is close to tautological once the nullspace itself "
              "is numerically correct (Section~\\ref{sec:transition}) and is "
              "omitted here for that reason. 30 seeds, Wilson 95\\% CI on Ideal "
              "match.}\n")
    out.write("\\label{tab:vortex}\n")
    out.write("\\footnotesize\n")
    out.write("\\begin{tabular}{lccccccc}\n\\toprule\n")
    out.write("\\textbf{System} & $P$ & $Q$ & $I$ & \\textbf{Ideal match} & "
              "RREF $P$ & RREF $Q$ & RREF $I$\\\\\n\\midrule\n")
    if not any(c.startswith("rref_") for c in df.columns):
        print("WARNING: vortex_summary.csv has no rref_* columns -- RREF "
              "cells will crash/skip. Rerun benchmark_vortex.py to populate "
              "them.")
        return
    for key, label in rows:
        r = df[df["system"] == key]
        if r.empty:
            continue
        rr = r.iloc[0]
        ci = f"[{float(rr['ci_low'])*100:.0f}\\%, {float(rr['ci_high'])*100:.0f}\\%]"
        out.write(f"{label} & {cell(rr['P_rate'])} & {cell(rr['Q_rate'])} & "
                  f"{cell(rr['I_rate'])} & {cell(rr['ideal_match_rate'])} & "
                  f"{cell(rr['rref_P_direct_rate'])} & {cell(rr['rref_Q_direct_rate'])} & "
                  f"\\textbf{{{cell(rr['rref_I_direct_rate'])}}}\\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: difference-dictionary generality probe
# ----------------------------------------------------------------------

def table_diff_dictionary_generality(out):
    """Q_dt / H recovery per (integrator, dt, sigma) on the difference
    dictionary (sec:modified-equation-future's generality probe)."""
    df = load_csv("difference_dictionary_generality_rates.csv")
    if df is None:
        print("WARNING: difference_dictionary_generality_rates.csv missing – table skipped.")
        return

    required = ["integrator", "dt", "sigma", "Qdt_rate", "H_rate",
                "Qdt_ci_low", "Qdt_ci_high"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in difference_dictionary_generality_rates.csv")
            return

    pretty = {"symplectic_euler": "Symplectic Euler",
              "stormer_verlet": "St\\\"{o}rmer--Verlet"}
    write_table_header(out,
        "Difference-dictionary recovery of the modified invariant $Q_{dt}$ "
        "(30 seeds, 95\\% Wilson CI). $H$ rate counts trials returning the "
        "true energy $x^2+v^2$ instead of $Q_{dt}$, the collapse mode "
        "expected once $3\\sigma$ exceeds the $Q_{dt}$--$H$ coefficient gap.",
        "tab:diff-dict-generality",
        "llccc")
    out.write("Integrator & $dt$ & $\\sigma$ & $Q_{dt}$ recovery & $H$ rate \\\\\n")
    out.write("\\midrule\n")
    df = df.sort_values(["integrator", "dt", "sigma"],
                        ascending=[False, True, True])
    for _, r in df.iterrows():
        name = pretty.get(r["integrator"], r["integrator"])
        qdt = format_ci(r["Qdt_rate"], r["Qdt_ci_low"], r["Qdt_ci_high"])
        out.write(f"{name} & {r['dt']:g} & {r['sigma']:.2f} & {qdt} "
                  f"& {r['H_rate']*100:.1f}\\% \\\\\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Table: noise ceiling for higher-degree invariants
# ----------------------------------------------------------------------

def table_noise_ceiling(out):
    df = load_csv("noise_ceiling_probe_summary.csv")
    if df is None:
        print("WARNING: noise_ceiling_probe_summary.csv missing – noise-ceiling table skipped.")
        return
    # (csv system key, paper row label) in the paper's order.
    rows = [("circle_locus (deg 2)", "Circle locus (deg 2)"),
            ("I.12.2 (deg 3)", "I.12.2 (deg 3)")]
    sigmas = [0.0, 0.005, 0.01, 0.02, 0.05]

    def fmt_l2(v):
        # Near-zero at sigma=0 is a noiseless exact-recovery floor, reported
        # qualitatively; otherwise 2 significant figures, plain for O(1--100).
        v = float(v)
        if v < 1e-9:
            return "$\\approx 0$"
        if 1.0 <= v < 100.0:
            return f"${v:.1f}$"
        exp = int(np.floor(np.log10(v)))
        mant = v / 10 ** exp
        return f"${mant:.1f}\\times10^{{{exp}}}$"

    def fmt_rho(v):
        # sigma=0's gap ratio is astronomically large (exact nullspace); the
        # point is only that it is far from the degenerate rho ~ 1 regime.
        v = float(v)
        if v > 1e6:
            return "$\\gg 1$"
        return f"${v:.1f}$"

    out.write("\\begin{table}[H]\n\\centering\n")
    out.write("\\caption{Raw nullspace-coefficient $L_2$ error and mean gap ratio "
              "$\\rho_{M-1}$, degree-2 vs.\\ degree-3 (10 seeds, $N=5000$). Full "
              "per-seed output in \\texttt{Results/noise\\_ceiling\\_probe.csv} and "
              "\\texttt{Results/noise\\_ceiling\\_probe\\_summary.csv}.}\n")
    out.write("\\label{tab:noise-ceiling-probe}\n")
    out.write("\\begin{tabular}{lccccc}\n\\toprule\n")
    out.write("& \\multicolumn{5}{c}{$\\sigma$}\\\\\n")
    out.write("\\textbf{Equation} & 0.000 & 0.005 & 0.010 & 0.020 & 0.050\\\\\n\\midrule\n")
    for i, (key, label) in enumerate(rows):
        sub = df[df["system"] == key]
        l2, rho = [], []
        for s in sigmas:
            r = sub[np.isclose(sub["sigma"], s)]
            if r.empty:
                l2.append("---"); rho.append("---"); continue
            l2.append(fmt_l2(r["mean_coeff_l2_error"].iloc[0]))
            rho.append(fmt_rho(r["mean_gap_ratio"].iloc[0]))
        out.write(f"{label}, $L_2$ error   & " + " & ".join(l2) + "\\\\\n")
        out.write(f"{label}, $\\rho_{{M-1}}$  & " + " & ".join(rho) + "\\\\\n")
        if i == 0:
            out.write("\\addlinespace\n")
    write_table_footer(out)

# ----------------------------------------------------------------------
# Tables: Qmax (rational-denominator cap) and eps (snap tolerance) sensitivity
# ----------------------------------------------------------------------

# Readable invariant label per recovery-system name (matches
# ablation_qmax_tolerance.py RECOVERY_SYSTEMS).
_QMAX_SYS_LABEL = {
    "circle":         "$x^2+y^2-1$",
    "ellipse_357":    "$3x^2+5y^2-7$",
    "ellipse_5711":   "$5x^2+7y^2-11$",
    "ellipse_71113":  "$7x^2+11y^2-13$",
    "ellipse_131719": "$13x^2+17y^2-19$",
    "ellipse_171923": "$17x^2+19y^2-23$",
}
_QMAX_NEG_LABEL = {
    "generic_cloud": "Generic cloud (no relation)",
    "irrational_pi": "$x^2+\\pi y^2-1$ (irrational)",
}


def table_qmax_recovery(out):
    """Exact recovery vs the denominator cap Qmax at sigma=0. Rows are ordered
    by the invariant's minimum representable denominator q_star, defined the
    same way sr_gb.snap_round itself normalises a candidate: divide by
    whichever coefficient has the largest magnitude, not by whichever
    normalisation happens to give the smallest denominator (ablation_qmax_
    tolerance.py's _min_denom; see its docstring and TODO.md item 13 for the
    earlier version of this function, which minimised over every possible
    normalisation and understated q_star). Recovery is a clean step function
    of this q_star: 0% at every Qmax below it, 100% at every Qmax at or
    above it, with no observed margin or lag."""
    df = load_csv("ablation_qmax_recovery_summary.csv")
    if df is None:
        print("WARNING: ablation_qmax_recovery_summary.csv missing - Qmax recovery table skipped.")
        return
    required = ["system", "min_denom", "qmax", "rate", "med_runtime_s"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in ablation_qmax_recovery_summary.csv")
            return
    qmaxes = sorted(df["qmax"].unique())
    # order systems by min_denom
    order = (df[["system", "min_denom"]].drop_duplicates()
             .sort_values("min_denom")["system"].tolist())
    
    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell(rate):
        return _fmt_rate_with_ci(rate, n_seeds)

    write_table_header(out,
        "Exact recovery vs the rational-denominator cap $Q_{\\max}$ (\\texttt{max\\_denom}) "
        "on conics with coprime integer coefficients, sampled exactly on the "
        "variety ($\\sigma=0$, 30 seeds). Each invariant's minimum representable "
        "denominator $q^\\star$ is listed: the smallest $Q_{\\max}$ at which the "
        "invariant is representable under snap-rounding's own normalisation "
        "(divide by whichever coefficient has the largest magnitude, not "
        "whichever normalisation happens to be cheapest). Recovery is a clean "
        "step function of $q^\\star$: $0\\%$ at every $Q_{\\max}<q^\\star$ and "
        "$100\\%$ at every $Q_{\\max}\\ge q^\\star$, with no observed margin, so "
        "e.g.\\ the $q^\\star=19$ row stays at $0\\%$ through the default "
        "$Q_{\\max}=16$ and turns on immediately at $Q_{\\max}=32$.",
        "tab:qmax-recovery",
        "l c " + "c" * len(qmaxes))
    out.write("\\textbf{Invariant} & $q^\\star$ & "
              + " & ".join(f"$Q_{{\\max}}{{=}}{q}$" for q in qmaxes) + " \\\\\n")
    out.write("\\midrule\n")
    for sysname in order:
        sub = df[df["system"] == sysname]
        md = int(sub["min_denom"].iloc[0])
        label = _QMAX_SYS_LABEL.get(sysname, sysname)
        cells = []
        for q in qmaxes:
            row = sub[sub["qmax"] == q]
            cells.append(cell(row.iloc[0]['rate']) if len(row) == 1 else "---")
        out.write(f"{label} & {md} & " + " & ".join(cells) + " \\\\\n")
    # median runtime per Qmax (across systems/seeds) to show cost vs Qmax
    out.write("\\midrule\n")
    rt_cells = []
    for q in qmaxes:
        med = df[df["qmax"] == q]["med_runtime_s"].median()
        rt_cells.append(f"{med:.3f}" if med == med else "---")
    out.write("\\emph{median runtime (s)} & & " + " & ".join(rt_cells) + " \\\\\n")
    write_table_footer(out)


def table_qmax_falsepos(out):
    """False-positive (non-abstention) rate vs Qmax on negative systems that
    have no low-denominator rational invariant."""
    df = load_csv("ablation_qmax_falsepos_summary.csv")
    if df is None:
        print("WARNING: ablation_qmax_falsepos_summary.csv missing - Qmax false-positive table skipped.")
        return
    required = ["system", "qmax", "fp_rate"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in ablation_qmax_falsepos_summary.csv")
            return
    qmaxes = sorted(df["qmax"].unique())
    
    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell(fp_rate):
        return _fmt_rate_with_ci(fp_rate, n_seeds)

    write_table_header(out,
        "False-positive rate (a nonzero generator returned when the method "
        "should abstain) vs the denominator cap $Q_{\\max}$, 30 seeds. Neither "
        "a generic point cloud nor a variety with an irrational coefficient "
        "admits a low-denominator rational invariant; raising $Q_{\\max}$ well "
        "past the default 16 does not manufacture one.",
        "tab:qmax-falsepos",
        "l " + "c" * len(qmaxes))
    out.write("\\textbf{Negative system} & "
              + " & ".join(f"$Q_{{\\max}}{{=}}{q}$" for q in qmaxes) + " \\\\\n")
    out.write("\\midrule\n")
    for sysname in _QMAX_NEG_LABEL:
        sub = df[df["system"] == sysname]
        if len(sub) == 0:
            continue
        cells = []
        for q in qmaxes:
            row = sub[sub["qmax"] == q]
            cells.append(cell(row.iloc[0]['fp_rate']) if len(row) == 1 else "---")
        out.write(f"{_QMAX_NEG_LABEL[sysname]} & " + " & ".join(cells) + " \\\\\n")
    write_table_footer(out)


def table_eps_sensitivity(out):
    """Recovery (noisy circle) and abstention (irrational conic) vs the snap /
    rational-window tolerance eps."""
    df = load_csv("ablation_eps_sensitivity_summary.csv")
    if df is None:
        print("WARNING: ablation_eps_sensitivity_summary.csv missing - eps table skipped.")
        return
    required = ["eps", "recovery_rate", "fp_rate"]
    for col in required:
        if col not in df.columns:
            print(f"ERROR: column '{col}' missing in ablation_eps_sensitivity_summary.csv")
            return
    
    n_seeds = 30
    if "n_seeds" in df.columns:
        n_seeds = int(df["n_seeds"].iloc[0])

    def cell(rate):
        return _fmt_rate_with_ci(rate, n_seeds)

    write_table_header(out,
        "Sensitivity to the snap / rational-window tolerance $\\varepsilon$, "
        "30 seeds. Recovery is measured on the unit circle at $\\sigma=0.01$; "
        "the false-positive column is abstention on the irrational conic "
        "$x^2+\\pi y^2-1$. Recovery is stable across two orders of magnitude of "
        "$\\varepsilon$ around the default $10^{-4}$.",
        "tab:eps-sensitivity",
        "c c c")
    out.write("$\\varepsilon$ & \\textbf{Circle recovery} & \\textbf{False positive (irrational)} \\\\\n")
    out.write("\\midrule\n")
    for _, r in df.sort_values("eps", ascending=False).iterrows():
        eps_val = r["eps"]
        rec_str = cell(r["recovery_rate"])
        fp_str = cell(r["fp_rate"])
        out.write(f"$10^{{{int(round(np.log10(eps_val)))}}}$ & {rec_str} & {fp_str} \\\\\n")
    write_table_footer(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _safe_call(table_fn, out):
    """Run one table_* function, catching any exception (missing/renamed
    column, empty-dataframe indexing, etc.) so a schema mismatch in one
    table warns and skips it instead of crashing main() and losing every
    table that would have been generated after it."""
    try:
        table_fn(out)
    except Exception as e:
        print(f"WARNING: {table_fn.__name__} failed ({e}) -- table skipped.",
              file=sys.stderr)


def main():
    out_filename = "Results/all_paper_tables.tex"
    os.makedirs("Results", exist_ok=True)
    with open(out_filename, "w") as f:
        f.write("% Automatically generated tables by generate_result_tables.py\n")
        f.write("% Run: python generate_result_tables.py\n\n")
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\begin{document}\n\n")

        # Table numbers are assigned automatically by LaTeX in the paper and
        # differ between the main and condensed versions, so these comments name
        # each table by content and \label only, never by a hardcoded number.
        f.write("% Computational-cost table is hardcoded in the paper.\n")
        f.write("% For runtime measurements, see runtime_breakdown_summary.csv\n\n")

        f.write("% Redundancy elimination (tab:synthetic)\n")
        _safe_call(table_redundancy, f)
        f.write("\n")

        f.write("% Over‑lifting robustness (tab:overlifting)\n")
        _safe_call(table_overlifting, f)
        f.write("\n")

        f.write("% Transition invariants (tab:transition)\n")
        _safe_call(table_transition, f)
        f.write("\n")

        f.write("% Noise sensitivity (tab:noise)\n")
        _safe_call(table_noise, f)
        f.write("\n")

        f.write("% Feynman overall + baseline comparison (tab:feynman / "
                "tab:feynman-baselines, merged as panels (a)/(b))\n")
        _safe_call(table_feynman_overall, f)
        f.write("\n")

        f.write("% Groebner-basis scalability timings (tab:gb-scalability)\n")
        _safe_call(table_gb_scalability, f)
        f.write("\n")

        f.write("% Holonomic constraints (tab:holonomic)\n")
        _safe_call(table_holonomic, f)
        f.write("\n")

        f.write("% OMP vs CSNP (tab:omp)\n")
        _safe_call(table_omp, f)
        f.write("\n")

        f.write("% Deflation multi-invariant recovery (tab:deflation)\n")
        _safe_call(table_deflation, f)
        f.write("\n")

        f.write("% SOS sensitivity (tab:sos)\n")
        _safe_call(table_sos, f)
        f.write("\n")

        f.write("% Appendix: Feynman per-equation\n")
        _safe_call(table_feynman_per_eq, f)
        f.write("\n")

        f.write("% Table: Adaptive degree discovery\n")
        _safe_call(table_adaptive_degree, f)
        f.write("\n")

        f.write("% Table: Fixed‑degree recovery\n")
        _safe_call(table_fixed_degree, f)
        f.write("\n")

        f.write("% Table: Loop invariants\n")
        _safe_call(table_loop_invariants, f)
        f.write("\n")

        f.write("% Table: AVI border-basis cardinality\n")
        _safe_call(table_avi, f)
        f.write("\n")

        f.write("% Table: fixed-dt vs variable-dt discriminator\n")
        _safe_call(table_dt_discriminator, f)
        f.write("\n")

        f.write("% Table: sparsity-first vs rationality-first selection\n")
        _safe_call(table_sparsity_vs_rationality, f)
        f.write("\n")

        f.write("% Table: unit-invariance stress test\n")
        _safe_call(table_unit_scale, f)
        f.write("\n")

        f.write("% Table: point-vortex multi-invariant recovery\n")
        _safe_call(table_vortex, f)
        f.write("\n")

        f.write("% Table: noise ceiling for higher-degree invariants\n")
        _safe_call(table_noise_ceiling, f)
        f.write("\n")

        f.write("% Table: difference-dictionary generality probe\n")
        _safe_call(table_diff_dictionary_generality, f)
        f.write("\n")

        f.write("% Table: Qmax (denominator cap) recovery / ceiling\n")
        _safe_call(table_qmax_recovery, f)
        f.write("\n")

        f.write("% Table: Qmax false-positive rate on negatives\n")
        _safe_call(table_qmax_falsepos, f)
        f.write("\n")

        f.write("% Table: eps (snap tolerance) sensitivity\n")
        _safe_call(table_eps_sensitivity, f)
        f.write("\n")

        f.write("\\end{document}\n")

    print(f"All tables written to {out_filename}\n")
    print("The computational-cost table is not generated from CSV (hardcoded in paper).")


if __name__ == "__main__":
    main()