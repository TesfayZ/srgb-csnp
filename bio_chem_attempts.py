#!/usr/bin/env python3
"""
bio_chem_attempts.py - Real biochemical-system invariant-discovery ATTEMPTS.

Consolidates three earlier scripts (benchmark_crn.py, benchmark_glycolysis.py,
benchmark_real_system_baselines.py) into one runner over the same five real
systems, split by field:
  * biology  - glycolytic oscillators: Wolf2000, Bier2000
  * chemistry - CRN signaling/cell-cycle models: Gardner, Kholodenko, Markevich

These are exploratory ATTEMPTS, not paper benchmarks. SR-GB+CSNP recovers no
NEW polynomial invariant on any of them, and the baselines fare no better;
the root cause is independent-IC perturbation breaking the pooled linear
totals, and the eliminated ADP/NADH species leaving the moiety SVD without a
constant column, so the true conserved quantities are not representable in
the raw sampled coordinates. The script is kept so the
attempt is reproducible and honestly recorded, not because it produces a
positive result, and it is deliberately NOT registered in
run_all_experiments.py or referenced in the paper.

For each system it:
  1. ensures a cached trajectory CSV (simulating from the BioModels SBML or the
     transcribed ODEs only when the committed CSV is missing, so a fresh clone
     reproduces the run offline, with no network access or SBML libraries);
  2. runs SR-GB+CSNP on the pooled state;
  3. runs the four nullspace / vanishing-ideal baselines (SINDy-null,
     SINDy-ST, OMP-on-nullspace, AVI border basis) on the same data.

Each (system, baseline) pair runs in its own subprocess with a wall-clock
timeout (_bio_chem_baseline_worker.py, run one at a time, never in parallel),
because numpy/scipy linear-algebra calls run in C and do not yield to Python
signal handlers until they return; only killing the whole subprocess can
actually preempt a stuck call.

Usage:
    python bio_chem_attempts.py                 # all five systems
    python bio_chem_attempts.py wolf            # a single system
    python bio_chem_attempts.py --quick         # one system, fast smoke test
    python bio_chem_attempts.py --srgb-only     # skip the baseline sweep
"""

import hashlib
import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sr_gb import sr_gb

RESULTS_DIR = "Results"
CRN_DIR = "crn_data"
GLYCO_DIR = "glycolysis_data"
for _d in (RESULTS_DIR, CRN_DIR, GLYCO_DIR):
    os.makedirs(_d, exist_ok=True)

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "_bio_chem_baseline_worker.py")
BASELINES = ["sindy_null", "sindy_st", "omp_nullspace", "avi_border_basis"]
BASELINE_TIMEOUT_S = 240

# Simulation constants shared with the original scripts.
CRN_N_SIMS = 10
CRN_TIME_POINTS = 200
CRN_T_MAX = 100          # seconds
CRN_IC_PERTURB_SEED = 0  # IC-perturbation draws are otherwise unseeded, so a
                          # resimulate after deleting the cached CSV would not
                          # reproduce the committed data
WOLF_N_SIMS = 10
WOLF_TIME_POINTS = 1000  # ensures N >> M


# ----------------------------------------------------------------------
# Cache fingerprinting (mirrors vortex_data.py's pattern: a cache written
# under one parameter combination must never be silently reused for a
# different one).
# ----------------------------------------------------------------------
def _params_fingerprint(payload):
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _meta_file(csv_file):
    base, _ = os.path.splitext(csv_file)
    return base + ".params.json"


def _load_cached(csv_file, fingerprint):
    meta_file = _meta_file(csv_file)
    if os.path.exists(csv_file) and os.path.exists(meta_file):
        with open(meta_file) as f:
            cached = json.load(f).get("fingerprint")
        if cached == fingerprint:
            return pd.read_csv(csv_file)
    return None


def _save_cache(csv_file, fingerprint, df):
    df.to_csv(csv_file, index=False)
    with open(_meta_file(csv_file), "w") as f:
        json.dump({"fingerprint": fingerprint}, f)


# ----------------------------------------------------------------------
# Simulation helpers (only used when a cached trajectory CSV is missing or
# stale relative to the simulation parameters that produced it)
# ----------------------------------------------------------------------
def download_sbml(url, dest):
    if os.path.exists(dest):
        return
    print(f"Downloading SBML from {url} ...")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")


def _load_roadrunner(sbml_file):
    try:
        import tellurium as te
        return te.loadSBML(sbml_file)
    except (ImportError, AttributeError):
        try:
            import roadrunner
            return roadrunner.RoadRunner(sbml_file)
        except ImportError:
            print("Please install roadrunner or tellurium: pip install roadrunner")
            sys.exit(1)


def simulate_crn(cfg):
    """CRN chemistry systems: RoadRunner/Tellurium over the BioModels SBML,
    initial concentrations perturbed +-20% across CRN_N_SIMS runs."""
    csv_file = cfg["csv_file"]
    fingerprint = _params_fingerprint({
        "sbml_file": cfg["sbml_file"],
        "n_sims": CRN_N_SIMS,
        "time_points": CRN_TIME_POINTS,
        "t_max": CRN_T_MAX,
        "ic_perturb_seed": CRN_IC_PERTURB_SEED,
    })
    cached = _load_cached(csv_file, fingerprint)
    if cached is not None:
        return cached
    print(f"CSV not found or stale. Simulating {cfg['description']}...")
    rr = _load_roadrunner(cfg["sbml_file"])
    species = rr.getFloatingSpeciesIds()
    init_vals = {}
    for s in species:
        try:
            init_vals[s] = rr.getInitialConcentration(s)
        except Exception:
            init_vals[s] = 0.5
    all_data = []
    rng = np.random.RandomState(CRN_IC_PERTURB_SEED)
    for sim_id in range(CRN_N_SIMS):
        rr.reset()
        for s in species:
            perturb = 1.0 + 0.2 * (rng.rand() - 0.5)
            try:
                rr.setInitConcentration(s, init_vals[s] * perturb)
            except Exception:
                pass
        sim_data = rr.simulate(0, CRN_T_MAX, CRN_TIME_POINTS - 1)
        df = pd.DataFrame(sim_data, columns=['time'] + list(species))
        df.insert(0, 'sim_id', sim_id)
        all_data.append(df)
    df_all = pd.concat(all_data, ignore_index=True)
    _save_cache(csv_file, fingerprint, df_all)
    print(f"Saved trajectories to {csv_file}")
    return df_all


def simulate_wolf(cfg):
    """Wolf2000 glycolytic oscillator: RoadRunner/Tellurium over the SBML,
    initial glucose swept across WOLF_N_SIMS runs."""
    csv_file = cfg["csv_file"]
    fingerprint = _params_fingerprint({
        "sbml_file": cfg["sbml_file"],
        "n_sims": WOLF_N_SIMS,
        "glucose_range": cfg["glucose_range"],
        "t_max": cfg["t_max"],
        "n_points": cfg["n_points"],
    })
    cached = _load_cached(csv_file, fingerprint)
    if cached is not None:
        return cached
    print("CSV not found or stale. Simulating Wolf2000 via RoadRunner...")
    rr = _load_roadrunner(cfg["sbml_file"])
    species = rr.getFloatingSpeciesIds()
    glucose_id = None
    for cand in ['s1', 'S1', 'glucose', 'Glucose', 'S_1']:
        if cand in species:
            glucose_id = cand
            break
    if glucose_id is None:
        glucose_id = species[0]
    glucose_vals = np.linspace(*cfg["glucose_range"], WOLF_N_SIMS)
    all_data = []
    for sim_id, glc in enumerate(glucose_vals):
        rr.reset()
        try:
            rr.setInitConcentration(glucose_id, glc)
        except AttributeError:
            rr.setValue('init(' + glucose_id + ')', glc)
        sim_data = rr.simulate(0, cfg["t_max"], cfg["n_points"] - 1)
        df = pd.DataFrame(sim_data, columns=['time'] + list(species))
        df.insert(0, 'sim_id', sim_id)
        all_data.append(df)
    df_all = pd.concat(all_data, ignore_index=True)
    _save_cache(csv_file, fingerprint, df_all)
    print(f"Saved Wolf trajectories to {csv_file}")
    return df_all


def simulate_bier(cfg):
    """Bier2000 two-cell glycolytic model: BIOMD0000000254's own rate rules
    (glucose G1/G2 and ATP T1/T2, coupled by ATP diffusion), integrated with
    scipy.odeint."""
    csv_file = cfg["csv_file"]
    fingerprint = _params_fingerprint({
        "species": cfg["species"],
        "init_conditions": cfg["init_conditions"],
        "params": cfg["params"],
        "t_max": cfg["t_max"],
        "n_points": cfg["n_points"],
    })
    cached = _load_cached(csv_file, fingerprint)
    if cached is not None:
        return cached
    print("CSV not found or stale. Simulating Bier2000 manually...")
    from scipy.integrate import odeint
    species = cfg["species"]
    p = cfg["params"]

    def bier_odes(y, t, p):
        G1, T1, G2, T2 = y
        V_in, k1, kp, km, epsilon = p['V_in'], p['k1'], p['kp'], p['km'], p['epsilon']
        dG1 = V_in - k1 * G1 * T1
        dG2 = V_in - k1 * G2 * T2
        dT1 = 2 * k1 * T1 * G1 - kp * T1 / (km + T1) + epsilon * (T2 - T1)
        dT2 = 2 * k1 * G2 * T2 - kp * T2 / (km + T2) - epsilon * (T2 - T1)
        return [dG1, dT1, dG2, dT2]

    t = np.linspace(0, cfg["t_max"], cfg["n_points"])
    sol = odeint(bier_odes, cfg["init_conditions"], t, args=(p,))
    df = pd.DataFrame(sol, columns=species)
    df.insert(0, 'time', t)
    df.insert(0, 'sim_id', 0)
    _save_cache(csv_file, fingerprint, df)
    print(f"Saved Bier trajectories to {csv_file}")
    return df


# ----------------------------------------------------------------------
# Unified system registry (biology + chemistry, one source of truth)
# ----------------------------------------------------------------------
SYSTEMS = {
    # ---- biology: glycolytic oscillators (D_max=4) ----
    "wolf": {
        "field": "biology",
        "description": "Wolf2000 glycolytic oscillator (BIOMD0000000206)",
        "D_max": 4,
        "simulate": simulate_wolf,
        "sbml_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000206?filename=BIOMD0000000206_url.xml",
        "sbml_file": os.path.join(GLYCO_DIR, "BIOMD0000000206.sbml"),
        "csv_file": os.path.join(GLYCO_DIR, "wolf2000_trajectories.csv"),
        "glucose_range": (1.0, 5.0),
        "t_max": 200,
        "n_points": WOLF_TIME_POINTS,
    },
    "bier": {
        "field": "biology",
        "description": "Bier2000 two-cell glycolysis (BIOMD0000000254)",
        "D_max": 4,
        "simulate": simulate_bier,
        "sbml_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000254?filename=BIOMD0000000254_url.xml",
        "sbml_file": os.path.join(GLYCO_DIR, "BIOMD0000000254.sbml"),
        "csv_file": os.path.join(GLYCO_DIR, "bier2000_trajectories.csv"),
        "species": ["G1", "T1", "G2", "T2"],
        "init_conditions": [6.6, 7.6, 10.3, 0.41],
        "params": {'V_in': 0.36, 'k1': 0.02, 'kp': 6.0, 'km': 13.0, 'epsilon': 0.01},
        "t_max": 500,
        "n_points": 1000,
    },
    # ---- chemistry: CRN signaling / cell-cycle models (D_max=3) ----
    "gardner": {
        "field": "chemistry",
        "description": "Gardner1998 cell cycle (BIOMD0000000008)",
        "D_max": 3,
        "simulate": simulate_crn,
        "sbml_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000008?filename=BIOMD0000000008_url.xml",
        "sbml_file": os.path.join(CRN_DIR, "BIOMD0000000008.sbml"),
        "csv_file": os.path.join(CRN_DIR, "gardner_trajectories.csv"),
    },
    "kholodenko": {
        "field": "chemistry",
        "description": "Kholodenko2000 MAPK cascade (BIOMD0000000010)",
        "D_max": 3,
        "simulate": simulate_crn,
        "sbml_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000010?filename=BIOMD0000000010_url.xml",
        "sbml_file": os.path.join(CRN_DIR, "BIOMD0000000010.sbml"),
        "csv_file": os.path.join(CRN_DIR, "kholodenko_trajectories.csv"),
    },
    "markevich": {
        "field": "chemistry",
        "description": "Markevich2004 MAPK cascade (BIOMD0000000028)",
        "D_max": 3,
        "simulate": simulate_crn,
        "sbml_url": "https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000028?filename=BIOMD0000000028_url.xml",
        "sbml_file": os.path.join(CRN_DIR, "BIOMD0000000028.sbml"),
        "csv_file": os.path.join(CRN_DIR, "markevich_trajectories.csv"),
    },
}


def prepare_design_matrix(df):
    keep_cols = [c for c in df.columns if c not in ("sim_id", "time")]
    return df[keep_cols].values, keep_cols


def ensure_trajectories(cfg):
    """Return the pooled trajectory dataframe, simulating only if the cached
    CSV is absent or stale relative to the simulation parameters. Fetches the
    SBML file first whenever it isn't already on disk, independent of whether
    the CSV cache turns out to be fresh; download_sbml is a no-op if the file
    is already present, so a fully-cached run (SBML and CSV both present)
    stays offline."""
    if not os.path.exists(cfg["sbml_file"]) and cfg.get("sbml_url") and cfg["simulate"] is not simulate_bier:
        download_sbml(cfg["sbml_url"], cfg["sbml_file"])
    return cfg["simulate"](cfg)


def run_srgb(X, var_names, D_max):
    print(f"  SR-GB+CSNP (N={X.shape[0]}, vars={len(var_names)}, D_max={D_max})")
    return sr_gb(X, var_names, degree=None, D_max=D_max,
                 sigma_estimate=0.0, use_bb=True, k_max=6)


def run_baseline(csv_path, d_max, method):
    """Run one baseline in an isolated subprocess with a wall-clock timeout."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    cmd = [sys.executable, WORKER, csv_path, str(d_max), method, out_json]
    try:
        proc = subprocess.run(cmd, timeout=BASELINE_TIMEOUT_S,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return {"elapsed": BASELINE_TIMEOUT_S, "rows": [],
                    "error": f"nonzero exit: {proc.stderr[-500:]}"}
        with open(out_json) as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        return {"elapsed": BASELINE_TIMEOUT_S, "rows": [], "error": "timeout"}
    finally:
        if os.path.exists(out_json):
            os.remove(out_json)


def process_system(name, cfg, run_baselines=True):
    print("\n" + "=" * 70)
    print(f"{name.upper()}  [{cfg['field']}]  {cfg['description']}")
    print("=" * 70)
    rows = []
    df = ensure_trajectories(cfg)
    X, var_names = prepare_design_matrix(df)
    print(f"  data: {X.shape[0]} rows x {X.shape[1]} vars {var_names}")

    # --- SR-GB+CSNP ---
    try:
        gb = run_srgb(X, var_names, cfg["D_max"])
    except Exception as e:
        gb = None
        print(f"  SR-GB+CSNP error: {e}")
    if not gb:
        print("  SR-GB+CSNP: no invariant found (abstained)")
        rows.append({"system": name, "field": cfg["field"], "method": "sr_gb_csnp",
                     "generator_idx": None, "expression": None, "status": "abstained"})
    else:
        print(f"  SR-GB+CSNP: {len(gb)} generator(s)")
        for i, p in enumerate(gb):
            expr = p.as_expr() if hasattr(p, "as_expr") else p
            print(f"    {i + 1}: {expr}")
            rows.append({"system": name, "field": cfg["field"], "method": "sr_gb_csnp",
                         "generator_idx": i, "expression": str(expr), "status": "returned"})

    # --- baselines (subprocess-isolated) ---
    if run_baselines:
        for method in BASELINES:
            out = run_baseline(cfg["csv_file"], cfg["D_max"], method)
            if out.get("error"):
                print(f"  {method}: {out['error']} ({out['elapsed']:.1f}s)")
                rows.append({"system": name, "field": cfg["field"], "method": method,
                             "generator_idx": None, "expression": None,
                             "status": out["error"], "elapsed_s": out["elapsed"]})
                continue
            result_rows = out["rows"]
            if not result_rows or result_rows[0]["status"] == "abstained":
                print(f"  {method}: abstained ({out['elapsed']:.1f}s)")
                rows.append({"system": name, "field": cfg["field"], "method": method,
                             "generator_idx": None, "expression": None,
                             "status": "abstained", "elapsed_s": out["elapsed"]})
                continue
            print(f"  {method}: {len(result_rows)} generator(s) ({out['elapsed']:.1f}s)")
            for r in result_rows:
                print(f"    {r['generator_idx'] + 1}: {r['expression']}")
                rows.append({"system": name, "field": cfg["field"], "method": method,
                             **r, "elapsed_s": out["elapsed"]})
    return rows


def main():
    args = [a for a in sys.argv[1:]]
    run_baselines = "--srgb-only" not in args
    args = [a for a in args if a != "--srgb-only"]

    if args and args[0] == "--quick":
        requested = [next(iter(SYSTEMS))]
    elif args and not args[0].startswith("-"):
        requested = [args[0].lower()]
    else:
        requested = list(SYSTEMS.keys())

    all_rows = []
    for name in requested:
        if name not in SYSTEMS:
            print(f"Unknown system '{name}'. Choose from: {list(SYSTEMS.keys())}")
            sys.exit(1)
        all_rows.extend(process_system(name, SYSTEMS[name], run_baselines=run_baselines))

    out_file = os.path.join(RESULTS_DIR, "bio_chem_attempts.csv")
    pd.DataFrame(all_rows).to_csv(out_file, index=False)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
