"""Recompute SVM ratio experiments with editable settings.

Run from the repository root with:
    .venv/bin/python SVM/run_svm_experiments.py

Each run gets a new timestamped directory under SVM/results/ratios, with its
plots stored alongside the CSVs in a plots/ subdirectory. Plot styling is
shared with ERM ratio plots through SVM.plotting / ERM.plot_ratios.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir

# ---------------------------------------------------------------------------
# Edit these settings to choose datasets and experiment parameters.
# ---------------------------------------------------------------------------
DATASET_NAMES = ("wine", "mnist_rp50", "folktables")
KERNELS = ("linear", "rbf")
C = 1e6
GAMMA = None
VERIFY_TOL = 1e-3
MAX_ITER = 20_000
TOL = 1e-4
CACHE_SIZE_MB = 1024
N_JOBS = 1
TRIALS = 25
MIN_FRAC = 0.1
MIN_SIZE = None
NUM_SIZES = 10
MAX_FRAC = 0.6
REAL_FRAC = 1.0
REAL_SIZE = 0
REAL_ATTEMPTS = 25
REAL_SHRINK = 0.9
SEED = 0
MAX_N = 20_000
SHOW_PLOTS = False

# Use SVM_old experiment parameters with the ERM loaders and preprocessing.
DATASETS = {
    "wine": {"X": None, "y_bin": None, "specs": "red"},
    "mnist_rp50": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 50),
        "standardize_before_projection": True,
    },
    "folktables": {
        "X": None,
        "y_bin": None,
        "specs": ("CA", 2018, "person"),
    },
}
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parents[1]
SVM_DIR = REPO_DIR / "SVM"
ERM_DIR = REPO_DIR / "ERM"
RESULTS_DIR = SVM_DIR / "results" / "ratios"

# Set plotting configuration before importing the SVM plotting modules.
MPLCONFIG_DIR = Path(gettempdir()) / "retain_sensitivity_svm_mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import matplotlib

matplotlib.use("Agg")

from ERM.load_data import load_datasets
from SVM.experiment import run_experiments


def _safe_number(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    unknown = sorted(set(DATASET_NAMES) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {sorted(DATASETS)}")

    dataset_tag = "-".join(DATASET_NAMES)
    run_name = (
        f"run_{timestamp}_{dataset_tag}_C{_safe_number(C)}"
        f"_t{TRIALS}_n{MAX_N}_s{SEED}"
    )
    run_dir = RESULTS_DIR / run_name
    ratios_dir = run_dir / "csv"
    plots_dir = run_dir / "plots"
    ratios_dir.mkdir(parents=True, exist_ok=False)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # The shared ERM loaders use paths relative to ERM/, so load from there.
    os.chdir(ERM_DIR)
    dataset_specs = {name: DATASETS[name].copy() for name in DATASET_NAMES}
    datasets = load_datasets(dataset_specs)

    settings = {
        "datasets": list(DATASET_NAMES),
        "kernels": list(KERNELS),
        "C": C,
        "gamma": GAMMA,
        "verify_tol": VERIFY_TOL,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "cache_size_mb": CACHE_SIZE_MB,
        "n_jobs": N_JOBS,
        "trials": TRIALS,
        "min_frac": MIN_FRAC,
        "min_size": MIN_SIZE,
        "num_sizes": NUM_SIZES,
        "max_frac": MAX_FRAC,
        "real_frac": REAL_FRAC,
        "real_size": REAL_SIZE,
        "real_attempts": REAL_ATTEMPTS,
        "real_shrink": REAL_SHRINK,
        "seed": SEED,
        "max_n": MAX_N,
        "show_plots": SHOW_PLOTS,
        "plot_style": "shared ERM ratio plotting style",
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    outputs = run_experiments(
        datasets,
        ratios_dir=ratios_dir,
        plots_dir=plots_dir,
        kernels=KERNELS,
        C=C,
        gamma=GAMMA,
        verify_tol=VERIFY_TOL,
        max_iter=MAX_ITER,
        tol=TOL,
        cache_size_mb=CACHE_SIZE_MB,
        n_jobs=N_JOBS,
        trials=TRIALS,
        min_frac=MIN_FRAC,
        min_size=MIN_SIZE,
        num_sizes=NUM_SIZES,
        max_frac=MAX_FRAC,
        real_frac=REAL_FRAC,
        real_size=REAL_SIZE,
        real_attempts=REAL_ATTEMPTS,
        real_shrink=REAL_SHRINK,
        seed=SEED,
        max_n=MAX_N,
        show_plots=SHOW_PLOTS,
    )

    print(f"Run outputs: {run_dir}")
    for path in outputs["csv_paths"] + outputs["plot_paths"]:
        print(path)


if __name__ == "__main__":
    main()
