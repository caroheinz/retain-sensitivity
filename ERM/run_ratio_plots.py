"""Recalculate ERM sensitivity ratios and regenerate their plots.

Run from this directory with:
    ../../.venv/bin/python run_ratio_plots.py

Raw runs and ratio CSVs are saved in a parameterized subfolder of
results/ratios/. The combined plots are saved directly in results/ratios/.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render plots on headless cluster nodes.
import numpy as np

from compute_rs import run_experiments_on_datasets
from load_data import export_ratio_csvs, load_datasets
from plot_ratios import plot_available_ratios


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = Path("results/ratios")
DATASETS = {
    "wine": {"X": None, "y_bin": None, "specs": "red"},
    "mnist_rp50": {
        "X": None, "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 50),
        "standardize_before_projection": True,
    },
    "folktables": {
        "X": None, "y_bin": None, "specs": ("CA", 2018, "person")
    },
}
N_VALUES = [10_000]
LAMBDA_VALUES = np.logspace(-6, -1, 25).tolist()
SEEDS = [42,43,44,45,46]
PASSIVE_LOSSES = ("MSE", "LogLoss")
ACTIVE_LOSSES = ("MSE_d2d", "LogLoss_d2d", "LogLoss_newton")

# Edit these values to match the desired theory/experiment setting.
B_X = 1.0
R_W = 1.0
EPSILON = 1.0
DELTA = 1e-9
SIGMA = 0.1


def _validate_settings(parser: argparse.ArgumentParser) -> None:
    for name, value in (("B_X", B_X), ("R_W", R_W), ("EPSILON", EPSILON), ("SIGMA", SIGMA)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name} must be finite and positive")
    if not np.isfinite(DELTA) or not 0 < DELTA < 1:
        parser.error("DELTA must lie in (0, 1)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-trials", type=int, default=len(SEEDS),
        help=f"Trials per setting (default: {len(SEEDS)}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="Directory for run data and plots (default: results/ratios).",
    )
    args = parser.parse_args()
    _validate_settings(parser)
    if args.num_trials < 1 or args.num_trials > len(SEEDS):
        parser.error(f"--num-trials must be between 1 and {len(SEEDS)}")

    os.chdir(PROJECT_DIR)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / (
        f"run_{stamp}_n{'-'.join(map(str, N_VALUES))}"
        f"_bx{B_X:g}_rw{R_W:g}_eps{EPSILON:g}_delta{DELTA:g}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    seeds = SEEDS[:args.num_trials]

    print("ERM ratio recalculation settings:")
    print(f"  datasets: {', '.join(DATASETS)}")
    print(f"  n: {N_VALUES}; lambda: {LAMBDA_VALUES}; seeds: {seeds}")
    print(f"  B_x: {B_X}; R_w: {R_W}; epsilon: {EPSILON}; delta: {DELTA}; sigma: {SIGMA}")
    print(f"  raw and ratio CSV directory: {run_dir}")
    print(f"  plot directory: {output_dir}", flush=True)

    started = time.perf_counter()
    datasets = load_datasets(DATASETS)
    passive_results = run_experiments_on_datasets(
        n_list=N_VALUES, lambda_list=LAMBDA_VALUES, loss_list=PASSIVE_LOSSES,
        seeds=seeds, datasets=datasets, B_x=B_X, R_w=R_W, passive=True,
    )
    active_results = run_experiments_on_datasets(
        n_list=N_VALUES, lambda_list=LAMBDA_VALUES, loss_list=ACTIVE_LOSSES,
        seeds=seeds, datasets=datasets, B_x=B_X, R_w=R_W, passive=False,
        eps_list=[EPSILON], delta=DELTA, sigma=SIGMA,
    )

    passive_results.to_csv(run_dir / "passive_raw.csv", index=False)
    active_results.to_csv(run_dir / "active_raw.csv", index=False)
    passive_paths = export_ratio_csvs(
        passive_results, out_dir=run_dir, prefix="passive_erm"
    )
    active_paths = export_ratio_csvs(
        active_results, out_dir=run_dir, prefix="active_erm"
    )
    metadata = {
        "datasets": list(DATASETS), "n_values": N_VALUES,
        "lambda_values": LAMBDA_VALUES, "seeds": seeds,
        "B_x": B_X, "R_w": R_W, "epsilon": EPSILON,
        "delta": DELTA, "sigma": SIGMA,
        "newton_rs": "margin-aware local certificate; uncertified settings excluded from ratio summaries",
        "passive_erm": "fit with the configured R_w hard constraint",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    plots = plot_available_ratios(
        results_dir=run_dir, out_dir=run_dir, datasets=DATASETS.keys(),
        passive_losses=PASSIVE_LOSSES, active_losses=ACTIVE_LOSSES,
    )
    print(f"Saved {len(passive_paths) + len(active_paths)} ratio CSVs and raw runs to {run_dir}")
    print(f"Saved {len(plots)} plots:")
    for path in plots:
        print(path)
    print(f"Elapsed time: {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
