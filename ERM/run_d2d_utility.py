"""Run the D2D utility sweep configured in ERM_utility.ipynb.

Set the privacy and norm parameters in the configuration block below, then run:
    ../../.venv/bin/python run_d2d_utility.py

Each run gets a timestamped, parameterized folder under results/utility/
containing its CSVs, plots, and settings.json.
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

from active_utility_rs import run_experiment_d2d
from load_data import load_datasets
from plot_utility import plot_utility


PROJECT_DIR = Path(__file__).resolve().parent

# Match the D2D section of ERM_utility.ipynb, using six decade-spaced lambdas.
DATASETS = {
    "mnist_rp15": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 15),
        "standardize_before_projection": True,
    },
    "mnist_rp20": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 20),
        "standardize_before_projection": True,
    },
    "mnist_rp25": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 25),
        "standardize_before_projection": True,
    },
    "mnist_rp50": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 50),
        "standardize_before_projection": True,
    },
}
N_VALUES = [10_000]
LAMBDA_VALUES = np.logspace(-6, -1, 6).tolist()  # 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1
NUM_TRIALS = 5
RANDOM_SEED = 42
ITERATIONS = np.linspace(1, 1000, 200, dtype=int).tolist()
LOSS_NAMES = ("LogLoss", "MSE")
OUTPUT_DIR = Path("results/utility")

# Privacy and norm settings for this run
EPSILON = 1.0
DELTA = 1e-9
B_X = 1.0
R_W = 1.0


def _scientific_tag(value: float) -> str:
    """Compact filename form, e.g. 1e-06 -> 1e-6."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _default_run_stem(epsilon: float, delta: float, B_x: float, R_w: float, num_trials: int) -> str:
    dimensions = "-".join(str(spec["specs"][2]) for spec in DATASETS.values())
    return (
        f"utility_d2d_n{max(N_VALUES)}_rp{dimensions}"
        f"_trials{num_trials}_eps{_scientific_tag(epsilon)}"
        f"_delta{_scientific_tag(delta)}_bx{B_x:g}_rw{R_w:g}_seed{RANDOM_SEED}"
    )


def _run_directory(output_root: Path, num_trials: int) -> Path:
    dimensions = "-".join(str(spec["specs"][2]) for spec in DATASETS.values())
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / (
        f"run_{stamp}_d2d_n{max(N_VALUES)}_rp{dimensions}"
        f"_trials{num_trials}_eps{_scientific_tag(EPSILON)}"
        f"_delta{_scientific_tag(DELTA)}_bx{B_X:g}_rw{R_W:g}_seed{RANDOM_SEED}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-trials", type=int, default=NUM_TRIALS, help=f"Trials per setting (default: {NUM_TRIALS}).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSVs and plots (default: results/utility).",
    )
    args = parser.parse_args()
    for name, value in (("epsilon", EPSILON), ("B_x", B_X), ("R_w", R_W)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not np.isfinite(DELTA) or not 0 < DELTA < 1:
        parser.error("--delta must be finite and in (0, 1)")
    if args.num_trials < 1:
        parser.error("--num-trials must be at least 1")

    # The data loader uses ./data for MNIST, so make the script independent of
    # the directory from which it was launched.
    os.chdir(PROJECT_DIR)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _run_directory(output_dir, args.num_trials)
    run_stem = _default_run_stem(EPSILON, DELTA, B_X, R_W, args.num_trials)

    print("D2D utility sweep settings:")
    print(f"  datasets: {', '.join(DATASETS)}")
    print(f"  n: {N_VALUES}; trials per setting: {args.num_trials}")
    print(
        f"  lambda: {LAMBDA_VALUES[0]:.1e} to {LAMBDA_VALUES[-1]:.1e} "
        f"({len(LAMBDA_VALUES)} values)"
    )
    print(
        f"  iterations: {min(ITERATIONS)} to {max(ITERATIONS)} "
        f"({len(ITERATIONS)} values)"
    )
    print(f"  epsilon: {EPSILON}; delta: {DELTA}; base seed: {RANDOM_SEED}")
    print(f"  B_x: {B_X}; R_w: {R_W}")
    print(f"  run directory: {run_dir}", flush=True)

    started = time.perf_counter()
    datasets = load_datasets(DATASETS)
    for loss_name in LOSS_NAMES:
        results = run_experiment_d2d(
            datasets,
            N_VALUES,
            LAMBDA_VALUES,
            loss_name=loss_name,
            num_trials=args.num_trials,
            random_seed=RANDOM_SEED,
            iterations=ITERATIONS,
            epsilon=EPSILON,
            delta=DELTA,
            B_x=B_X,
            R_w=R_W,
        )
        # Store the runner configuration with every row for provenance.
        results["epsilon"] = EPSILON
        results["delta"] = DELTA
        results["B_x"] = B_X
        results["R_w"] = R_W
        results["random_seed_base"] = RANDOM_SEED
        results["trial_seed"] = RANDOM_SEED + results["trial"].astype(int)

        csv_path = run_dir / f"{run_stem}_{loss_name}.csv"
        results.to_csv(csv_path, index=False)
        print(f"Saved {len(results)} rows to {csv_path}", flush=True)

        for lambda_value in LAMBDA_VALUES:
            plot_paths = plot_utility(
                csv_path,
                lambda_value=lambda_value,
                utility_dir=run_dir,
                out_dir=run_dir,
                show=False,
            )
            for metric, path in plot_paths.items():
                print(
                    f"Saved {metric} plot for lambda={lambda_value:g} to {path}",
                    flush=True,
                )

    settings = {
        "method": "d2d", "datasets": list(DATASETS), "n_values": N_VALUES,
        "lambda_values": LAMBDA_VALUES, "num_trials": args.num_trials,
        "random_seed": RANDOM_SEED, "iterations": ITERATIONS,
        "epsilon": EPSILON, "delta": DELTA, "B_x": B_X, "R_w": R_W,
        "losses": list(LOSS_NAMES),
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved run settings to {run_dir / 'settings.json'}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Elapsed time: {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
