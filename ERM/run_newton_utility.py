"""Run the Newton utility sweep configured in ERM_utility.ipynb.

Set the privacy and feature-bound parameters in the configuration block below, then run:
    python run_newton_utility.py

Each run gets a timestamped, parameterized folder under results/utility/
containing its CSV, plots, and settings.json.
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

from active_utility_rs import run_experiment_newton
from load_data import load_datasets
from plot_utility import plot_utility


PROJECT_DIR = Path(__file__).resolve().parent

# Keep these settings aligned with the Newton section of ERM_utility.ipynb.
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
LAMBDA_VALUES = np.logspace(-6, -1, 25).tolist()
NUM_TRIALS = 5
RANDOM_SEED = 42
LOCAL_RADIUS_FACTOR = 2.0
OUTPUT_DIR = Path("results/utility")

# Privacy and feature-bound settings for this run (as in run_d2d_utility.py).
EPSILON = 1.0
DELTA = 1e-9
B_X = 1.0


def _scientific_tag(value: float) -> str:
    """Compact filename form, e.g. 1e-06 -> 1e-6."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _create_run_directory(output_root: Path) -> Path:
    dimensions = "-".join(str(spec["specs"][2]) for spec in DATASETS.values())
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / (
        f"run_{run_stamp}_newton_n{max(N_VALUES)}_rp{dimensions}"
        f"_lambda{_scientific_tag(min(LAMBDA_VALUES))}"
        f"-to-{_scientific_tag(max(LAMBDA_VALUES))}_k{len(LAMBDA_VALUES)}"
        f"_trials{NUM_TRIALS}_eps{_scientific_tag(EPSILON)}"
        f"_delta{_scientific_tag(DELTA)}_bx{B_X:g}_r{LOCAL_RADIUS_FACTOR:g}"
        f"_seed{RANDOM_SEED}"
    )
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Parent directory for timestamped run folders (default: results/utility).",
    )
    args = parser.parse_args()
    for name, value in (("epsilon", EPSILON), ("B_x", B_X)):
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.upper()} must be finite and positive")
    if not np.isfinite(DELTA) or not 0 < DELTA < 1:
        parser.error("DELTA must be finite and in (0, 1)")

    # The data loader uses ./data for MNIST, so make the script independent of
    # the directory from which it was launched.
    os.chdir(PROJECT_DIR)
    output_root = args.output_dir
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _create_run_directory(output_root)
    output_path = run_dir / "utility_newton.csv"

    print("Newton utility sweep settings:")
    print(f"  datasets: {', '.join(DATASETS)}")
    print(f"  n: {N_VALUES}; trials per setting: {NUM_TRIALS}")
    print(
        f"  lambda: {LAMBDA_VALUES[0]:.1e} to {LAMBDA_VALUES[-1]:.1e} "
        f"({len(LAMBDA_VALUES)} values)"
    )
    print(f"  epsilon: {EPSILON}; delta: {DELTA}; B_x: {B_X}; base seed: {RANDOM_SEED}")
    print(f"  run directory: {run_dir}", flush=True)

    started = time.perf_counter()
    datasets = load_datasets(DATASETS)
    results = run_experiment_newton(
        datasets,
        N_VALUES,
        LAMBDA_VALUES,
        num_trials=NUM_TRIALS,
        random_seed=RANDOM_SEED,
        local_radius_factor=LOCAL_RADIUS_FACTOR,
        epsilon=EPSILON,
        delta=DELTA,
        B_x=B_X,
    )
    # Include settings fixed in the runner so each result file is self-describing.
    results["epsilon"] = EPSILON
    results["delta"] = DELTA
    results["local_radius_factor"] = LOCAL_RADIUS_FACTOR
    results["B_x"] = B_X
    results["random_seed_base"] = RANDOM_SEED
    results["trial_seed"] = RANDOM_SEED + results["trial"].astype(int)
    results.to_csv(output_path, index=False)
    print(f"Saved {len(results)} rows to {output_path}", flush=True)

    plot_paths = plot_utility(
        output_path,
        utility_dir=output_path.parent,
        out_dir=output_path.parent,
        show=False,
    )
    for metric, path in plot_paths.items():
        print(f"Saved {metric} plot to {path}", flush=True)

    settings = {
        "method": "newton", "datasets": list(DATASETS), "n_values": N_VALUES,
        "lambda_values": LAMBDA_VALUES, "num_trials": NUM_TRIALS,
        "random_seed": RANDOM_SEED, "local_radius_factor": LOCAL_RADIUS_FACTOR,
        "epsilon": EPSILON, "delta": DELTA, "B_x": B_X,
    }
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Saved run settings to {run_dir / 'settings.json'}", flush=True)

    elapsed = time.perf_counter() - started
    print(f"Elapsed time: {elapsed / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
