import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .datasets import (
    dataset_title,
    default_svm_dataset_specs,
    load_svm_datasets,
    svm_labels,
)
from .plotting import load_series, plot_combined
from .svm import fit_svm_and_stats
from .utils import (
    compute_min_size_from_frac,
    ensure_writable_dirs,
    linspace_sizes,
    stratified_subsample_indices,
)


def run_dataset(
    dataset: str,
    datasets,
    *,
    ratios_dir: Path,
    kernels: Sequence[str],
    C: float,
    gamma: Optional[float],
    verify_tol: float,
    max_iter: int,
    tol: float,
    cache_size_mb: int,
    n_jobs: int,
    trials: int,
    min_frac: float,
    min_size: Optional[int],
    num_sizes: int,
    max_frac: float,
    real_frac: float,
    real_size: int,
    real_attempts: int,
    real_shrink: float,
    seed: int,
    max_n: int,
):
    """Calculate SVM margin-ratio CSVs for one preloaded ERM dataset."""
    if dataset not in datasets:
        raise ValueError(f"Dataset {dataset!r} was not loaded")

    started = time.time()
    print(f"[{dataset}] start")
    X = np.asarray(datasets[dataset]["X"], dtype=float)
    y = svm_labels(datasets[dataset]["y_bin"])
    title = dataset_title(dataset)

    if max_n and len(X) > int(max_n):
        cap_rng = np.random.default_rng(seed + 4242)
        indices = cap_rng.choice(len(X), size=int(max_n), replace=False)
        X, y = X[indices], y[indices]

    rng = np.random.default_rng(seed)
    n_total = len(X)
    print(
        f"[{dataset}] prepared data: n={n_total} d={X.shape[1]} "
        f"kernels={','.join(kernels)}"
    )

    target_ref = int(real_size) if int(real_size) > 0 else int(round(real_frac * n_total))
    target_ref = max(2, min(target_ref, n_total))
    if dataset == "folktables" and real_size == 0 and real_frac == 1.0:
        target_ref = min(target_ref, 5000)

    ref_indices = None
    reference_stats = {}
    candidate_size = target_ref
    while candidate_size >= 2 and ref_indices is None:
        for _ in range(max(1, int(real_attempts))):
            candidate = stratified_subsample_indices(y, candidate_size, rng)
            if candidate is None:
                continue

            per_kernel = {}
            for kernel in kernels:
                w_norm_sq, accuracy, min_yf = fit_svm_and_stats(
                    X[candidate],
                    y[candidate],
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    tol=tol,
                    max_iter=max_iter,
                    cache_size_mb=cache_size_mb,
                )
                is_separable = (
                    accuracy >= 1.0 - 1e-12
                    and min_yf >= 1.0 - verify_tol
                    and w_norm_sq > 0
                )
                if not is_separable:
                    break
                per_kernel[kernel] = (
                    float(w_norm_sq),
                    1.0 / math.sqrt(float(w_norm_sq)),
                )

            if len(per_kernel) == len(kernels):
                ref_indices = candidate
                reference_stats = per_kernel
                break

        if ref_indices is None:
            candidate_size = int(math.floor(candidate_size * real_shrink))

    if ref_indices is None:
        print(f"[{dataset}] no separable reference subset found; skipping")
        return []

    X_ref = X[ref_indices]
    y_ref = y[ref_indices]
    n_ref = len(X_ref)
    print(f"[{dataset}] reference subset: n_ref={n_ref}/{n_total}")

    max_size = max(2, min(int(math.floor(max_frac * n_ref)), n_ref))
    if min_size is None:
        first_size = compute_min_size_from_frac(min_frac, n_ref, max_size)
    else:
        first_size = max(2, min(int(min_size), max_size))
    sizes = linspace_sizes(first_size, max_size, num_sizes)

    subsample_rng = np.random.default_rng(seed + 12345)
    subsamples = {
        size: [
            stratified_subsample_indices(y_ref, size, subsample_rng)
            for _ in range(trials)
        ]
        for size in sizes
    }

    outputs = []
    for kernel in kernels:
        print(f"[{dataset}|{kernel}] calculating {len(sizes)} sizes")
        ref_w_norm_sq, real_margin = reference_stats[kernel]
        inverse_real_margin = math.sqrt(ref_w_norm_sq)
        rows = []

        def evaluate_one(indices):
            if indices is None:
                return None
            w_norm_sq, accuracy, min_yf = fit_svm_and_stats(
                X_ref[indices],
                y_ref[indices],
                kernel=kernel,
                C=C,
                gamma=gamma,
                tol=tol,
                max_iter=max_iter,
                cache_size_mb=cache_size_mb,
            )
            is_separable = (
                accuracy >= 1.0 - 1e-12
                and min_yf >= 1.0 - verify_tol
                and w_norm_sq > 0
            )
            return float(w_norm_sq) if is_separable else None

        for size in sizes:
            indices_list = subsamples[size]
            if n_jobs and int(n_jobs) != 1:
                from joblib import Parallel, delayed

                norm_squares = Parallel(n_jobs=int(n_jobs), prefer="threads")(
                    delayed(evaluate_one)(indices) for indices in indices_list
                )
            else:
                norm_squares = [evaluate_one(indices) for indices in indices_list]

            q_values = []
            ratio_values = []
            clipped = 0
            for w_norm_sq in norm_squares:
                if w_norm_sq is None:
                    continue
                difference = ref_w_norm_sq - w_norm_sq
                if difference < 0:
                    clipped += 1
                    difference = 0.0
                q_value = math.sqrt(difference)
                q_values.append(q_value)
                ratio_values.append(q_value / inverse_real_margin)

            q_array = np.asarray(q_values, dtype=float)
            ratio_array = np.asarray(ratio_values, dtype=float)

            def mean_std(values):
                if len(values) == 0:
                    return float("nan"), float("nan")
                std = values.std(ddof=1) if len(values) > 1 else 0.0
                return float(values.mean()), float(std)

            mean_q, std_q = mean_std(q_array)
            mean_ratio, std_ratio = mean_std(ratio_array)
            rows.append(
                {
                    "dataset": dataset,
                    "title": title,
                    "kernel": kernel,
                    "gamma": (
                        1.0 / X_ref.shape[1]
                        if kernel == "rbf" and gamma is None
                        else "" if kernel == "linear" else float(gamma)
                    ),
                    "C": C,
                    "n_total": n_total,
                    "n_ref": n_ref,
                    "size": size,
                    "frac": size / n_ref,
                    "real_margin": real_margin,
                    "real_w_norm_sq": ref_w_norm_sq,
                    "mean_q": mean_q,
                    "std_q": std_q,
                    "mean_ratio": mean_ratio,
                    "std_ratio": std_ratio,
                    "n_trials": trials,
                    "n_success": len(q_array),
                    "n_fail": trials - len(q_array),
                    "n_neg_clipped": clipped,
                }
            )
            print(
                f"[{dataset}|{kernel}] size={size}: "
                f"success={len(q_array)}/{trials}, ratio={mean_ratio:.3g}"
            )

        csv_path = ratios_dir / f"svm_margin_gap_{dataset}_{kernel}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs.append((kernel, dataset, csv_path))

    print(f"[{dataset}] done in {time.time() - started:.1f}s")
    return outputs


def run_experiments(
    datasets,
    *,
    ratios_dir=Path("results/ratios"),
    plots_dir=Path("results/plots/ratios"),
    kernels=("linear", "rbf"),
    C=1e6,
    gamma=None,
    verify_tol=1e-3,
    max_iter=20000,
    tol=1e-4,
    cache_size_mb=1024,
    n_jobs=1,
    trials=5,
    min_frac=0.1,
    min_size=None,
    num_sizes=8,
    max_frac=0.6,
    real_frac=1.0,
    real_size=0,
    real_attempts=25,
    real_shrink=0.9,
    seed=0,
    max_n=5000,
    show_plots=True,
):
    """Run all SVM calculations, export CSVs, then plot those CSVs."""
    ratios_dir = Path(ratios_dir)
    plots_dir = Path(plots_dir)
    ensure_writable_dirs(ratios_dir, plots_dir)

    kernels = tuple(kernels)
    invalid = sorted(set(kernels) - {"linear", "rbf"})
    if invalid:
        raise ValueError(f"Unsupported kernels: {invalid}")

    csv_records = []
    for dataset in datasets:
        csv_records.extend(
            run_dataset(
                dataset,
                datasets,
                ratios_dir=ratios_dir,
                kernels=kernels,
                C=float(C),
                gamma=gamma,
                verify_tol=float(verify_tol),
                max_iter=int(max_iter),
                tol=float(tol),
                cache_size_mb=int(cache_size_mb),
                n_jobs=int(n_jobs),
                trials=int(trials),
                min_frac=float(min_frac),
                min_size=min_size,
                num_sizes=int(num_sizes),
                max_frac=float(max_frac),
                real_frac=float(real_frac),
                real_size=int(real_size),
                real_attempts=int(real_attempts),
                real_shrink=float(real_shrink),
                seed=int(seed),
                max_n=int(max_n),
            )
        )

    plot_paths = []
    for kernel in kernels:
        kernel_records = [record for record in csv_records if record[0] == kernel]
        if not kernel_records:
            continue
        series = []
        csv_paths = []
        for _, dataset, csv_path in kernel_records:
            xs, ys, stds = load_series(csv_path)
            series.append((dataset, xs, ys, stds))
            csv_paths.append(csv_path)
        plot_paths.append(
            plot_combined(
                out_dir=plots_dir,
                kernel=kernel,
                series=series,
                csv_paths=csv_paths,
                log_x=False,
                log_y=False,
                out_name="svm_margin_ratio_all",
                x_label="Subsample fraction of reference set",
                y_label="RS / GS",
                show=show_plots,
            )
        )

    return {
        "csv_paths": [record[2] for record in csv_records],
        "plot_paths": plot_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="wine,mnist_rp50,folktables,wbc")
    parser.add_argument("--ratios-dir", default="results/ratios")
    parser.add_argument("--plots-dir", default="results/plots/ratios")
    parser.add_argument("--kernels", default="linear,rbf")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num-sizes", type=int, default=8)
    parser.add_argument("--max-n", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    specs = default_svm_dataset_specs(names)
    datasets = load_svm_datasets(specs)
    outputs = run_experiments(
        datasets,
        ratios_dir=args.ratios_dir,
        plots_dir=args.plots_dir,
        kernels=[kernel.strip() for kernel in args.kernels.split(",")],
        trials=args.trials,
        num_sizes=args.num_sizes,
        max_n=args.max_n,
        n_jobs=args.n_jobs,
    )
    for path in outputs["csv_paths"] + outputs["plot_paths"]:
        print(path)


if __name__ == "__main__":
    main()
