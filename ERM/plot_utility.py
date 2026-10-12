from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_UTILITY_DIR = Path("results/utility")
DEFAULT_PLOT_DIR = Path("results/plots/utility")


def dataset_styles():
    """Styles for the MNIST random-projection datasets."""
    return {
        "mnist_rp15": {"color": "#ffb71c", "marker": "o", "label": "MNIST (15)"},
        "mnist_rp20": {"color": "#b33535", "marker": "s", "label": "MNIST (20)"},
        "mnist_rp25": {"color": "#09a000", "marker": "D", "label": "MNIST (25)"},
        "mnist_rp40": {"color": "#9467bd", "marker": "v", "label": "MNIST (40)"},
        "mnist_rp50": {"color": "#1d56b8", "marker": "X", "label": "MNIST (50)"},
        "mnist_rp75": {"color": "#9467bd", "marker": "v", "label": "MNIST (75)"},
        "mnist_rp100": {"color": "#8c564b", "marker": "P", "label": "MNIST (100)"},
        "mnist_rp_none": {"color": "#e377c2", "marker": "*", "label": "MNIST (raw)"},
    }


def _resolve_csv(csv_file, utility_dir):
    csv_path = Path(csv_file)
    if not csv_path.is_absolute() and csv_path.parent == Path("."):
        csv_path = Path(utility_dir) / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"Utility CSV not found: {csv_path}")
    return csv_path


def _dataset_sort_key(dataset):
    try:
        return int(str(dataset).rsplit("rp", 1)[1])
    except (IndexError, ValueError):
        return float("inf")


def _plot_metric(
    df,
    *,
    metric,
    metric_label,
    x_column,
    x_label,
    log_x,
    csv_stem,
    filename_suffix,
    metric_first,
    out_dir,
    y_limits,
    dpi,
    show,
):
    styles = dataset_styles()
    datasets = sorted(df["dataset"].unique(), key=_dataset_sort_key)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "^", "v", "X", "P"]

    fig, ax = plt.subplots(figsize=(5, 4))
    dataset_handles = []

    for index, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        style = styles.get(
            dataset,
            {
                "color": color_cycle[index % len(color_cycle)],
                "marker": markers[index % len(markers)],
                "label": str(dataset),
            },
        )

        metric_summary = (
            subset.groupby(x_column)[metric]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(x_column)
        )
        retrain_summary = (
            subset.groupby(x_column)["acc_Retrain"]
            .mean()
            .reset_index()
            .sort_values(x_column)
        )

        xs = metric_summary[x_column].to_numpy(dtype=float)
        means = metric_summary["mean"].to_numpy(dtype=float)
        stds = metric_summary["std"].fillna(0.0).to_numpy(dtype=float)
        mark_every = max(1, len(xs) // 8)

        ax.plot(
            xs,
            means,
            color=style["color"],
            marker=style["marker"],
            markevery=mark_every,
            markersize=12,
            linewidth=4,
        )
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            color=style["color"],
            alpha=0.08,
        )
        ax.plot(
            retrain_summary[x_column],
            retrain_summary["acc_Retrain"],
            color=style["color"],
            linestyle=":",
            linewidth=4,
            alpha=0.6,
        )

        dataset_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.5,
                label=style["label"],
            )
        )

    dataset_legend = ax.legend(
        handles=dataset_handles,
        title="Dataset (dim)",
        loc="lower right",
        bbox_to_anchor=(1.0, 0.28),
        fontsize=8,
        title_fontsize=10,
    )
    ax.add_artist(dataset_legend)
    ax.legend(
        handles=[
            Line2D([0], [0], color="black", linestyle=":", linewidth=2.5, label="Retrain"),
            Line2D([0], [0], color="black", marker="o", linewidth=2.5, label=metric_label),
        ],
        title="Classifier",
        loc="lower right",
        fontsize=12,
        title_fontsize=13,
    )

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel("Test Accuracy", fontsize=14)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    fig.tight_layout()

    out_name = (
        f"{metric}_{filename_suffix}_{csv_stem}.png"
        if metric_first
        else f"{csv_stem}_{metric}_{filename_suffix}.png"
    )
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_utility(
    csv_file,
    *,
    lambda_value=0.1,
    utility_dir=DEFAULT_UTILITY_DIR,
    out_dir=DEFAULT_PLOT_DIR,
    y_limits=None,
    dpi=220,
    show=True,
):
    """Plot RS and GS accuracy against retraining for a utility CSV.

    Files containing ``d2d`` are filtered to ``lambda_value`` and plotted
    against iterations. Files containing ``newton`` are plotted against all
    lambda values on a logarithmic x-axis. A bare filename is loaded from
    ``results/utility`` and plots are saved to ``results/plots/utility``.
    """
    csv_path = _resolve_csv(csv_file, utility_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    common_required = {"dataset", "acc_Retrain", "acc_GS", "acc_RS"}
    missing = sorted(common_required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {missing}")

    lambda_column = "lambd" if "lambd" in df.columns else "lambda"
    if lambda_column not in df.columns:
        raise ValueError(f"{csv_path} is missing a lambda column")

    filename = csv_path.name.lower()
    if "newton" in filename:
        selected = df
        x_column = lambda_column
        x_label = "Lambda"
        log_x = True
        filename_suffix = "by_lambda"
        effective_y_limits = (0.4, 0.82) if y_limits is None else y_limits
    elif "d2d" in filename:
        if "iterations" not in df.columns:
            raise ValueError(f"{csv_path} is missing column: iterations")
        selected = df[np.isclose(df[lambda_column], lambda_value)].copy()
        if selected.empty:
            available = sorted(df[lambda_column].dropna().unique())
            raise ValueError(
                f"No rows found for lambda={lambda_value}. Available values: {available}"
            )
        x_column = "iterations"
        x_label = "Iterations I"
        log_x = False
        filename_suffix = f"lambda_{lambda_value:g}"
        effective_y_limits = (0.4, 0.8) if y_limits is None else y_limits
    else:
        raise ValueError(
            f"Cannot determine utility method from {csv_path.name!r}; "
            "the filename must contain 'newton' or 'd2d'"
        )

    plot_specs = (
        ("acc_RS", "Retain Sensitivity"),
        ("acc_GS", "Global Sensitivity"),
    )
    return {
        metric: _plot_metric(
            selected,
            metric=metric,
            metric_label=label,
            x_column=x_column,
            x_label=x_label,
            log_x=log_x,
            csv_stem=csv_path.stem,
            filename_suffix=filename_suffix,
            metric_first="d2d" in filename,
            out_dir=out_dir,
            y_limits=effective_y_limits,
            dpi=dpi,
            show=show,
        )
        for metric, label in plot_specs
    }
