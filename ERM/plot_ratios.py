import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

def dataset_style():
    return {
        'wdbc': dict(color='#ff7f0e', marker='o', linestyle='-', label='WBC'),
        'wbc': dict(color='#ff7f0e', marker='o', linestyle='-', label='WBC'),
        'folktables': dict(color='#2ca02c', marker='D', linestyle='-', label='ACS CA'),
        'mnist_rp25': dict(color='#d62728', marker='X', linestyle='-', label='MNIST (25)'),
        'mnist_rp50': dict(color='#1d56b8', marker='X', linestyle='-', label='MNIST (50)'),
        'wine': dict(color='#d62728', marker='o', linestyle='-', label='Wine'),
    }


def plot_combined(
    *,
    out_dir: Path,
    kernel: str,
    series,
    csv_paths=(),
    log_x=True,
    log_y=True,
    out_name="svm_martin_ratio_all",
    x_label="Lambda",
    y_label=None,
    show=True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    style = dataset_style()
    fig, ax = plt.subplots(figsize=(5,4))
    eps = 1e-12
    for ds, xs, ys, ss in series:
        st = style.get(ds, dict(color='black', marker='o', linestyle='-', label=ds))
        mark_every = max(1, len(xs) // 8)
        ax.plot(xs, ys, lw=4, markersize=12, markevery=mark_every, **{k: st[k] for k in ['color', 'marker', 'linestyle', 'label']})
        ax.fill_between(xs, np.maximum(ys - ss, eps), np.maximum(ys + ss, eps), color=st['color'], alpha=0.18)


    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=14)
    elif "d2d" in kernel:
        ax.set_ylabel(r'Iterations RS / GS', fontsize=14)
    else:
        ax.set_ylabel(r'RS / GS', fontsize=14)


    ax.set_xlabel(x_label, fontsize=14)
    ax.legend(title='Dataset', loc='best', fontsize=14,title_fontsize=16)
    fig.tight_layout()

    out_path = out_dir / f"{out_name}_{kernel}.png"
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def load_series(csv_path: Path):
    """Load ratios, retaining missing/nonpositive values as gaps on the x grid."""
    df = pd.read_csv(csv_path)
    required = {'frac', 'mean_ratio', 'std_ratio'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")

    df = df.sort_values('frac').replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['frac'])
    # On the log y-axis, zero and negative means cannot be shown. Keep their
    # lambda positions as NaNs so marker spacing stays anchored to the x grid.
    df.loc[df['mean_ratio'] <= 0, 'mean_ratio'] = np.nan
    df['std_ratio'] = df['std_ratio'].fillna(0.0)
    return (
        df['frac'].to_numpy(dtype=float),
        df['mean_ratio'].to_numpy(dtype=float),
        df['std_ratio'].to_numpy(dtype=float),
    )


def plot_available_ratios(
    *,
    results_dir: Path = Path('results/ratios'),
    datasets,
    passive_losses=(),
    active_losses=(),
    out_dir: Path | None = None,
):
    """Create one combined plot per loss from ratio CSVs that already exist.

    CSV names are expected to follow the format produced by
    ``export_ratio_csvs``: ``{passive|active}_erm_{dataset}_{loss}.csv``.
    Missing CSVs are skipped so the same call works before and after active
    experiments have been exported.
    """
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir is not None else results_dir.parent / 'plots'
    outputs = []

    for mode, losses in (('passive', passive_losses), ('active', active_losses)):
        for loss in losses:
            series = []
            csv_paths = []
            for dataset in datasets:
                csv_path = results_dir / f'{mode}_erm_{dataset}_{loss}.csv'
                if not csv_path.is_file():
                    continue

                xs, ys, ss = load_series(csv_path)
                if len(xs) == 0:
                    warnings.warn(f'Skipping {csv_path}: no finite ratio values')
                    continue

                series.append((dataset, xs, ys, ss))
                csv_paths.append(csv_path)

            if not series:
                continue

            outputs.append(
                plot_combined(
                    out_dir=out_dir,
                    kernel=f'{mode}_{loss}',
                    series=series,
                    csv_paths=csv_paths,
                    log_x=True,
                    log_y=True,
                    out_name=f'{mode}_erm',
                )
            )

    return outputs
