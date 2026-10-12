import math
import os
from pathlib import Path
from typing import Optional

import numpy as np


def ensure_writable_dirs(*directories: Path) -> None:
    """Create output directories and a writable Matplotlib cache."""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    if directories:
        cache_root = Path(directories[0])
        mpl_config = cache_root / ".mplconfig"
        mpl_config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))


def stratified_subsample_indices(
    y: np.ndarray, size: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    pos = np.where(y == 1.0)[0]
    neg = np.where(y == -1.0)[0]
    if size < 2 or len(pos) == 0 or len(neg) == 0:
        return None
    frac_pos = len(pos) / (len(pos) + len(neg))
    n_pos = int(round(size * frac_pos))
    n_pos = max(1, min(n_pos, size - 1, len(pos)))
    n_neg = size - n_pos
    if n_neg < 1 or n_neg > len(neg):
        n_neg = min(len(neg), size - 1)
        n_pos = size - n_neg
    if n_pos < 1 or n_neg < 1 or n_pos > len(pos) or n_neg > len(neg):
        return None
    idx = np.concatenate(
        [
            rng.choice(pos, size=n_pos, replace=False),
            rng.choice(neg, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(idx)
    return idx


def linspace_sizes(min_size: int, max_size: int, num: int) -> list[int]:
    min_size = min(int(min_size), int(max_size))
    max_size = int(max_size)
    if max_size < 2 or num <= 1:
        return [max_size]
    sizes = sorted(
        {
            int(round(value))
            for value in np.linspace(min_size, max_size, num=num)
        }
    )
    return [size for size in sizes if min_size <= size <= max_size]


def compute_min_size_from_frac(
    min_frac: float, n_ref: int, max_size: int
) -> int:
    fraction = max(0.0, min(float(min_frac), 1.0))
    size = int(math.ceil(fraction * n_ref))
    return max(2, min(size, int(max_size)))
