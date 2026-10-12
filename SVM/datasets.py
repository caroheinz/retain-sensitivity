from copy import deepcopy
from typing import Iterable

import numpy as np

from ERM.load_data import load_datasets


_DEFAULT_SPECS = {
    "wine": {"X": None, "y_bin": None, "specs": "red"},
    "mnist_rp50": {
        "X": None,
        "y_bin": None,
        "specs": ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9], 50),
    },
    "folktables": {
        "X": None,
        "y_bin": None,
        "specs": ("CA", 2018, "person"),
    },
    #"wbc": {"X": None, "y_bin": None, "specs": None},
}


def default_svm_dataset_specs(names: Iterable[str] | None = None):
    """Return fresh ERM-compatible dataset specifications for SVM runs."""
    selected = list(_DEFAULT_SPECS) if names is None else list(names)
    unknown = sorted(set(selected) - set(_DEFAULT_SPECS))
    if unknown:
        raise ValueError(
            f"Unknown datasets: {unknown}. Available: {sorted(_DEFAULT_SPECS)}"
        )
    return {name: deepcopy(_DEFAULT_SPECS[name]) for name in selected}


def load_svm_datasets(specs=None):
    """Load SVM datasets through the shared ERM dataset loaders."""
    if specs is None:
        specs = default_svm_dataset_specs()
    return load_datasets(specs)


def svm_labels(y_binary):
    """Convert shared 0/1 labels to the -1/+1 convention used by SVMs."""
    y = np.asarray(y_binary)
    unique = set(np.unique(y).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"Expected binary 0/1 labels, got {sorted(unique)}")
    return np.where(y == 1, 1.0, -1.0)


def dataset_title(name: str) -> str:
    return {
        "wine": "Wine quality: quality >= 6 vs < 6",
        "mnist_rp50": "MNIST RP50: digits >= 5 vs < 5",
        "folktables": "ACS CA: income > 50K vs <= 50K",
        #"wbc": "Wisconsin breast cancer: malignant vs benign",
    }.get(name, name)
