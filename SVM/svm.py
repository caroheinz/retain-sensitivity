from typing import Optional, Tuple

import numpy as np


def rbf_kernel_matrix(X: np.ndarray, gamma: float) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-gamma * dist2)


def fit_svm_and_stats(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: str,
    C: float,
    gamma: Optional[float],
    tol: float,
    max_iter: int,
    cache_size_mb: int,
) -> Tuple[float, float, float]:
    """
    Returns (w_norm_sq, train_acc, min_yf) where margin = 1/sqrt(w_norm_sq).
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.svm import LinearSVC, SVC

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    if kernel == "linear":
        clf = LinearSVC(
            C=C,
            loss="hinge",
            dual="auto",
            tol=tol,
            max_iter=max_iter,
            fit_intercept=True,
            random_state=0,
        )
        clf.fit(X, y)
        f = clf.decision_function(X).astype(float)
        w = clf.coef_.ravel().astype(float)
        w_norm_sq = float(np.dot(w, w))
    elif kernel == "rbf":
        g = (1.0 / X.shape[1]) if gamma is None else float(gamma)
        clf = SVC(C=C, kernel="rbf", gamma=g, tol=tol, max_iter=max_iter, cache_size=float(cache_size_mb))
        clf.fit(X, y)
        f = clf.decision_function(X).astype(float)
        dual = clf.dual_coef_.ravel().astype(float)  # alpha_i * y_i on SVs
        Ksv = rbf_kernel_matrix(clf.support_vectors_.astype(float), g)
        w_norm_sq = float(dual @ (Ksv @ dual))
    else:
        raise ValueError("kernel must be 'linear' or 'rbf'")

    pred = np.sign(f)
    pred[pred == 0] = 1.0
    train_acc = float(np.mean(pred == y))
    min_yf = float(np.min(y * f))
    return w_norm_sq, train_acc, min_yf

