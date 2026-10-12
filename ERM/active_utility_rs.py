import numpy as np
import pandas as pd
from compute_rs import (GStheory, RStheory, erm_model,
                        RStheory_logloss_newton_local)
from load_data import sample_data

from sklearn.model_selection import train_test_split
from torch import norm
from sklearn.metrics import accuracy_score


def active_unlearning_newton(w_full, X_rem, y_rem, X_retain, y_retain, lambd, noise_std):
    """
    Implements Unlearning via Newton Step Update (Sekhari et al., 2021).
    Uses the 'Recover Hessian' strategy:
    H_approx = 1/n * ( (n+1)*H_full - H_rem )
    w_new = w_full + 1/n * H_approx^-1 * grad_rem

    Args:
        w_full: Weights learned on the full dataset (n+1 points).
        X_rem, y_rem: The features/label of the point to remove.
        X_retain, y_retain: The retained dataset (n points).
        lambd: Regularization parameter used in original training.
        noise_std: Standard deviation of the noise to add.

    Returns:
        w_approx: The unlearned weight vector.
    """
    n = len(y_retain)         # This is 'n' (retained size)
    n_total = n + 1           # This is 'n+1' (full size)
    d = w_full.shape[0]

    # --- 1. Compute Full Hessian (H_full) on ALL data (Retained + Removed) ---
    # Reconstruct full dataset for calculation
    X_full = np.vstack([X_retain, X_rem])

    # Calculate probabilities for LogLoss (Sigmoid)
    scores_full = X_full @ w_full
    probs_full = 1 / (1 + np.exp(-scores_full))

    # H_full = (1/N_total) * X.T @ S @ X + lambda * I
    s_diag_full = probs_full * (1 - probs_full)
    H_full = (X_full.T * s_diag_full) @ X_full / n_total
    H_full += lambd * np.eye(d) # Add regularization to full Hessian

    # --- 2. Compute Removed Hessian (H_rem) on removed point only ---
    score_rem = X_rem @ w_full
    prob_rem = 1 / (1 + np.exp(-score_rem))

    # Treat the L2 penalty consistently as part of each per-example
    # objective: f(w; z) = loss(w; z) + (lambd / 2) * ||w||^2.
    s_rem = prob_rem * (1 - prob_rem)
    H_rem = (X_rem.T * s_rem) @ X_rem + lambd * np.eye(d)

    # Recover the retained-data Hessian for mean loss plus the same L2 penalty.
    H_approx = (1 / n) * ((n_total * H_full) - H_rem)

    # The removed point per-example gradient includes the L2 term too.
    grad_rem = (prob_rem - y_rem) * X_rem.flatten() + lambd * w_full

    # --- 5. Newton Update ---
    # w = w_full + 1/n * H^-1 * grad
    # Solve (A,b): Ax=b -> x= A^-1 b
    # Use solve for stability: step = H^-1 * grad
    step = np.linalg.solve(H_approx, grad_rem)
    w_unlearned = w_full + (1 / n) * step
    noise = np.random.normal(0, noise_std, size=w_full.shape)
    w_unlearned += noise

    return w_unlearned


def get_accuracy(w, X, y, loss_name='LogLoss'):
    """Computes accuracy for binary classification (0/1)."""
    if len(y) == 0: return 0.0
    if "LogLoss" in loss_name:
        # For logistic regression, use 0 threshold
        preds = (X @ w >= 0).astype(int)
    elif "MSE" in loss_name:
        # For MSE, also use 0.5 threshold for binary classification 
        preds = (X @ w >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
    
    return accuracy_score(y, preds)

def gaussian_mechanism_noise(sensitivity, epsilon, delta):
    """sigma = sqrt(2 * ln(1.25/delta)) * (sensitivity / epsilon)"""
    c = np.sqrt(2 * np.log(1.25 / delta))
    return c * sensitivity / epsilon



def _d2d_gradient_trajectory(w_full, X_retain, y_retain, smallest_eigen,
                            biggest_eigen, iteration_counts, R_w,
                            loss_name="LogLoss", lambd=0.0):
    """Cache deterministic projected-GD iterates at requested step counts."""
    n = len(y_retain)
    if n == 0:
        raise ValueError("X_retain must contain at least one point")
    if R_w <= 0:
        raise ValueError("R_w must be positive")

    requested = [int(T) for T in iteration_counts]
    if not requested:
        raise ValueError("iterations must contain at least one step count")
    if min(requested) < 0:
        raise ValueError("iteration counts must be nonnegative")
    targets = set(requested)

    lambda_min = max(float(smallest_eigen), 1e-12)
    lambda_max = max(float(biggest_eigen), lambda_min)
    step_size = 2.0 / (lambda_min + lambda_max)
    gamma = (lambda_max - lambda_min) / (lambda_max + lambda_min)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def retained_gradient(w):
        scores = X_retain @ w
        if "LogLoss" in loss_name:
            residual = sigmoid(scores) - y_retain
        elif "MSE" in loss_name:
            residual = scores - y_retain
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")
        grad = X_retain.T @ residual / n
        if lambd != 0:
            grad = grad + lambd * w
        return grad

    def project_to_ball(w):
        w_norm = np.linalg.norm(w)
        if w_norm <= R_w or w_norm == 0.0:
            return w
        return (R_w / w_norm) * w

    iterates = {}
    w = project_to_ball(w_full.copy())
    if 0 in targets:
        iterates[0] = w.copy()
    for step in range(1, max(requested) + 1):
        w = w - step_size * retained_gradient(w)
        w = project_to_ball(w)
        if step in targets:
            iterates[step] = w.copy()
    return iterates, gamma


def _d2d_noise_denominator(epsilon, delta):
    if epsilon <= 0 or delta <= 0 or delta >= 1:
        raise ValueError("epsilon must be positive and delta must be in (0, 1)")
    denom = np.sqrt(2 * np.log(1.0 / delta) + 2 * epsilon) - np.sqrt(
        2 * np.log(1.0 / delta)
    )
    if denom <= 0:
        raise ValueError("invalid epsilon, delta combination for paper noise formula")
    return denom


def active_unlearning_d2d(w_full, X_rem, y_rem, X_retain, y_retain,
                          smallest_eigen, biggest_eigen, sensitivity,
                          epsilon, delta, T, R_w, loss_name="LogLoss",
                          lambd=0.0):
    """Run T projected-GD steps on the retained objective and add noise."""
    if R_w <= 0:
        raise ValueError("R_w must be positive")
    denom = _d2d_noise_denominator(epsilon, delta)
    iterates, gamma = _d2d_gradient_trajectory(
        w_full, X_retain, y_retain, smallest_eigen, biggest_eigen, [T],
        R_w, loss_name=loss_name, lambd=lambd,
    )
    sigma = sensitivity * (gamma ** int(T)) / denom
    return iterates[int(T)] + np.random.normal(0.0, sigma, size=w_full.shape)


def _validate_experiment_parameters(*, epsilon, delta, B_x, R_w=None):
    """Reject invalid privacy and norm parameters before running a sweep."""
    for name, value in (("epsilon", epsilon), ("B_x", B_x)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be finite and in (0, 1)")
    if R_w is not None and (not np.isfinite(R_w) or R_w <= 0):
        raise ValueError("R_w must be finite and positive")


def run_experiment_d2d(datasets, n_values, lambd_values, loss_name="LogLoss",
                       num_trials=5, random_seed=42, iterations=[1000], *,
                       epsilon, delta, B_x, R_w):
    """Run D2D utility experiments, reusing each fit and GD trajectory."""
    results = []
    _validate_experiment_parameters(epsilon=epsilon, delta=delta, B_x=B_x, R_w=R_w)
    noise_denom = _d2d_noise_denominator(epsilon, delta)
    iteration_counts = [int(T) for T in iterations]
    if not iteration_counts:
        raise ValueError("iterations must contain at least one step count")
    if min(iteration_counts) < 0:
        raise ValueError("iteration counts must be nonnegative")

    for dataset_name, data in datasets.items():
        print(f"Starting experiment on {dataset_name}...")
        X_pool = data["X"]
        y_pool = data["y_bin"]
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=random_seed
        )
        X_test, y_test = sample_data(
            X_test, y_test, len(X_test), seed=random_seed, B=B_x
        )

        for n_requested in n_values:
            for lambd in lambd_values:
                rows_by_T = {T: [] for T in set(iteration_counts)}
                for trial in range(num_trials):
                    seed = random_seed + trial
                    X_full_sample, y_full_sample = sample_data(
                        X_train_pool, y_train_pool, n_requested + 1,
                        seed=seed, B=B_x,
                    )
                    n = len(y_full_sample) - 1

                    if "LogLoss" in loss_name:
                        w_full, curv_full = erm_model(
                            "LogLoss", X_full_sample, y_full_sample, lambd, R_w
                        )
                    elif "MSE" in loss_name:
                        w_full, curv_full = erm_model(
                            "MSE", X_full_sample, y_full_sample, lambd, R_w
                        )
                    else:
                        raise ValueError(f"Unknown loss name: {loss_name}")

                    remove_idx = np.random.randint(n + 1)
                    X_rem = X_full_sample[remove_idx:remove_idx + 1]
                    y_rem = y_full_sample[remove_idx:remove_idx + 1]
                    X_retain = np.delete(X_full_sample, remove_idx, axis=0)
                    y_retain = np.delete(y_full_sample, remove_idx, axis=0)
                    assert len(y_retain) == n

                    if "LogLoss" in loss_name:
                        w_retrain, _ = erm_model(
                            "LogLoss", X_retain, y_retain, lambd, R_w
                        )
                    else:
                        w_retrain, _ = erm_model(
                            "MSE", X_retain, y_retain, lambd, R_w
                        )

                    eigvals = np.linalg.eigvalsh(X_retain.T @ X_retain)
                    smallest_eigen = np.min(eigvals)
                    max_eigen = np.max(eigvals)

                    if "LogLoss" in loss_name:
                        GS_sens = GStheory(
                            "LogLoss", B_x, lambd, n, R_w,
                            passive=True, eps=epsilon, delta=delta,
                        )
                        lambda_min_GS = lambd
                        lambda_max_GS = 0.25 * B_x**2 + lambd

                        RS_sens = RStheory(
                            "LogLoss", smallest_eigen, B_x, R_w, lambd, n,
                            curvature=curv_full, passive=True,
                            max_eigen_value=max_eigen,
                            eps=epsilon, delta=delta,
                        )
                        logistic_curvature_factor = (
                            2 * np.cosh(B_x * R_w / 2)
                        ) ** -2
                        lambda_min_RS = (
                            smallest_eigen / n
                        ) * logistic_curvature_factor + lambd
                        lambda_max_RS = 0.25 * max_eigen / n + lambd
                    else:
                        GS_sens = GStheory(
                            "MSE", B_x, lambd, n, R_w,
                            passive=True, eps=epsilon, delta=delta,
                        )
                        lambda_min_GS = lambd
                        lambda_max_GS = B_x**2 + lambd

                        RS_sens = RStheory(
                            "MSE", smallest_eigen, B_x, R_w, lambd, n,
                            curvature=curv_full, passive=True,
                            max_eigen_value=max_eigen,
                            eps=epsilon, delta=delta,
                        )
                        lambda_min_RS = smallest_eigen / n + lambd
                        lambda_max_RS = max_eigen / n + lambd

                    # Compute each deterministic path only once. Use separate
                    # paths for GS and RS because their curvature bounds differ.
                    path_GS, gamma_GS = _d2d_gradient_trajectory(
                        w_full, X_retain, y_retain,
                        lambda_min_GS, lambda_max_GS, iteration_counts, R_w,
                        loss_name=loss_name, lambd=lambd,
                    )
                    path_RS, gamma_RS = _d2d_gradient_trajectory(
                        w_full, X_retain, y_retain,
                        lambda_min_RS, lambda_max_RS, iteration_counts, R_w,
                        loss_name=loss_name, lambd=lambd,
                    )

                    # Preserve the prior common-random-number coupling: each T
                    # shares the same standard-normal draw per calibration,
                    # scaled by that T's own calibrated standard deviation.
                    noise_z_GS = np.random.normal(size=w_full.shape)
                    noise_z_RS = np.random.normal(size=w_full.shape)

                    acc_full = get_accuracy(w_full, X_test, y_test, loss_name)
                    acc_retrain = get_accuracy(w_retrain, X_test, y_test, loss_name)
                    dist_base = np.linalg.norm(w_full - w_retrain)

                    for T in iteration_counts:
                        sigma_GS = GS_sens * (gamma_GS ** T) / noise_denom
                        sigma_RS = RS_sens * (gamma_RS ** T) / noise_denom
                        w_d2d_GS = path_GS[T] + sigma_GS * noise_z_GS
                        w_d2d_RS = path_RS[T] + sigma_RS * noise_z_RS
                        acc_GS = get_accuracy(w_d2d_GS, X_test, y_test, loss_name)
                        acc_RS = get_accuracy(w_d2d_RS, X_test, y_test, loss_name)
                        rows_by_T[T].append({
                            "dataset": dataset_name, "n": n, "lambd": lambd,
                            "trial": trial, "iterations": T,
                            "loss_name": loss_name,
                            "GS_sens": GS_sens, "RS_sens": RS_sens,
                            "acc_Full": acc_full, "acc_Retrain": acc_retrain,
                            "acc_GS": acc_GS, "acc_RS": acc_RS,
                            "dist_GS": np.linalg.norm(w_d2d_GS - w_retrain),
                            "dist_RS": np.linalg.norm(w_d2d_RS - w_retrain),
                            "dist_Base": dist_base,
                        })

                # Retain the original iteration-major, then trial-major row order.
                for T in iteration_counts:
                    results.extend(rows_by_T[T])

    return pd.DataFrame(results)


def run_experiment_newton(datasets, n_values, lambd_values, num_trials=5, random_seed=42,
                          local_radius_factor=2.0, *, epsilon, delta, B_x):
    """Run Newton utility experiments with certified local RS calibration.

    Raises if the fixed trial radius does not certify RS; it never substitutes
    the legacy Newton RS value. Increase local_radius_factor to try a larger
    radius, or add a radius-search method separately.
    """
    results = []
    _validate_experiment_parameters(epsilon=epsilon, delta=delta, B_x=B_x)

    # Iterate over each dataset in the dictionary
    for dataset_name, data in datasets.items():
        print(f"Starting experiment on {dataset_name}...")

        X_pool = data["X"]
        y_pool = data["y_bin"]

        # 1. Create Fixed Test Set (20%)
        # We strip the test set first so n_values are sampled from the remaining pool
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=random_seed
        )
        # Match the training representation: clip features and append one
        # intercept column before evaluating held-out accuracy.
        X_test, y_test = sample_data(
            X_test, y_test, len(X_test), seed=random_seed, B=B_x
        )

        for n in n_values:
            for lambd in lambd_values:
                for trial in range(num_trials):
                    seed = random_seed + trial

                    # --- A. Data Prep ---
                    # Sample n+1 points (Full dataset size)
                    X_full_sample, y_full_sample = sample_data(
                        X_train_pool, y_train_pool, n + 1, seed=seed, B=B_x
                    )
                    n = len(y_full_sample) - 1

                    # --- B. Learn Base Classifier (on n+1 points) ---
                    w_full, _ = erm_model("LogLoss", X_full_sample, y_full_sample, lambd)

                    # --- Prepare for Unlearning ---
                    # Pick one point to remove
                    remove_idx = np.random.randint(n + 1)

                    X_rem = X_full_sample[remove_idx:remove_idx+1]
                    y_rem = y_full_sample[remove_idx:remove_idx+1]

                    X_retain = np.delete(X_full_sample, remove_idx, axis=0)
                    y_retain = np.delete(y_full_sample, remove_idx, axis=0)

                    # Verify retain size is n
                    assert len(y_retain) == n

                    # --- B.1. Retraining (Gold Standard) ---
                    w_retrain, _ = erm_model("LogLoss", X_retain, y_retain, lambd)

                    # --- Sensitivity Calculations ---
                    # For regularized logistic ERM, F(w*) <= F(0) = log(2) implies
                    # lambd * ||w*||^2 / 2 <= log(2), giving a valid global radius.
                    R_w_bound = np.sqrt(2.0 * np.log(2.0) / lambd) if lambd > 0 else np.inf
                    GS_sens = GStheory("LogLoss_newton", B_x, lambd, n, R_w_bound,
                                       passive=False, eps=epsilon, delta=delta)

                    RS_sens, local_rs_details = RStheory_logloss_newton_local(
                        X_retain, w_retrain, B_x, lambd,
                        m=len(y_full_sample) - len(y_retain),
                        radius_factor=local_radius_factor)
                    if not local_rs_details["certified"]:
                        raise RuntimeError(
                            "The fixed-radius local RS certificate failed for "
                            f"dataset={dataset_name}, n={n}, lambda={lambd}, "
                            f"trial={trial}: {local_rs_details}. Increase "
                            "local_radius_factor or use a radius-search method."
                        )

                    # Calculate Noise Sigma
                    sigma_GS = gaussian_mechanism_noise(GS_sens, epsilon, delta)
                    sigma_RS = gaussian_mechanism_noise(RS_sens, epsilon, delta)


                    # --- B.2. Active Unlearning (Newton) ---
                    w_newton_GS = active_unlearning_newton(w_full, X_rem, y_rem, X_retain, y_retain, lambd, sigma_GS)
                    w_newton_RS = active_unlearning_newton(w_full, X_rem, y_rem, X_retain, y_retain, lambd, sigma_RS)

                    # --- C. Metrics ---
                    # Accuracy on Held-out Test Set
                    model_dimensions = {
                        len(w_full), len(w_retrain), len(w_newton_GS), len(w_newton_RS)
                    }
                    if model_dimensions != {X_test.shape[1]}:
                        raise ValueError(
                            "Model and test feature dimensions do not match: "
                            f"models={sorted(model_dimensions)}, test={X_test.shape[1]}"
                        )
                    acc_Full = get_accuracy(w_full, X_test, y_test)
                    acc_Retrain = get_accuracy(w_retrain, X_test, y_test)
                    acc_GS = get_accuracy(w_newton_GS, X_test, y_test)
                    acc_RS = get_accuracy(w_newton_RS, X_test, y_test)

                    # L2 Distance to Retrained Model
                    dist_GS = np.linalg.norm(w_newton_GS - w_retrain)
                    dist_RS = np.linalg.norm(w_newton_RS - w_retrain)
                    dist_Full = np.linalg.norm(w_full - w_retrain)

                    results.append({
                        "dataset": dataset_name, "n": n, "lambd": lambd, "trial": trial,
                        "GS_sens": GS_sens, "RS_sens": RS_sens,
                        "RS_local_certified": (
                            local_rs_details["certified"]
                            if local_rs_details is not None else False
                        ),
                        "RS_local_radius": (
                            local_rs_details["radius"]
                            if local_rs_details is not None else None
                        ),
                        "RS_local_displacement": (
                            local_rs_details["displacement"]
                            if local_rs_details is not None else None
                        ),
                        "acc_Full": acc_Full, "acc_Retrain": acc_Retrain,
                        "acc_GS": acc_GS, "acc_RS": acc_RS,
                        "dist_GS": dist_GS, "dist_RS": dist_RS, "dist_Base": dist_Full
                    })

    return pd.DataFrame(results)
