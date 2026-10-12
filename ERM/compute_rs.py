import numpy as np
import pandas as pd
from load_data import sample_data
from sklearn.linear_model import LogisticRegression


def compute_smallest_eigenvalue(X):
    H = X.T @ X
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[0]

def compute_biggest_eigenvalue(X):
    H = X.T @ X
    eigenvalues = np.linalg.eigvalsh(H)
    return eigenvalues[-1]



# Sensitivity computations helpers

def return_sens_ERM(L, n, lambda_S):
    return L / (n * lambda_S)

def return_sens_ERM_active_d2d(L, n, lambda_S, lambda_top,  eps, delta, sigma):
    sens = return_sens_ERM(L, n, lambda_S)
    b = np.sqrt(2*np.log(1/delta)+2*eps) - np.sqrt(2*np.log(1/delta))
    kappa = lambda_top / lambda_S
    if kappa < 1:
        raise ValueError("kappa error")
    gamma = (kappa -1) / (kappa +1)
    I = np.ceil(np.log( sens / (sigma * b)) / np.log(1/gamma))
    return I 

def return_sens_ERM_active_newton(L, n, M,  lambda_S):
    GS = return_sens_ERM(L, n, lambda_S)
    return GS**2 * (M / (2 * lambda_S))


def RStheory_logloss_newton_local(X_retain, w_retain, B, lambd, m=1,
                                  radius_factor=2.0):
    """Margin-aware Newton RS certificate for logistic ERM.

    Uses a fixed trial radius (no search): r = radius_factor * q / mu_R(0).
    If q / mu_R(r) < r, returns the certified Newton sensitivity and details;
    otherwise returns (None, details), allowing callers to use the legacy bound.

    Assumes F_D(w) = mean_{z in D} loss(w; z) + lambd/2 * ||w||^2,
    ||x|| <= B, and a neighboring dataset formed by adding at most m points.
    """
    X_retain = np.asarray(X_retain, dtype=float)
    w_retain = np.asarray(w_retain, dtype=float).reshape(-1)
    n, d = X_retain.shape
    if w_retain.size != d:
        raise ValueError("w_retain dimension must match X_retain")
    if m < 1 or radius_factor <= 1:
        raise ValueError("m must be positive and radius_factor must exceed 1")

    # At the retain ERM, bound the gradient change caused by adding m points.
    # The shared regularizer cancels except for the changed empirical-mean weight.
    q = m * (B + lambd * np.linalg.norm(w_retain)) / (n + m)
    M = B**3 / (6 * np.sqrt(3))
    x_norms = np.linalg.norm(X_retain, axis=1)
    margins = np.abs(X_retain @ w_retain)

    def curvature_lower_bound(radius):
        logits = margins + x_norms * radius
        exp_neg = np.exp(-np.minimum(logits, 745.0))
        weights = exp_neg / (1.0 + exp_neg)**2
        H_lower = (X_retain.T * weights) @ X_retain / (n + m)
        H_lower += lambd * np.eye(d)
        return float(np.linalg.eigvalsh(H_lower)[0])

    mu_zero = curvature_lower_bound(0.0)
    details = {"certified": False, "q": q, "radius": None,
               "curvature": mu_zero, "displacement": None}
    if mu_zero <= 0 or not np.isfinite(mu_zero):
        return None, details

    # A simple trial radius, not a search for the smallest certified one.
    radius = radius_factor * q / mu_zero
    if radius == 0:
        details.update(certified=True, radius=0.0, displacement=0.0)
        return 0.0, details

    mu_radius = curvature_lower_bound(radius)
    if mu_radius <= 0 or not np.isfinite(mu_radius):
        details.update(radius=radius, curvature=mu_radius)
        return None, details

    displacement = q / mu_radius
    details.update(radius=radius, curvature=mu_radius,
                   displacement=displacement)
    if not displacement < radius:
        return None, details

    sensitivity = displacement**2 * M / (2.0 * mu_radius)
    details["certified"] = True
    return sensitivity, details


# Global Sensitivity computations

def GStheory(loss_name, B, lambd, n, R, passive = True, eps = None, delta = None, sigma = None):
    """
    Calculates Theoretical Global Sensitivity.
    """

    if 'MSE' in loss_name:
        if lambd == 0:
            GS = np.inf
            
        else:
            L = B**3 / lambd + B

            lambda_S = lambd
            lambda_top = B**2 + lambd

            if passive:
                GS = return_sens_ERM(L, n, lambda_S)
            else: # Active case, d2d as newton we dont have
                GS = return_sens_ERM_active_d2d(L, n, lambda_S, lambda_top, eps, delta, sigma)

        return GS
        
    elif 'LogLoss' in loss_name:
        if lambd == 0:
            GS = np.inf
        else:
            L = B + lambd * R
            lambda_top = 1/4 * B**2 + lambd
            lambda_S = lambd

            if passive:
                GS = return_sens_ERM(L, n, lambda_S)

            else: 
                if "d2d" in loss_name:
                    GS = return_sens_ERM_active_d2d(L, n, lambda_S, lambda_top, eps, delta, sigma)
                elif "newton" in loss_name: # "newton" in loss_name:
                    M = B**3 / (6 * np.sqrt(3))
                    GS = return_sens_ERM_active_newton(L, n, M, lambda_S)
        return GS
    else:
        print(f"Skipping unknown loss name: {loss_name}") 
        pass


 # Retain Sensitivity computations  

def RStheory(loss_name, smallest_eigen_value, B, R, lambd, n, curvature=None, passive = True, max_eigen_value = None, eps = None, delta = None, sigma = None):
    """
    Calculates Theoretical Retain Sensitivity.
    """

    if 'MSE' in loss_name:

        lambda_S = 1/n * smallest_eigen_value + lambd
        L = B**3 / (1/(n+1) * smallest_eigen_value + lambd) + B
        # L = B**3 / (lambd) + B


        if passive:
            RS = return_sens_ERM(L, n, lambda_S)

        else: # Active case, d2d as newton we dont have
            lambda_top = 1/n * max_eigen_value + lambd
            RS = return_sens_ERM_active_d2d(L, n, lambda_S, lambda_top, eps, delta, sigma)
        return RS
        
    elif 'LogLoss' in loss_name:
        L = B + lambd * R
        lambda_S = 1/n * smallest_eigen_value * ((2 * np.cosh(B * R / 2))**-2) + lambd
    
        if passive:

            if "_root" in loss_name:
                M_Hessian = B**3 / (6 * np.sqrt(3))
                if curvature**2 - 4 * M_Hessian * L / n < 0:
                    RS = np.inf
                else:
                    RS = (curvature - np.sqrt(curvature**2 - 4 * M_Hessian * L / n)) / (2 * M_Hessian)

            else :
                RS = return_sens_ERM(L, n, lambda_S)

        else: 
            if "d2d" in loss_name:
                lambda_top = 1/4 * 1/n * max_eigen_value + lambd
                RS = return_sens_ERM_active_d2d(L, n, lambda_S, lambda_top, eps, delta, sigma)
            elif "newton" in loss_name: # "newton" in loss_name:
                M = B**3 / (6 * np.sqrt(3))
                RS = return_sens_ERM_active_newton(L, n, M, lambda_S)
        return RS
    else:
        print(f"Skipping unknown loss name: {loss_name}") 
        pass



# ERM model computation for curvature     

def erm_model(loss_name, X, y, lambd, R_w=None):
    n, d = X.shape

    def project_to_ball(w):
        if R_w is None:
            return w
        w_norm = np.linalg.norm(w)
        if w_norm <= R_w or w_norm == 0.0:
            return w
        return (R_w / w_norm) * w

    if "MSE" in loss_name:
        H = (X.T @ X) / n + lambd * np.eye(d)
        curvature = np.min(np.linalg.eigvalsh(H))

        if R_w is None:
            I = np.eye(d)
            w_erm = np.linalg.inv(X.T @ X + n * lambd * I) @ (X.T @ y)
            return w_erm, curvature

        eigvals = np.linalg.eigvalsh((X.T @ X) / n)
        smooth = np.max(eigvals) + lambd
        step_size = 1.0 / max(smooth, 1e-12)
        w = project_to_ball(np.linalg.solve(X.T @ X + n * lambd * np.eye(d), X.T @ y))

        for _ in range(5000):
            grad = (X.T @ (X @ w - y)) / n + lambd * w
            w_next = project_to_ball(w - step_size * grad)
            if np.linalg.norm(w_next - w) < 1e-8:
                w = w_next
                break
            w = w_next

        return w, curvature

    elif "LogLoss" in loss_name:
        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        def logloss_gradient(w):
            probs = sigmoid(X @ w)
            return X.T @ (probs - y) / n + lambd * w

        if R_w is None:
            if lambd == 0:
                C = np.inf  # Large C for no regularization
            else:
                C = 1 / (n * lambd)
            model = LogisticRegression(C=C, fit_intercept=False, solver='lbfgs', max_iter=1000)
            model.fit(X, y)
            w_erm = model.coef_.flatten()
        else:
            smooth = 0.25 * np.max(np.linalg.eigvalsh((X.T @ X) / n)) + lambd
            step_size = 1.0 / max(smooth, 1e-12)

            if lambd == 0:
                w = np.zeros(d)
            else:
                C = 1 / (n * lambd)
                model = LogisticRegression(C=C, fit_intercept=False, solver='lbfgs', max_iter=1000)
                model.fit(X, y)
                w = model.coef_.flatten()
            w = project_to_ball(w)

            for _ in range(5000):
                grad = logloss_gradient(w)
                w_next = project_to_ball(w - step_size * grad)
                if np.linalg.norm(w_next - w) < 1e-8:
                    w = w_next
                    break
                w = w_next
            w_erm = w

        probs = sigmoid(X @ w_erm)
        S = probs * (1 - probs)
        H = (X.T * S) @ X / n + lambd * np.eye(d)
        curvature = np.min(np.linalg.eigvalsh(H))

        return w_erm, curvature

    else:
        raise ValueError(f"Unknown loss name: {loss_name}")





def run_experiments(X_pool, y_pool,n_list, lambda_list, loss_list, seeds, *, B_x, R_w, passive=True, eps_list=None, delta=None, sigma=None):
    """
    Runs the full suite of experiments.
    X_pool uses it instead of loading from data_source.
    Returns actual n and requested_n, with RS/GS bounds and run metadata.
    """
    results = []


    for loss_name in loss_list:
        for n_requested in n_list:
            for lambd in lambda_list:
                for seed in seeds:
                    X, y = sample_data(X_pool, y_pool, n_requested, seed, B_x)
                    # Use the actual sample size when a dataset pool is smaller.
                    n = len(y)

                    if passive:
                        min_eig = compute_smallest_eigenvalue(X)
                        # These bounds assume ||w|| <= R_w; fit the matching
                        # constrained ERM rather than assuming the bound.
                        w_erm, curvature = erm_model(loss_name, X, y, lambd, R_w)
                        gs_val = GStheory(loss_name=loss_name, B=B_x, lambd=lambd, n=n, R=R_w)
                        rs_val = RStheory(
                            loss_name, smallest_eigen_value=min_eig, B=B_x, R=R_w, lambd=lambd, n=n, curvature=curvature
                        )
                        results.append({
                            'loss': loss_name,
                            'n': n,
                            'requested_n': n_requested,
                            'lambda': lambd,
                            'RS': rs_val,
                            'GS': gs_val,
                            'seed': seed,
                            'passive': True
                        })
                    else:
                        # Active case (kept for compatibility; expects eps_list etc.)
                        X_pool_local = X
                        y_pool_local = y
                        for eps in eps_list or []:
                            min_eig = compute_smallest_eigenvalue(X_pool_local)
                            max_eig = compute_biggest_eigenvalue(X_pool_local)
                            if "LogLoss_newton" in loss_name:
                                # Match the current Newton utility calibration.
                                # GS uses the regularization-derived global radius;
                                # RS uses the certified margin-aware local bound.
                                w_retain, _ = erm_model(
                                    "LogLoss", X_pool_local, y_pool_local, lambd
                                )
                                radius_gs = np.sqrt(2.0 * np.log(2.0) / lambd)
                                gs_val = GStheory(
                                    loss_name=loss_name, B=B_x, lambd=lambd, n=n,
                                    R=radius_gs, passive=False, eps=eps,
                                    delta=delta, sigma=sigma,
                                )
                                rs_val, rs_details = RStheory_logloss_newton_local(
                                    X_pool_local, w_retain, B_x, lambd, m=1
                                )
                                if not rs_details["certified"]:
                                    rs_val = np.nan
                            else:
                                gs_val = GStheory(
                                    loss_name=loss_name, B=B_x, lambd=lambd, n=n,
                                    R=R_w, passive=False, eps=eps, delta=delta,
                                    sigma=sigma,
                                )
                                rs_val = RStheory(
                                    loss_name, smallest_eigen_value=min_eig,
                                    max_eigen_value=max_eig, B=B_x, R=R_w,
                                    lambd=lambd, n=n, passive=False, eps=eps,
                                    delta=delta, sigma=sigma,
                                )
                            results.append({
                                'loss': loss_name,
                                'n': n,
                                'requested_n': n_requested,
                                'lambda': lambd,
                                'RS': rs_val,
                                'GS': gs_val,
                                'seed': seed,
                                'eps': eps,
                                'passive': False
                            })

    return pd.DataFrame(results)


def run_experiments_on_datasets(n_list, lambda_list, loss_list, seeds, datasets, *, B_x, R_w, passive=True, eps_list=None, delta=None, sigma=None):
    """
    Runs run_experiments over multiple preloaded datasets and returns a combined DataFrame.
    datasets: list of dicts with keys: name, X, y_bin.
    Returns DataFrame with columns: [loss, seed, n, lambda, RS, GS, dataset].
    """
    frames = []

    for ds in datasets:
        name = ds
        X_pool = datasets[ds]['X']
        y_bin = datasets[ds]['y_bin']

        if name is None or X_pool is None or y_bin is None:
            raise ValueError("Each dataset must provide 'name', 'X', and 'y_bin'")

        df = run_experiments(
            X_pool=X_pool,
            y_pool=y_bin,
            n_list=n_list,
            lambda_list=lambda_list,
            loss_list=loss_list,
            seeds=seeds,
            B_x=B_x,
            R_w=R_w,
            passive=passive,
            eps_list=eps_list,
            delta=delta,
            sigma=sigma,
        )

        df['dataset'] = name
        frames.append(df)

    if len(frames) == 0:
        return pd.DataFrame(columns=['loss','seed','n','lambda','RS','GS','dataset'])

    return pd.concat(frames, ignore_index=True)
