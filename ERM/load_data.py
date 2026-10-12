import numpy as np
import pandas as pd
from scipy import datasets
from torchvision.transforms import transforms
from medmnist import OCTMNIST
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from folktables import ACSDataSource, ACSIncome
from sklearn.datasets import load_breast_cancer
from torchvision.datasets import MNIST



# MNIST binary classification loader (two digit groups)
def load_mnist_binary_once(
    digits_a=(4,),
    digits_b=(5,),
    train=True,
    standardize=True,
    target_dim=None,
    standardize_before_projection=False,
):
    """
    Loads MNIST and returns a binary classification dataset for digits_a vs digits_b.
    digits_a are mapped to 0, digits_b are mapped to 1.
    If target_dim is set, applies Gaussian Random Projection to that dimension.
    If standardize_before_projection is true, standardizes the original pixel
    features before projection.
    Returns X (features), y_reg (scaled to [0,1]), y_bin (0/1).
    """

    digits_a = tuple(digits_a)
    digits_b = tuple(digits_b)

    if set(digits_a) & set(digits_b):
        raise ValueError("digits_a and digits_b must be disjoint")

    transform = transforms.ToTensor()
    dataset = MNIST(root='./data', train=train, download=True, transform=transform)

    imgs = []
    labels = []
    for img, label in dataset:
        if label in digits_a:
            imgs.append(img.numpy())
            labels.append(0)
        elif label in digits_b:
            imgs.append(img.numpy())
            labels.append(1)

    X = np.stack(imgs, axis=0).reshape(len(imgs), -1).astype(float)
    y_bin = np.array(labels, dtype=int)

    if standardize and standardize_before_projection:
        X = StandardScaler().fit_transform(X)

    if target_dim is not None:
        rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
        X = rp.fit_transform(X)

    if standardize and not standardize_before_projection:
        X = StandardScaler().fit_transform(X)

    y_reg = y_bin.astype(float)

    return X, y_bin


# WBC (Wisconsin Breast Cancer) loader
def load_wbc_once(standardize=True):
    """
    Loads the Wisconsin Breast Cancer dataset (sklearn).
    Returns X (features), y_reg (scaled to [0,1]), y_bin (labels 0/1).
    """

    data = load_breast_cancer()
    X = data.data.astype(float)
    y = data.target.astype(int)

    if standardize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    y_bin = y
    if y.max() > y.min():
        y_reg = (y - y.min()) / (y.max() - y.min())
    else:
        y_reg = y.astype(float)

    # y == y_bin

    return X, y_bin


## OctMNIST


def load_octmnist_once(target_dim=15):
    """
    Loads OctMNIST, flattens, and applies Random Projection.
    Reduces 784 features -> target_dim (e.g. 50).
    """
    print(f"Loading OctMNIST and projecting to {target_dim} dimensions...")

    # 1. Load Data
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = OCTMNIST(split='train', transform=data_transform, download=True, size=28)

    X_all = dataset.imgs
    y_all = dataset.labels.flatten()

    # 2. Custom Binary Mapping (Normal=0, Abnormal=1)
    y_binary = np.zeros_like(y_all)
    y_binary[(y_all == 0) | (y_all == 1) | (y_all == 2)] = 1

    # 3. Flatten & Normalize
    X_flat = X_all.reshape(X_all.shape[0], -1).astype(np.float32)
    X_flat /= 255.0

    # 4. RANDOM PROJECTION (The Fix)
    # We project 784 -> 50. This fixes the eigenvalue issue.
    # We use a fixed random_state so the projection is consistent across experiments
    rp = GaussianRandomProjection(n_components=target_dim, random_state=42)
    X_projected = rp.fit_transform(X_flat)

    print(f"Data shape changed from {X_flat.shape} to {X_projected.shape}")

    return X_projected, y_binary


# Folktables ACSIncome loader



# there is ACSPublicCoverage, ACSEmployment, ACSEmploymentStatus, ACSMobility, ACSPublicCoverage, ACSTravelTime
# 

def load_acs_income_once(states=("CA",), year=2018, survey="person", standardize=True):
    """
    Loads ACSIncome from folktables.
    Returns X (features), y_reg (scaled to [0,1]), y_bin (as provided by ACSIncome).
    """
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey=survey)
    acs_data = data_source.get_data(states=list(states), download=True)

    X, y, _ = ACSIncome.df_to_numpy(acs_data)

    # Convert to numpy arrays and optionally standardize
    X = X.astype(float)
    if standardize:
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(X)

    y = y.astype(float)
    # y from ACSIncome is binary for classification; keep as y_bin
    y_bin = y.astype(int)

    # For MSE, create a simple regression target in [0,1]
    if y.max() > y.min():
        y_reg = (y - y.min()) / (y.max() - y.min())
    else:
        y_reg = y.copy()

    # y == y_bin

    return X, y_bin


def sample_data(X_pool, y_pool, n, seed, B):
    """
    Sample data and return augmented features with L2 norm at most B.
    Clip features to radius B, append a constant intercept, then scale the
    entire vector by B / sqrt(B**2 + 1) (1/sqrt(2) when B=1).
    """
    
    if not np.isfinite(B) or B <= 0:
        raise ValueError("B must be finite and positive")
    np.random.seed(seed)

    if len(X_pool) < n:
        indices = np.arange(len(X_pool))
    else:
        indices = np.random.choice(len(X_pool), n, replace=False)

    X = X_pool[indices]
    y = y_pool[indices]

    # Projection to L2 ball (Clipping)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    scale_factors = np.minimum(1.0, B / (norms + 1e-8))
    X = X * scale_factors

    # Use one shared scale so the intercept remains constant and the full
    # augmented vector satisfies the feature-radius bound.
    intercept = np.ones((len(X), 1), dtype=X.dtype)
    X = np.hstack([X, intercept]) * (B / np.sqrt(B**2 + 1.0))

    return X, y

# UCI Wine Quality loader (red/white)
# Source: UCI Machine Learning Repository

def load_wine_quality_once(variant="red", standardize=True):
    """
    Loads UCI Wine Quality dataset.
    variant: 'red', 'white', or 'both'
    Returns X (features), y_reg (quality scaled to [0,1]), y_bin (quality >= 6).
    """
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    urls = {
        "red": base_url + "winequality-red.csv",
        "white": base_url + "winequality-white.csv"
    }

    if variant not in ("red", "white", "both"):
        raise ValueError("variant must be 'red', 'white', or 'both'")

    if variant == "both":
        df_red = pd.read_csv(urls["red"], sep=';')
        df_white = pd.read_csv(urls["white"], sep=';')
        df = pd.concat([df_red, df_white], ignore_index=True)
    else:
        df = pd.read_csv(urls[variant], sep=';')

    y = df['quality'].to_numpy()
    X = df.drop(columns=['quality']).to_numpy(dtype=float)

    if standardize:
        X = StandardScaler().fit_transform(X)

    y_reg = y / 10.0
    y_bin = (y >= 6).astype(int)

    return X, y_bin


def load_datasets(df_dict):
    """
    Given a dictionary of dataset names to (X, y_bin) tuples,
    returns a list of datasets in the format:
    [ name {"name": name, "X": X, "y_bin": y_bin, "specs": specs}, ...]
    """

    for name in df_dict.keys():
        specs = df_dict[name]["specs"]
        print(f"Loading dataset: {name} with specs: {specs}")
        if name == "wine":
            X, y_bin = load_wine_quality_once(variant=specs)
        elif "mnist" in name:
            X, y_bin = load_mnist_binary_once(
                digits_a=specs[0],
                digits_b=specs[1],
                target_dim=specs[2],
                standardize_before_projection=df_dict[name].get(
                    "standardize_before_projection", False
                ),
            )
        elif name == "octmnist":
            X, y_bin = load_octmnist_once(target_dim=50)
        elif name == "folktables":
            X, y_bin = load_acs_income_once(states=(specs[0],), year=specs[1], survey=specs[2])
        elif name == "acs_income_ny_2018":
            X, y_bin = load_acs_income_once(states=(specs[0],), year=specs[1], survey=specs[2])
        elif name == "wbc":
            X, y_bin = load_wbc_once()
        else:
            raise ValueError(f"Unknown dataset name: {name}")
        
        df_dict[name]["X"] = X
        df_dict[name]["y_bin"] = y_bin  

    return df_dict 



def export_ratio_csvs(df, out_dir='.', prefix='passive_erm'):
    """
    Export CSVs per (dataset, loss) with columns: frac, mean_ratio, std_ratio.
    frac is lambda; mean_ratio is RS/GS; std_ratio is std of RS/GS across n.
    Filenames: {prefix}_{dataset}_{loss}.csv
    """
    import os

    required = {'dataset', 'loss', 'lambda', 'RS', 'GS'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    os.makedirs(out_dir, exist_ok=True)

    # Compute ratio and summarize by lambda within each dataset/loss
    work = df.copy()
    work = work[(work['GS'] > 1e-7) & (work['GS'] != np.inf)]
    work['ratio'] = work['RS'] / work['GS']

    def _sanitize(s):
        keep = []
        for ch in str(s):
            if ch.isalnum() or ch in ('-', '_'):
                keep.append(ch)
            else:
                keep.append('_')
        return ''.join(keep)

    outputs = []
    for (dataset, loss), sub in work.groupby(['dataset', 'loss']):
        grouped = sub.groupby('lambda')['ratio']
        out = grouped.agg(['mean', 'std']).reset_index()
        out = out.rename(columns={'lambda': 'frac', 'mean': 'mean_ratio', 'std': 'std_ratio'})
        out = out.sort_values('frac')

        dataset_s = _sanitize(dataset)
        loss_s = _sanitize(loss)
        filename = f"{prefix}_{dataset_s}_{loss_s}.csv"
        path = os.path.join(out_dir, filename)
        out.to_csv(path, index=False)
        outputs.append(path)

    return outputs
