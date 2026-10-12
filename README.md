# Retain Sensitivity Experiments

Code and notebooks for the experiments accompanying the paper 'Less Noise, Same Certificate: Retain Sensitivity for Unlearning'. The repository contains experiment implementations for ERM, SVM, and minimum spanning trees (MST). 

## Repository layout

- `ERM/` — ERM sensitivity calculations, utility experiments, plotting, dataset loaders, and runner scripts.
- `SVM/` — hard-margin SVM implementation and ratio experiments. The runner uses the shared loaders in `ERM/`.
- `MST/` — MST sensitivity code, data loading, and exploratory notebook.

## Environment setup

Use a supported Python version and create an environment in the repository root. For example, with `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn matplotlib torch torchvision medmnist folktables
```

The current ERM and SVM loaders use these packages. The MST code additionally uses `networkx`, `tqdm`, `seaborn`, and `requests`; its graph loader imports `graph-tool`, which may require a platform-specific installation outside pip. Install those only if running the MST experiments:

```bash
python -m pip install networkx tqdm seaborn requests
```


## Data

The ERM loaders fetch MNIST through `torchvision`, Wine Quality from the UCI repository, and ACS/Folktables data through `folktables` when needed. The first run therefore needs internet access. Data is stored under the relevant module's local `data/` directory and should not be committed. The loaders define the task and preprocessing—for example, the Wine binary target is quality score at least 6, and the MNIST experiments use the digit groups specified in their runner.


## Running the current ERM and SVM experiments

Run these commands from the repository root after activating `.venv`:

```bash
python ERM/run_ratio_plots.py
python ERM/run_d2d_utility.py
python ERM/run_newton_utility.py
python SVM/run_svm_experiments.py
```

The runner settings are currently defined near the top of each script:

- `ERM/run_ratio_plots.py`: Wine red, MNIST projected to 50 dimensions, and California ACS Income (2018, person survey); `n=10,000`; 25 values of `lambda` from `1e-6` to `1e-1`; seeds `42`–`46`; `B_x=1`, `R_w=1`, `epsilon=1`, `delta=1e-9`, and `sigma=0.1`.
- `ERM/run_d2d_utility.py`: MNIST projections of 15, 20, 25, and 50 dimensions; `n=10,000`; five trials; `lambda` values `1e-6` through `1e-1` by decades; 200 iteration counts from 1 to 1000; `epsilon=1`, `delta=1e-9`, `B_x=1`, and `R_w=1`; both LogLoss and MSE.
- `ERM/run_newton_utility.py`: the same four MNIST projections and `n=10,000`; five trials; 25 log-spaced `lambda` values from `1e-6` to `1e-1`; `epsilon=1`, `delta=1e-9`, `B_x=1`, and local-radius factor 2.
- `SVM/run_svm_experiments.py`: Wine red, MNIST RP50, and California ACS Income; linear and RBF kernels; `C=1e6`; 25 trials; subsample fractions from 0.1 to 0.6; maximum reference data size 20,000; seed 0. Further settings, including margin/separability checks, are recorded in the runner and each run's `settings.json`.

