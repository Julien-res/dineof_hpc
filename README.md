# DINEOF-HPC

![GitHub License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)

DINEOF-HPC is a high-performance Python implementation of the Data Interpolating Empirical Orthogonal Functions (DINEOF) algorithm, optimized for massive geophysical and satellite datasets on High-Performance Computing (HPC) architectures.

The project originated as a fork of [`pyDINEOF`](https://github.com/acoque/pyDINEOF) (Arthur Coqué, RBINS, 2025), a 100% Python translation of canonical [DINEOF](https://github.com/aida-alvera/DINEOF) (GHER). It has since been re-engineered into an autonomous, scalable engine capable of handling high-resolution satellite rasters without out-of-memory crashes.

Applying canonical DINEOF to high-resolution missions (Sentinel-2, Landsat,...) runs into two ceilings: **memory allocation** (dense spatial mode matrices $U \in \mathbb{R}^{M \times k}$ trigger OOM failures above $10^{7}$ pixels) and **variance attenuation** (truncated EOF iterations act as a regression-to-the-mean filter, underestimating episodic extremes in high-dynamic-range tracers such as SPM, POC, or Chlorophyll-a). DINEOF-HPC addresses both through in-place chunked updates, an uncertainty-modulated Quantile Mapping scheme, streaming Optimal Interpolation error, isolated-point cross-validation, and local residual objective analysis - all detailed in the [Core Formulations](#core-formulations) section below.

---

## Table of Contents
- [Feature Matrix](#feature-matrix-upstream-pydineof-vs-dineof-gher)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Minimal In-Memory Example (NumPy Array)](#1-minimal-in-memory-example-numpy-array)
  - [2. NetCDF Pipeline](#2-netcdf-pipeline)
  - [3. Batch Execution on a Cluster](#3-batch-execution-on-a-cluster)
  - [4. Recommended Settings](#4-recommended-settings)
- [Variables and Data Conventions](#variables-and-data-conventions)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Parameters of `dineof(...)`](#parameters-of-dineof)
  - [Parameters of `run_reconstruction_pipeline(...)`](#parameters-of-run_reconstruction_pipeline)
- [Benchmarks](#benchmarks)
- [Equivalence with DINEOF GHER and Differences](#equivalence-with-dineof-gher-and-differences)
- [Core Formulations](#core-formulations)
  - [1. Unconditionally Stable 1D Implicit Temporal Diffusion](#1-unconditionally-stable-1d-implicit-temporal-diffusion)
  - [2. Isolated-Point Hold-out & Pooled RMSE](#2-isolated-point-hold-out--pooled-rmse)
  - [3. In-Stream Optimal Interpolation (OI) Analysis Error](#3-in-stream-optimal-interpolation-oi-analysis-error)
  - [4. Exact Analytical Physical Error for Log-Normal Distributions](#4-exact-analytical-physical-error-for-log-normal-distributions)
  - [5. Tracer-Agnostic Quantile Mapping with KGE Arbitration Gate](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate)
  - [6. Local Residual Objective Analysis (Barnes Refinement)](#6-local-residual-objective-analysis-barnes-refinement)
  - [7. Space-Time Barnes Refinement](#7-space-time-barnes-refinement)
- [Implementation Details](#implementation-details)
- [Software Architecture & Dataflow](#software-architecture--dataflow)
  - [Directory Structure](#directory-structure)
  - [Computational Dataflow](#computational-dataflow)
- [Known Limitations](#known-limitations)
- [Tests](#tests)
- [Methodological References](#methodological-references)
- [License](#license)
- [Citation](#citation)

---

## Feature Matrix: Upstream `pyDINEOF` vs. DINEOF GHER

| Component | DINEOF GHER (reference) | Upstream `pyDINEOF` | DINEOF-HPC (This Implementation) |
| :--- | :--- | :--- | :--- |
| **Spectral Solver** | Iterative restarted Lanczos via ARPACK (`dsaupd`/`dseupd`). | Iterative restarted Lanczos via `scipy.sparse.linalg.eigsh`. | Dense Hermitian eigensolver (`scipy.linalg.eigh`) with dynamic crossover to Lanczos / Randomized SVD - see [Implementation Details](#implementation-details) and [Benchmarks](#benchmarks). |
| **Memory & Iterative Update** | BLAS matrix-vector operations on 1D ocean coordinate vectors. | Explicit spatial mode computation $U = A V \Sigma^{-1}$, followed by dense $U \Sigma V^{T}$ allocation. | Sequential in-place spatial chunking with multi-threaded Numba JIT kernels - see [Implementation Details](#implementation-details). |
| **Covariance Construction** | Direct matrix products. | Full matrix multiplication $A^{T} A$ (`np.dot`). | Matrix product $A^{T} A$ (`np.matmul`) with precomputed, frozen spatial subsampling indices. |
| **Missing Data Indexing** | 1D index arrays of ocean points and gaps. | Global coordinate tuples via `np.isnan(X).nonzero()`. | Boolean masking combined with chunked BLAS projections (1 byte per spatiotemporal element). |
| **Temporal Diffusion Filter** | Discrete Laplacian diffusion with uniform step increments (`dindiff.f`). | Discrete diffusion with time delta in seconds cast to `uint16`. | Unconditionally stable 1D implicit diffusion (Thomas algorithm) - see [§1](#1-unconditionally-stable-1d-implicit-temporal-diffusion). |
| **Cross-Validation Sampling** | Cloud mask files or uniform random points. | Stochastic sampling of isolated pixels (`numpy.random.choice`). | Isolated-point hold-out, as in DINEOF GHER - see [§2](#2-isolated-point-hold-out--pooled-rmse). |
| **CV Sample Sizing** | User-defined parameter. | Continuous formula: $\min(0.01 M N + 40, 0.03 M N)$. | Adaptive sizing with an upper ceiling (`cv_max_points`, default $5 \times 10^{5}$), preventing extraction bottlenecks on rasters with $M N \gt 10^{9}$. |
| **Post-Validation Convergence** | Restores CV points and iterates until `toliter`. | Restores CV points and iterates until `toliter`. | Protocol maintained; out-of-sample predictions frozen at $k_{\text{opt}}$ before the final pass, preventing leakage. Optimal state cached on disk via `memmap`. |
| **Uncertainty Estimation (OI)** | Companion routine (`dineof_error.f90`). | Not implemented. | Analytical reduced-subspace OI analysis error - see [§3](#3-in-stream-optimal-interpolation-oi-analysis-error). |
| **Log-Normal Error Derivation** | Not implemented. | Not implemented. | Exact analytical formulation - see [§4](#4-exact-analytical-physical-error-for-log-normal-distributions). |
| **Distribution Mapping (Q-Map)** | Not implemented. | Not implemented. | Optional (off by default) in-stream Quantile Mapping with KGE arbitration gate; no benefit demonstrated in our tests - see [§5](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate). |
| **Local Spatial Autocorrelation**| Not implemented (purely global EOF modes). | Not implemented (purely global EOF modes). | Normalized 2D Barnes objective analysis of spatial observation residuals with CV support gating - see [§6](#6-local-residual-objective-analysis-barnes-refinement). |
| **Observation Preservation** | Binary substitution (`keep_obs=True` only). | Binary substitution (`keep_obs=True` only). | Dual-mode: strict gap-filling or seamless filtered EOF continuum - see [Implementation Details](#implementation-details). |

---

## Installation

**Requirements**: Python 3.10 or newer; `numpy`, `scipy`, `pandas`, `xarray`, `netCDF4`, `numba`, `joblib`, `scikit-learn`, `matplotlib`, `tqdm` (all declared in `pyproject.toml`). `numba` JIT-compiles the update kernels on first use, so the first call of `dineof(...)` in a session is slower than the following ones.

**With conda / mamba (recommended)** - `environment.yaml` installs the dependencies and the package in editable mode:

```bash
git clone https://github.com/Julien-res/dineof_hpc.git
cd dineof_hpc
conda env create -f environment.yaml
conda activate dineof_hpc
```

**With pip only** (inside a virtual environment or an existing conda environment):

```bash
git clone https://github.com/Julien-res/dineof_hpc.git
cd dineof_hpc
pip install -e .
```

**Check the installation**:

```bash
python -c "import dineof_hpc; from dineof_hpc.core.main import dineof; print(dineof_hpc.__version__)"
pip install pytest
pytest tests/
```

The test suite (21 tests) should pass in a few seconds. A `RuntimeWarning` about `numpy.ndarray size changed` raised while importing a compiled dependency is harmless.

**Threads.** `dineof(...)` sizes its own thread pool (see `n_threads` in the [parameter table](#parameters-of-dineof)); you do not need to set `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` for it (BLAS is kept single-threaded during the call). Two optional environment variables are read:

* `NUMBA_NUM_THREADS`: forces the number of compute threads (it also caps the pool).
* `DINEOF_ELEMS_PER_THREAD`: matrix elements per thread for the automatic rule (default `500000`).

---

## Quick Start

### 1. Minimal In-Memory Example (NumPy Array)

Calls the numerical core directly on a NumPy array that contains `NaN` values. The array has shape `(M, N)`: `M` spatial points (rows) by `N` dates (columns). This example was run as written and takes a few seconds.

```python
import numpy as np
from dineof_hpc.core.main import dineof

# 1. Synthetic dimensions: M points, N dates
M, N = 50_000, 120
time_coords = np.arange(N, dtype=np.float32)      # dates in days

# 2. Rank-3 synthetic field plus additive noise
rng = np.random.default_rng(42)
true_space = rng.standard_normal((M, 3), dtype=np.float32)
true_time = rng.standard_normal((3, N), dtype=np.float32)
X_truth = true_space @ true_time + 0.05 * rng.standard_normal((M, N), dtype=np.float32)

# 3. Remove 40 % of the values
cloud_mask = rng.uniform(0.0, 1.0, size=(M, N)) < 0.40
X_synthetic = X_truth.copy()
X_synthetic[cloud_mask] = np.nan

# 4. Reconstruct
X_rec, meta = dineof(
    data=X_synthetic,
    time_coords=time_coords,
    nev=10,
    toliter=1e-3,
    alpha=0.01,
    numit=3,
    subsample_ratio=0.2,
    verbose=False,
)

print(f"Optimal EOF rank retained : {meta['n_opt']}")
print(f"Out-of-sample CV RMSE     : {meta['valcopt']:.5f}")
print(f"Explained variance ratio  : {meta['var_ratio']:.2%}")
print(f"True RMSE on the gaps     : {np.sqrt(np.mean((X_rec[cloud_mask] - X_truth[cloud_mask]) ** 2)):.5f}")
```

Expected output (`seed=42`): 3 modes retained, a CV RMSE of about 0.194 and a true RMSE on the gaps of about 0.193, close to the CV estimate; the exact digits can differ slightly between platforms. `X_rec` has the shape of the input and contains no `NaN`. A 3D cube `(time, lat, lon)` is also accepted when a 2D boolean `mask` of shape `(lat, lon)` is given.

### 2. NetCDF Pipeline

`run_reconstruction_pipeline` reads a NetCDF cube and a water mask, reconstructs, and writes the reconstruction, its OI error and diagnostics. The script below builds a small synthetic dataset (so that it can be run anywhere) and then calls the pipeline with the space-time Barnes refinement. It was run as written (a few seconds).

```python
import numpy as np
import xarray as xr
from dineof_hpc.core.io_utils import run_reconstruction_pipeline

# --- Synthetic input: 120 daily scenes on a 40 x 50 grid, 40 % missing ---
rng = np.random.default_rng(0)
T, Y, X = 120, 40, 50
lat = np.linspace(50.0, 51.0, Y)
lon = np.linspace(1.0, 2.5, X)
time = np.datetime64("2022-01-01") + np.arange(T).astype("timedelta64[D]")
yy, xx = np.meshgrid(np.linspace(0, 1, Y), np.linspace(0, 1, X), indexing="ij")
field = np.exp(1.0 + np.stack([np.sin(2 * np.pi * t / 30 + 3 * xx) * yy + 0.3 * np.cos(t / 7.0) * xx
                               for t in range(T)]) + 0.05 * rng.standard_normal((T, Y, X)))
field = field.astype("float32")
field[rng.uniform(size=field.shape) < 0.4] = np.nan          # missing = NaN
mask = np.ones((Y, X), dtype="int8")                         # 1 = water, 0 = excluded
mask[:5, :5] = 0

xr.Dataset({"chl": (("time", "lat", "lon"), field, {"units": "mg m-3"})},
           coords={"time": time, "lat": lat, "lon": lon}).to_netcdf("input.nc")
xr.Dataset({"mask": (("lat", "lon"), mask)}, coords={"lat": lat, "lon": lon}).to_netcdf("mask.nc")

# --- Reconstruction ---
run_reconstruction_pipeline(
    input_nc="input.nc",
    mask_file="mask.nc",
    var_name="chl",
    out_dir="output/",
    apply_log=True,                         # log-normal tracer (chlorophyll, SPM...)
    use_climatology=True,
    apply_local_refinement=True,            # Barnes refinement, selected by cross-validation
    local_refinement_mode="spacetime",
    dineof_kwargs={"nev": 20, "verbose": False},
)
```

`output/` then contains `DINEOF_Reconstructed_chl.nc` (variables `data`, `error`, `error_rel`), `Error_Matrix_chl.csv`, `Error_Profile_chl.csv` and `Dashboard_chl.png` (see [Outputs](#outputs)). On this toy dataset the refinement gate reports a negligible gain (the synthetic field is almost exactly low-rank).

For your own data, replace `var_name` and the paths, keep `apply_log=True` only for strictly positive, roughly log-normal variables, and see [Inputs](#inputs) for the file layout. `src/dineof_hpc/example.py` is a ready-to-edit script with a `conservative` and a `fast` profile.

### 3. Batch Execution on a Cluster

Edit the paths in `src/dineof_hpc/example.py`, then submit it with your scheduler (OAR, Slurm...). The package sizes its thread pool from the problem and the CPUs it is allowed to use, so a plain call is enough:

```bash
python -u src/dineof_hpc/example.py
```

To impose a thread count (for instance to match the cores granted by the scheduler):

```bash
NUMBA_NUM_THREADS=32 python -u src/dineof_hpc/example.py
```

`JOBLIB_TEMP_FOLDER` can point to a fast scratch directory used by the parallel NetCDF reader. On nodes with a lot of memory keep `use_disk=False` (the default); `use_disk=True` stores the working matrix in a `numpy.memmap` and is only meant for matrices larger than the RAM.

### 4. Recommended Settings

These recommendations come from our tests on coastal series (daily to 5-day sampling, 4 km to 60 m); see [Benchmarks](#benchmarks).

| Situation | Recommendation |
| :--- | :--- |
| Daily or near-daily series (MODIS, OLCI, VIIRS...) | Keep `alpha=0.01`: without the temporal filter the cross-validation overfits high-order modes. |
| Sparse revisit (5 days or more: Sentinel-2, Landsat) | The temporal filter has little effect; `alpha=0.01` can be kept as a universal default. |
| Rank search | Use `nev=30` for series of a few hundred dates or more; `nev=None` estimates a ceiling between 6 and 30. Always compare `meta['n_opt']` with `nev`: if they are equal the search was cut by the ceiling. |
| Early stopping | `cv_patience=3` (default). A larger value explores a flat CV plateau further, at the cost of more modes. |
| Gap structure | Consider the space-time Barnes refinement (`apply_local_refinement=True, local_refinement_mode='spacetime'`), which uses the residuals of nearby observations; it is off by default and adds computing time on large grids. |
| Very large grids | Leave `n_threads=None`; set `subsample_ratio` (for instance `0.1`) only when the covariance product dominates the run time; `subsample_min` (default 5 000 000 elements) is the size below which it is ignored. |
| Uncertainty | Use `error` / `error_rel` as a map of where the reconstruction is constrained by the data, not as a calibrated error bar (see [Known Limitations](#known-limitations)). |
| Quantile mapping | `apply_qmap=False` (default): no measurable benefit in our tests. |

---

## Variables and Data Conventions

### Inputs

**Core engine `dineof(...)`**

| Object | Shape / type | Convention |
| :--- | :--- | :--- |
| `data` (2D) | `(M, N)` array | `M` spatial points (rows) by `N` dates (columns). `NaN` marks a missing value. The input is copied and handled in `float32`. |
| `data` (3D) | `(N, Y, X)` array = `(time, lat, lon)` | Requires `mask`. The output has the same shape; points outside `mask` stay `NaN`. |
| `mask` | `(Y, X)` boolean array | `True` for the pixels to reconstruct (water). If omitted, pixels that have at least one valid value are used. |
| `time_coords` | length `N`, datetime or numeric | Dates. Numeric values are read as days; datetimes are converted to decimal days. They define the time steps of the temporal filter, so `alpha` is in days$^2$. |
| Values `<= threshold` | | Set to `NaN` before the optional log transform. With `apply_log=True`, values `<= 0` are also treated as missing. |

The call fails with `ValueError` if fewer than 100 valid observations remain or if `nev >= N`. When `use_climatology=True`, the per-pixel temporal mean is removed before the decomposition and added back afterwards; otherwise one global mean is used.

**Pipeline `run_reconstruction_pipeline(...)`**

| File | Layout |
| :--- | :--- |
| `input_nc` | NetCDF with a variable `var_name` of dimensions `(time, lat, lon)` and coordinate variables named `time`, `lat` and `lon` (1D, or 2D for curvilinear grids). Missing values: `NaN` or the variable `_FillValue`. The attribute `units` is copied to the output. |
| `mask_file` | NetCDF with a 2D variable named `mask`; pixels equal to `1` are reconstructed. Either on the grid of `input_nc`, or a sub-region of it with its own `lat`/`lon` (the pipeline then aligns the two by coordinates). The cube is cropped to the bounding box of the mask. |

### Outputs

**`dineof(...)` returns `(X_out, meta)`.** `X_out` has the shape and units of the input (original observations preserved when `keep_observations=True`). Main entries of `meta`:

| Key | Content |
| :--- | :--- |
| `n_opt` | Number of EOF modes retained ($k_{\text{opt}}$). Compare with `nev_used`. |
| `nev_used` | Ceiling of the rank search that was actually used. |
| `valcopt`, `sigma_o` | Pooled CV RMSE at `n_opt` (in the working space: ln units if `apply_log=True`). |
| `valc` | CV RMSE for each explored rank (the CV curve). |
| `var_ratio`, `varini`, `varend` | Explained-variance diagnostics. |
| `u`, `s`, `v` | Spatial modes, singular values and temporal modes of the retained solution. |
| `cv_rows`, `cv_cols`, `cv_truth`, `cv_pred` | Position, truth and prediction of the withheld cross-validation points (physical units). |
| `obs_mask`, `clim`, `final_iters`, `disk_file`, `timings` | Observation mask, climatology, iterations of the final pass, memmap path (if `use_disk`), timings (if `debug=True`). |

**`run_reconstruction_pipeline(...)` returns `None`** and writes into `out_dir`:

| File | Content |
| :--- | :--- |
| `DINEOF_Reconstructed_<var>.nc` | `data`: reconstructed variable in physical units. `error`: OI error as a standard deviation in physical units ([§3](#3-in-stream-optimal-interpolation-oi-analysis-error), [§4](#4-exact-analytical-physical-error-for-log-normal-distributions)). `error_rel`: relative error $r_t(m) = P_t(m)/B(m) \in [0, 1]$ (0 = fully constrained by the data, 1 = no more information than the background). Dimensions `(time, lat, lon)`; `time` is stored in seconds since 1970-01-01. |
| `Error_Matrix_<var>.csv` | One row of metrics: `Modes_Optimum`, `Variance_Initial`, `Variance_Reconstructed`, `Variance_Ratio`, `Scale_Mode`, `Sigma_o_ln`, `Sigma_o_log10`, `RMSE_Transformed`, `Bias_Transformed`, `R2_Transformed`, `RMSE_Physical`, `Bias_Physical`, `OI_Mean_Error`. The `*_Transformed` metrics are computed on the CV points in the log scale when `apply_log=True`. |
| `Error_Profile_<var>.csv` | Error profile across concentration deciles. |
| `Dashboard_<var>.png` | 6-panel figure: hexbin scatter (truth vs. prediction), error across concentration deciles, mode-selection curve, maps of the mean analysis error and of the mean relative error, temporal error evolution. |
| `QMap_Transfer_Curve_<var>.png` | Only with `apply_qmap=True`. |

The selection made by the Barnes refinement gate (pair $\sigma$, $\sigma_t$, and whether the refinement was kept) is printed in the console log.

### Parameters of `dineof(...)`

Defined in `core/main.py`. All defaults below are read from the function signature.

| Parameter | Type | Default | Role and advice |
| :--- | :--- | :--- | :--- |
| `data` | `ndarray` | *required* | 2D `(M, N)` or 3D `(time, lat, lon)` array with `NaN` for missing values. |
| `time_coords` | `ndarray` or `None` | `None` | Dates (datetime, timestamps or days). Needed for `alpha > 0` on irregular series. |
| `mask` | `ndarray` or `None` | `None` | 2D boolean domain mask `(lat, lon)`; required for 3D `data`. |
| `apply_log` | `bool` | `False` | Reconstruct $\ln X$ and exponentiate afterwards. Use for strictly positive, log-normal tracers. |
| `use_climatology` | `bool` | `False` | Remove the per-pixel temporal mean (instead of one global mean) before the decomposition. |
| `keep_observations` | `bool` | `True` | `True`: original valid values are kept, only gaps are filled. `False`: returns the filtered low-rank field everywhere. |
| `exclude_cv_in_final` | `bool` | `True` | Leaves the CV points out of the final convergence pass, as DINEOF GHER does, so the fitted model is the one that was validated; the genuine observations are put back in the output when `keep_observations=True`. `False` restores them as data before the final pass. |
| `threshold` | `float` or `None` | `None` | Values $\le$ `threshold` become `NaN`. |
| `nev` | `int` or `None` | `None` | Ceiling of the rank search (must be `< N`). `None`: estimated from the explained variance and clipped to 6-30. It bounds $k$: when `meta['n_opt'] == nev`, the CV optimum may be higher; increase `nev`. A value of 30 is a common choice. |
| `toliter` | `float` | `1e-3` | Relative RMS convergence threshold of the iteration, $\text{RMS}(dx)/\text{stdvini}$. |
| `nitemax` | `int` | `300` | Maximum iterations per mode. |
| `alpha` | `float` | `0.01` | Strength of the temporal Laplacian filter (days$^2$). `0` disables it. For `alpha > 0` each mode is scaled by $\sqrt{\lambda_{\text{filtered}}/v^{T}Bv}$, the DINEOF GHER convention ([§1](#1-unconditionally-stable-1d-implicit-temporal-diffusion)). |
| `numit` | `int` | `3` | Number of diffusion passes applied to the covariance matrix. |
| `cv_ratio` | `float` | `0.02` | Fraction of the valid observations withheld, as isolated points, for cross-validation (capped at 5 % of them). |
| `cv_patch_size` | `int` or `None` | `None` | **Deprecated and ignored** (the 2D patch hold-out was removed; isolated points select the same rank as DINEOF GHER). |
| `cv_max_points` | `int` or `None` | `500_000` | Upper bound on the number of withheld points. |
| `cv_patience` | `int` | `3` | The rank search stops after `cv_patience` consecutive modes without a new CV minimum. DINEOF GHER instead needs three consecutive increases and resets its counter on a small dip, so it may go further on flat plateaus. |
| `n_cv_runs` | `int` or `None` | `None` | **Deprecated and ignored.** |
| `min_improvement` | `float` | `0.0` | Minimum relative CV gain required to try another mode (`0` disables this criterion). |
| `subsample_ratio` | `float` | `1.0` | Fraction of the rows used to build the temporal covariance (`1.0` = exact). |
| `subsample_min` | `int` | `5_000_000` | Matrix size below which `subsample_ratio` is ignored. |
| `seed` | `int` or `None` | `42` | Seed of the hold-out selection and of the subsampling. |
| `verbose` | `bool` | `True` | Print progress. |
| `use_disk` | `bool` | `False` | Store the working matrix in a temporary `numpy.memmap`; for matrices that do not fit in RAM. |
| `solver` | `str` | `'dense'` | `'dense'` (LAPACK), `'lanczos'`, `'randomized'`, or `'auto'` (dense up to $N^*(k) \approx \max(70, 10k)$ dates and for short series, Lanczos above). Note: `run_reconstruction_pipeline` defaults to `'auto'`. |
| `n_threads` | `int` or `None` | `None` | Threads of the updates and of the covariance product. `None`: `NUMBA_NUM_THREADS` if set, otherwise one thread per `DINEOF_ELEMS_PER_THREAD` matrix elements (default 500,000), capped by the larger of the physical cores and three quarters of the CPUs available to the process. BLAS stays single-threaded during the call. |
| `debug` | `bool` | `False` | Collect timings in `meta['timings']` (and print them if `verbose`). |

### Parameters of `run_reconstruction_pipeline(...)`

Defined in `core/io_utils.py`.

| Parameter | Type | Default | Role and advice |
| :--- | :--- | :--- | :--- |
| `input_nc` | `str` | *required* | Input NetCDF cube. |
| `mask_file` | `str` | *required* | NetCDF file with the variable `mask`. |
| `var_name` | `str` | *required* | Variable to reconstruct. |
| `out_dir` | `str` | *required* | Output directory (created if needed). |
| `threshold` | `float` or `None` | `None` | Values $\le$ `threshold` become `NaN` on loading. |
| `vmax` | `float` or `None` | `None` | Upper physical bound: larger observations are set to `NaN` before the reconstruction, and the output is bounded. |
| `min_valid_ratio` | `float` | `0.01` | Dates with a smaller fraction of valid pixels are blanked and fully reconstructed (`0.0` keeps all dates as they are). |
| `apply_log` | `bool` | `True` | Work on $\ln X$. Set `False` for variables that can be zero or negative. |
| `use_climatology` | `bool` | `True` | Per-pixel centering. |
| `apply_qmap` | `bool` | `False` | Quantile mapping (opt-in, no demonstrated benefit; [§5](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate)). |
| `qmap_nb_quantiles` | `int` | `100` | Number of points of the quantile transfer curve. |
| `apply_local_refinement` | `bool` | `False` | Barnes refinement of the gaps ([§6](#6-local-residual-objective-analysis-barnes-refinement), [§7](#7-space-time-barnes-refinement)). Recommended for coastal imagery. |
| `local_refinement_mode` | `str` | `'spatial'` | `'spatial'`: per-date 2D analysis. `'spacetime'`: 3D space-time analysis. |
| `local_refinement_sigma` | `float`, list or `None` | `None` | Spatial bandwidth(s) in pixels. `None`: candidates arbitrated on the CV points ($[1.5, 2.5, 3.5, 5, 7]$ in `'spatial'` mode, $[1.5, 3.5, 7, 10, 15]$ in `'spacetime'` mode). |
| `local_refinement_sigma_t` | `float`, list or `None` | `None` | Temporal bandwidth(s) in dates, `'spacetime'` only. `None`: $[0, 1, 2]$ arbitrated on the CV points. |
| `use_disk` | `bool` | `False` | Forwarded to `dineof` (ignored if also set in `dineof_kwargs`). |
| `solver` | `str` | `'auto'` | Forwarded to `dineof` (ignored if also set in `dineof_kwargs`). |
| `dineof_kwargs` | `dict` or `None` | `None` | Any parameter of `dineof(...)` (`nev`, `toliter`, `alpha`, `cv_ratio`, `cv_patience`, `subsample_ratio`, `n_threads`, `keep_observations`, `verbose`, `debug`...). The refinement options may also be given here and then take precedence. |
| `corr_length_km` | `float`, `'auto'` or `None` | `'auto'` | Correlation length $L$ of the observation error in the OI error ([§3](#3-in-stream-optimal-interpolation-oi-analysis-error)). `'auto'` estimates it (50 km fallback), a number fixes it, `None` or `0` disables the correction (the error is then far too optimistic on satellite data). |

When the refinement is on, the core engine is run with `keep_observations=False` and the genuine observations are restored after the refinement; the `keep_observations` value passed in `dineof_kwargs` applies to the final output.

---

## Benchmarks

DINEOF-HPC has been compared with the reference code DINEOF GHER on satellite chlorophyll-*a* and turbidity series at resolutions from 4 km to 60 m. The benchmark results (equivalence with DINEOF GHER, run times, accuracy on independent cloud masks, effect of the space-time Barnes refinement) will be added to this section after publication of the article (Masson et al., *DINEOF-HPC v1.0*, submitted to Geoscientific Model Development).

---

## Equivalence with DINEOF GHER and Differences

**Same as DINEOF GHER**
* Cross-validation on isolated hold-out points (`cv_ratio = 0.02`), with the CV points left out of the final convergence pass (`exclude_cv_in_final=True`); original observations restored in the output when `keep_observations=True`.
* Singular values taken from the *filtered* covariance (shrink factor $\sqrt{\lambda_{\text{filtered}}/v^{T}Bv}$), progressive mode-by-mode initialization, same relative-RMS convergence criterion (`toliter`) and early stop of the rank search.
* Temporal Laplacian filter: GHER applies an explicit diffusion scheme, DINEOF-HPC an implicit one; for the same `alpha` and `numit` the two filters are equivalent (see [§1](#1-unconditionally-stable-1d-implicit-temporal-diffusion)).

**Different**
* Implicit (Thomas algorithm) rather than explicit temporal diffusion, hence no CFL restriction on irregular dates.
* Single precision (`float32`) arrays and a Lanczos / dense eigensolver pair instead of double-precision ARPACK; the covariance may be spatially subsampled (`subsample_ratio`).
* Spatially chunked, multi-threaded (Numba) update; 64-bit indexing; automatic thread selection.
* In-stream OI analysis error (`corr_length_km='auto'`), and optional Barnes local refinement and Quantile Mapping (the latter disabled by default).

---

## Core Formulations

The seven formulations below are DINEOF-HPC's genuine methodological additions beyond canonical DINEOF and `pyDINEOF`. Implementation-only changes (chunked projection, progressive initialization, observation-preservation modes, automatic rank search) are covered more briefly in [Implementation Details](#implementation-details).

### 1. Unconditionally Stable 1D Implicit Temporal Diffusion
To filter temporal covariance matrices over irregular observation dates without CFL time-step clamping, an implicit backward Euler formulation is solved via the Thomas algorithm:

$$(I - \alpha L) B^{n+1} = B^{n}$$

Using variable intervals $\Delta t_{i} = t_{i+1} - t_{i}$, cell widths use half-cell boundaries ($\Delta t_{e,0} = 0.5 \Delta t_{0}$, $\Delta t_{e,N-1} = 0.5 \Delta t_{N-2}$, $\Delta t_{e,i} = 0.5(\Delta t_{i-1} + \Delta t_{i})$ elsewhere). Symmetrization $\tilde{B}_{\text{sym}} = 0.5(\tilde{B} + \tilde{B}^{T})$ guarantees self-adjointness.

As in the reference DINEOF, the singular value of a mode is the square root of the eigenvalue of the *filtered* matrix, with $U = AV/\|AV\|$. Each mode is therefore scaled by $\sqrt{\lambda_{\text{filtered}}/(v^{T}Bv)}$ (equal to 1 when `alpha = 0`) in both the EM update and the final reconstruction. Without this factor the iteration does not converge for `alpha > 0` and the cross-validation curve rises after a few modes; with it the curve is flat after its minimum, as in the reference code.

### 2. Isolated-Point Hold-out & Pooled RMSE
A reproducible random subset of the observed (pixel, date) pairs, `cv_ratio` of the valid observations (2 % by default, capped at 5 % of them and at `cv_max_points`), is withheld, as in DINEOF GHER. Optimal rank $k_{\text{opt}}$ minimizes the pooled RMSE:

$$\text{RMSE}_{\text{CV}}(k) = \sqrt{\frac{1}{N_{\text{eval}}} \sum_{i \in \text{withheld}} \left(X_a(i) - X_o(i)\right)^2}$$

Out-of-sample predictions are frozen at $k_{\text{opt}}$ prior to the final convergence pass with restored observations, preventing validation data leakage.

### 3. In-Stream Optimal Interpolation (OI) Analysis Error
Following Beckers et al. (2006), analysis error variance is calculated in the reduced EOF subspace ($k \times k$) rather than in spatial coordinates ($M \times M$):

$$A_t = \Lambda_k^{-1} + \frac{1}{\sigma_o^2} U_{\text{obs},t}^T U_{\text{obs},t}$$

where:

$$\Lambda_k = \frac{1}{N}\mathrm{diag}(s_1^2, \dots, s_k^2), \qquad \sigma_o = \text{RMSE}_{\text{CV}}(k_{\text{opt}})$$

The stationary background variance diagonal is precomputed once:

$$[B]_{mm} = \sum_{i=1}^k \lambda_i (U_k)_{mi}^2$$

During NetCDF export, local error variance and relative error ratio are evaluated on the fly:

$$[P_t]_{mm} = \sum_{i=1}^k \sum_{j=1}^k (U_k)_{mi} [A_t^{-1}]_{ij} (U_k)_{mj}, \qquad r_t(m) = \frac{[P_t]_{mm}}{[B]_{mm}} \in [0, 1]$$

$P_t$ is an absolute variance (in $\ln$ units when `apply_log=True`): the exported `error` is derived from it directly, and `error_rel` is only the ratio $P_t/B$.

**Spatially correlated observation error.** Neighbouring pixels of a satellite image share their error, so treating the pixels as independent makes the analysis error far too small. As in Beckers et al. (2006), the observation error variance entering $A_t$ is inflated by the number of pixels per independent error cell:

$$\mu^2_{\text{eff}} = \sigma_o^2 \cdot \max\left(\frac{L^2}{\Delta x \, \Delta y}, 1\right)$$

with the pixel area $\Delta x \, \Delta y$ deduced from the coordinates. The correlation length $L$ is set by `corr_length_km` in `run_reconstruction_pipeline`:

* `'auto'` (default): $L$ is estimated from the spatial autocorrelation of the residual *observation - retained EOF modes* (reconstruction without restored observations), averaged over up to 30 dates. With $r_{\text{eff}} = 1 + \sum \rho$ over the lags up to the radius where the radial autocorrelation falls below 0.05, $L = \sqrt{r_{\text{eff}} \, \Delta x \, \Delta y}$, bounded to [pixel size, 200 km]. If the estimate fails (unknown pixel size, too few dates, autocorrelation still above 0.05 at the largest usable lag), $L = 50$ km is used and a message says so.
* a number: $L$ in km, fixed by the user.
* `None` or `0`: no correction, $\mu^2_{\text{eff}} = \sigma_o^2$ (the error is then much too optimistic on satellite data).

> **Note**: the OI error is an *indicator* of where the reconstruction is more or less constrained by the data, not an exact error bar. On held-out test pixels the announced error is optimistic (observed coverage below the Gaussian expectation) and only moderately correlated with the actual error at pixel level.

### 4. Exact Analytical Physical Error for Log-Normal Distributions
For log-transformed environmental variables:

$$\ln X \sim \mathcal{N}\left(\mu, \sigma_{\text{ln}}^{2}\right)$$

first-order Taylor approximations underestimate physical dispersion under moderate-to-high uncertainty. Physical error is evaluated using the exact analytical identity:

$$\sigma_{\text{phys}}(m, t) = X_{\text{rec}}(m, t) \cdot \exp\left(\frac{\sigma_{\ln}^{2}(m, t)}{2}\right) \sqrt{\exp\left(\sigma_{\ln}^{2}(m, t)\right) - 1}$$

where the log-space error variance is scaled by the observational noise variance and the relative analysis error ratio:

$$\sigma_{\ln}^{2}(m, t) = \sigma_{o}^{2} \cdot r_{t}(m) = \sigma_{o}^{2} \cdot \frac{[P_{t}]_{mm}}{[B]_{mm}}$$

### 5. Tracer-Agnostic Quantile Mapping with KGE Arbitration Gate
Variance attenuation in unconstrained cloudy areas can be corrected via Quantile Mapping modulated by the local relative error ratio $r_t(m) \in [0, 1]$.

> **Status: optional, disabled by default (`apply_qmap=False`).** Independent tests on several zones, resolutions and seeds showed no measurable effect of Q-Map on the test RMSE, and the KGE gate accepted or rejected it essentially at random; without the $r_t$ weighting it degraded the reconstruction. Its usefulness is therefore not demonstrated: enable it explicitly and validate it on your own data before relying on it.

The formulation below applies when it is enabled:

**Linear scale (`apply_log=False`):**

$$Y_{\text{final}}(m, t) = Y_{\text{rec}}(m, t) + r_t(m) \cdot \left[\mathcal{Q}_{\text{obs}}\left(\mathcal{F}_{\text{rec}}\left(Y_{\text{rec}}(m, t)\right)\right) - Y_{\text{rec}}(m, t)\right]$$

**Logarithmic scale (`apply_log=True`):**

The modulated increment is evaluated directly in native logarithmic space before exponential inversion to prevent scale distortion:

$$\ln\left(Y_{\text{final}}(m, t)\right) = \ln\left(Y_{\text{rec}}(m, t)\right) + r_t(m) \cdot \Delta_{\text{qmap},\ln}(m, t)$$

$$Y_{\text{final}}(m, t) = \exp\left(\ln\left(Y_{\text{rec}}(m, t)\right) + r_t(m) \cdot \Delta_{\text{qmap},\ln}(m, t)\right)$$

where:

$$\Delta_{\text{qmap},\ln}(m, t) = \mathcal{Q}_{\text{obs}}\left(\mathcal{F}_{\text{rec}}\left(\ln Y_{\text{rec}}(m, t)\right)\right) - \ln Y_{\text{rec}}(m, t)$$

**Operational Safeguards:**

* **Data-leakage-free calibration**: Transfer functions $\mathcal{Q}$<sub>obs</sub> and $\mathcal{F}$<sub>rec</sub> are fitted strictly on valid training observations; withheld cross-validation points are excluded.
* **KGE arbitration gate**: Out-of-sample predictions are checked against withheld observations via the Kling-Gupta Efficiency. If performance degrades:

$$\text{KGE}_{\text{QMap}} \lt \text{KGE}_{\text{raw}}$$

quantile mapping is automatically deactivated to prevent artificial variance over-inflation (in our tests this gate was not a reliable predictor of the independent-test benefit).
* **Bounded extrapolation slopes**: High-quantile extrapolation is bounded ($\text{slope}_{\text{right}} \in [0.0, 1.0]$) in log space to avoid runaway under persistent cloud cover while preserving physical flood peaks.

### 6. Local Residual Objective Analysis (Barnes Refinement)
Because canonical DINEOF reconstructs missing data using global low-rank spatial modes $u_k$, it is spatially permutation-invariant and tends to smooth sharp, localized gradients (thermal fronts, river plumes, algal bloom boundaries). DINEOF-HPC introduces an optional post-reconstruction Barnes objective analysis step to propagate high-frequency residuals from measurements into adjacent unobserved gaps without destabilizing the numerical core:

1. **Observation Residual Computation**:
   Evaluates the local departure of genuine observations from the global EOF field:

   $$R(x, y, t) = X_{\text{obs}}(x, y, t) - X_{\text{rec}}(x, y, t)$$

2. **Normalized Gaussian Objective Analysis (Barnes Filter)**:
   The residual field and observation binary mask $M_{\text{obs}}$ are convolved with a 2D Gaussian kernel $G_\sigma$ of standard deviation $\sigma_{\text{spatial}}$ (in pixels):

   $$\tilde{R}(x, y, t) = (R * G_\sigma)(x, y, t), \qquad W(x, y, t) = (M_{\text{obs}} * G_\sigma)(x, y, t)$$

3. **Conditioned Gap Correction**:
   For unobserved pixels where surrounding observation density exceeds the cutoff threshold $W(x, y, t) \ge \epsilon_{\text{cutoff}}$:

   $$\Delta_{\text{local}}(x, y, t) = \frac{\tilde{R}(x, y, t)}{W(x, y, t)}$$

   In linear space (`apply_log=False`):

   $$X_{\text{refined}}(x, y, t) = X_{\text{rec}}(x, y, t) + \Delta_{\text{local}}(x, y, t)$$

   In logarithmic space (`apply_log=True`), corrections are applied multiplicatively to strictly enforce tracer positivity:

   $$X_{\text{refined}}(x, y, t) = X_{\text{rec}}(x, y, t) \cdot \exp\left(\Delta_{\text{local},\ln}(x, y, t)\right)$$

4. **Cross-Validation Arbitration Gate with Support Masking**:
   When `local_refinement_sigma=None`, an adaptive grid search ($\sigma \in [1.5, 2.5, 3.5, 5.0, 7.0]$ pixels) is arbitrated against withheld validation points without data leakage. Skill is evaluated strictly on reachable evaluation points ($W(x, y, t) \ge \epsilon_{\text{cutoff}}$). The optimal bandwidth $\sigma^*$ is retained only if it improves reachable RMSE by at least 1.0% without degrading global out-of-sample RMSE; otherwise, the filter is safely neutralized.

Genuine observations remain strictly untouched. Inside large cloud gaps where distance to measurements exceeds $3\sigma_{\text{spatial}}$, $W \to 0$, ensuring a smooth asymptotic fallback to the canonical global EOF reconstruction.

### 7. Space-Time Barnes Refinement
The Barnes analysis of §6 works date by date. The space-time variant (`local_refinement_mode='spacetime'`) smooths the observation residual over **both space and time** with one normalised 3D Gaussian kernel ($\sigma_{\text{space}}$ pixels, $\sigma_{t}$ dates):

$$\Delta(x, y, t) = \frac{(R * G_{\sigma, \sigma_t})(x, y, t)}{(M_{\text{obs}} * G_{\sigma, \sigma_t})(x, y, t)}, \qquad \Delta = 0 \text{ where } (M_{\text{obs}} * G_{\sigma, \sigma_t}) \le \epsilon_{\text{cutoff}}$$

with $R = X_{\text{obs}} - X_{\text{rec}}$ (in $\ln$ space when `apply_log=True`, applied multiplicatively). A gap pixel then borrows the residual of observations that are close in space *and* in time, which matters on series where a cloud-covered pixel is clear a day before or after.

* **Selection**: all pairs of $\sigma \in [1.5, 3.5, 7, 10, 15]$ pixels and $\sigma_t \in [0, 1, 2]$ dates are scored on the withheld cross-validation points (the residual field is built *without* them); the pair with the lowest CV RMSE is kept. The refinement is applied only if it lowers the CV RMSE of the unrefined reconstruction; otherwise it is switched off. $\sigma_t = 0$ is a purely spatial analysis.
* **Use**: `run_reconstruction_pipeline(..., apply_local_refinement=True, local_refinement_mode='spacetime')`, optionally fixing `local_refinement_sigma` and/or `local_refinement_sigma_t` (a float, or a list of candidates). The selected values and the gate decision are printed in the console log. Genuine observations are never modified by the refinement itself.
* **Cost and memory**: the whole `(time, lat, lon)` cube is filtered, so memory grows with the full grid size (a few float32 copies) and the candidate search costs 15 pairs of 3D Gaussian filters.
* **Limits**: the gain is largest where gaps are small or short-lived; inside wide, long-lasting gaps the kernel weight falls under the cutoff and the correction vanishes, leaving the plain EOF reconstruction. The correlation scales are global (one pair per run).
* **Functions**: `arbitrate_spacetime_refinement` (selection) and `apply_spacetime_refinement` (application) in `dineof_hpc.core.io_utils`.

---

## Implementation Details

* **Chunked spatial projection.** Since $U = A V \Sigma^{-1}$, reconstruction satisfies:

  $$U \Sigma V^{T} = (A V \Sigma^{-1}) \Sigma V^{T} = A V V^{T}$$

  DINEOF-HPC evaluates this sequentially across spatial slices ($m \ll M$) directly in place:

  $$X_{\text{chunk}} = A_{\text{chunk}} V_{k} V_{k}^{T}$$

  via a multi-threaded Numba JIT kernel (`prange`), never allocating the full $U \in \mathbb{R}^{M \times k}$ and updating only missing coordinates.
* **Progressive mode initialization.** To prevent unconverged higher modes from injecting high-frequency noise into gaps, iterations start strictly at $k=1$; each successive mode is initialized from the converged state of the preceding rank.
* **Observation preservation vs. spatial continuum.** `keep_observations=True` retains observed pixels unchanged (strict gap-filling; may show a gradient seam at cloud edges due to raw sensor noise vs. low-rank smoothing). `keep_observations=False` returns the full continuous projection $\sum_{k=1}^{k_{\text{opt}}} u_{k} s_{k} v_{k}^{T}$, filtering noise everywhere for seamless gradients - useful for hydrodynamic front detection.


* **Automatic rank search.** When `nev=None`, the search ceiling is estimated from the trace of the initial zero-filled covariance $B_0$:

  $$\sum_{i=1}^{k_{\text{target}}} \lambda_{i} \ge 0.95 \cdot \text{Tr}(B_{0}) \implies k_{\text{est}} = \text{clip}(k_{\text{target}} + 3, 6, 30)$$

---
* **Spectral solver selection (`solver='auto'`).** Temporal EOF modes come from diagonalizing $B = A^{T} A \in \mathbb{R}^{N \times N}$. LAPACK dense (`syevr`) performs a complete tridiagonal reduction in $\mathcal{O}(N^{3})$, invariant to $k$; ARPACK Lanczos (`eigsh`) and Randomized SVD instead solve reduced Krylov projections in $\mathcal{O}(k \cdot N^{2})$. `'auto'` applies the empirical crossover $N^*(k) \approx \max(70, 10k)$: dense LAPACK below, Lanczos above.
* **Precomputed covariance subsampling.** When `subsample_ratio < 1` the spatial subsampling indices are frozen once at startup rather than redrawn every iteration, cutting covariance cost on large grids.
* **Row-chunked Gram matrix and thread pools.** $A^{T}A$ is accumulated from row blocks computed concurrently with single-threaded BLAS, and BLAS is kept single-threaded during the EM loop so that it does not compete with the Numba kernels; the thread count follows the size of the problem (`n_threads`).
* **Stationary background for OI.** The background covariance diagonal is computed once outside the temporal loop.

---

## Software Architecture & Dataflow

DINEOF-HPC isolates raw array numerical algebra (`main.py`) from NetCDF dataset handling and post-processing (`io_utils.py`), guaranteeing zero file-format dependencies in the mathematical core.

### Directory Structure

```text
dineof_hpc/
├── pyproject.toml            # Build metadata & editable installation
├── environment.yaml          # Conda environment (dependencies + editable install)
├── README.md                 # Documentation
├── LICENSE                   # GPL-3.0
├── Optional/                 # Illustrations
├── tests/                    # pytest suite (21 tests)
└── src/
    └── dineof_hpc/
        ├── __init__.py
        ├── example.py        # Ready-to-edit batch script (conservative / fast profiles)
        ├── example_CALC.py   # Batch examples for large datasets on HPC nodes
        ├── Example_CALC_area.py
        └── core/
            ├── __init__.py
            ├── main.py       # Mathematical engine (NumPy / Numba JIT)
            └── io_utils.py   # NetCDF I/O, streaming OI error, Barnes refinement
```

### Computational Dataflow

```text
               +-------------------------------------------+
               |  Input NetCDF (3D) + 2D Water Domain Mask |
               +-------------------------------------------+
                                     |
                                     v
                 [ io_utils.load_netcdf_masked ]
                   - Coordinate-based sub-region geolocation
                   - Physical bounding-box cropping
                   - Low-coverage date masking (NaN preservation)
                   - Natural log transform & climatology
                                     |
                                     v
                        [ main.dineof (Core) ]
       +-------------------------------------------------------------+
       | 1. Isolated hold-out point selection (select_cv_points)     |
       | 2. Automatic rank search (_estimate_nev)                    |
       | 3. Progressive SVD updates                                  |
       | 4. Out-of-sample RMSE convergence & early stopping          |
       | 5. Final full-field restoration (keep_obs=True/False)       |
       +-------------------------------------------------------------+
                                     |
                                     v
                [ io_utils.run_reconstruction_pipeline ]
       +-------------------------------------------------------------+
       | 1. Precompute stationary background variance B_diag         |
       | 2. Optional Barnes refinement (2D per date or 3D space-time)|
       | 3. Cross-validation arbitration gate (support-masked CV)    |
       | 4. Optional: calibrate Quantile Mapping (apply_qmap=True)   |
       | 5. Evaluate KGE arbitration gate (deactivate if degraded)   |
       | 6. Stream NetCDF slices (reduced OI dual error P_t, comp=1) |
       | 7. Export validation metrics (CSV) and 6-panel dashboard    |
       +-------------------------------------------------------------+
                                     |
                                     v
               +-------------------------------------------+
               | Output NetCDF (data, error, error_rel)    |
               | Diagnostics CSV + Dashboard_<var>.png     |
               +-------------------------------------------+
```

---

## Known Limitations

* **`nev` bounds the rank.** The search never goes beyond `nev`. With `alpha > 0` on long daily series the CV optimum can lie at or beyond it. Check `meta['n_opt']` against `nev`.
* **Choice of the number of modes.** Isolated-point cross-validation does not target extrapolation inside large gaps: it can select a rank that is too high or too low for wide contiguous gaps. When the CV curve is flat the selected $k$ depends on the hold-out sample, the seed and `cv_patience`. The space-time Barnes refinement can compensate for part of this.
* **Variance loss in large gaps.** Truncated EOF reconstructions retain less variance than the data inside large gaps; the CV RMSE of isolated points is optimistic with respect to wide contiguous cloud gaps.
* **OI error is an indicator**, not an exact error bar: on held-out test pixels the announced error is optimistic, even with `corr_length_km='auto'`. Without the correlation-length correction (`corr_length_km=None`) it is far too small on satellite data.
* **Single `alpha`.** `alpha` is a user-chosen constant, not cross-validated; the best value depends on the temporal sampling.
* **Threads.** The automatic thread count was calibrated on one machine; on other architectures `n_threads` or `NUMBA_NUM_THREADS` may be worth tuning.
* **Memory.** The working matrix is `float32` of size $M \times N$ (plus a few copies); the Barnes space-time refinement filters the whole `(time, lat, lon)` cube and needs a few more `float32` copies of the grid.
* **Scope of the validation.** The method was tested on chlorophyll-*a* and turbidity series; no sea-surface temperature or multivariate case was tested.
* Optional Quantile Mapping showed no measurable benefit and is off by default.

---

## Tests

```bash
pip install pytest
pytest tests/
```

The 21 tests cover the isolated hold-out selection and the deprecated arguments, the exclusion of the CV points from the final pass, the OI analysis error (including the correlation-length estimate) and the space-time Barnes refinement. They run in a few seconds on a laptop.

---

## Methodological References

* **DINEOF Core Method**: Beckers, J. M., & Rixen, M. (2003). *EOF calculations and data filling from incomplete oceanographic datasets*. J. Atmos. Ocean. Technol., 20(12), 1839-1856.
* **Geophysical Applications**: Alvera-Azcarate, A., et al. (2005). *Reconstruction of incomplete oceanographic data sets using DINEOF*. Ocean Modelling, 9(4), 325-346.
* **Error Formulation (OI)**: Beckers, J. M., et al. (2006). *Error calculation on DINEOF data reconstruction using an optimal interpolation approach*. Ocean Science, 2(2), 79-99.
* **Temporal Filtering**: Alvera-Azcarate, A., et al. (2009). *The use of the temporal filter in DINEOF: analysis of a Gulf of Mexico SST data set*. Ocean Science, 5(4), 475-485.
* **Objective Analysis & Barnes Filter**: Barnes, S. L. (1964). *A technique for maximizing details in numerical weather map analysis*. J. Appl. Meteorol. Climatol., 3(4), 396-409.
* **Log-Normal Environmental Distributions**: Andersson, A. (2021). *Mechanisms for log-normal concentration distributions in the environment*. Scientific Reports, 11(1), 16418.
* **Numerical Linear Algebra**: Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

---

## License

Distributed under the terms of the **GNU General Public License version 3 (GPLv3)**.

---

## Citation

An article describing DINEOF-HPC and its comparison with DINEOF GHER is in preparation. Until it appears, please cite this repository (Masson, J., DINEOF-HPC, https://github.com/Julien-res/dineof_hpc).
