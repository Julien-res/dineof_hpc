# DINEOF-HPC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24083994.svg)](https://doi.org/10.5281/zenodo.24083994) ![GitHub License](https://img.shields.io/badge/license-GPL--3.0-blue) ![Python Version](https://img.shields.io/badge/python-3.9%E2%80%933.14-blue)

DINEOF-HPC is a high-performance Python implementation of the Data Interpolating Empirical Orthogonal Functions (DINEOF) algorithm, optimized for massive geophysical and satellite datasets on High-Performance Computing (HPC) architectures.

The project originated as a fork of [`pyDINEOF`](https://github.com/acoque/pyDINEOF) (Arthur Coqué, RBINS, 2025), a 100% Python translation of canonical [DINEOF](https://github.com/aida-alvera/DINEOF) (GHER). It has since been re-engineered into a scalable engine for high-resolution satellite rasters (Sentinel-2, Landsat...), which hit two ceilings with canonical DINEOF: **memory allocation** (dense spatial mode matrices $U \in \mathbb{R}^{M \times k}$ trigger OOM failures above $10^{7}$ pixels) and **variance attenuation** (truncated EOF iterations regress to the mean and underestimate extremes of high-dynamic-range tracers such as SPM, POC or Chlorophyll-a). DINEOF-HPC addresses them through in-place chunked updates, streaming Optimal Interpolation error, isolated-point cross-validation and local residual objective analysis, detailed in [Core Formulations](#core-formulations).

---

## Table of Contents
- [Differences from DINEOF GHER and pyDINEOF](#differences-from-dineof-gher-and-pydineof)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Minimal In-Memory Example (NumPy Array)](#1-minimal-in-memory-example-numpy-array)
  - [2. NetCDF Pipeline](#2-netcdf-pipeline)
  - [3. Batch Execution on a Cluster](#3-batch-execution-on-a-cluster)
- [Variables and Data Conventions](#variables-and-data-conventions)
  - [Inputs](#inputs)
  - [Outputs](#outputs)
  - [Parameters](#parameters)
- [Recommendations and Known Limitations](#recommendations-and-known-limitations)
- [Benchmarks](#benchmarks)
- [Core Formulations](#core-formulations)
  - [1. Unconditionally Stable 1D Implicit Temporal Diffusion](#1-unconditionally-stable-1d-implicit-temporal-diffusion)
  - [2. Isolated-Point Hold-out & Pooled RMSE](#2-isolated-point-hold-out--pooled-rmse)
  - [3. In-Stream Optimal Interpolation (OI) Analysis Error](#3-in-stream-optimal-interpolation-oi-analysis-error)
  - [4. Exact Analytical Physical Error for Log-Normal Distributions](#4-exact-analytical-physical-error-for-log-normal-distributions)
  - [5. Tracer-Agnostic Quantile Mapping with KGE Arbitration Gate](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate)
  - [6. Local Residual Objective Analysis (Barnes Refinement)](#6-local-residual-objective-analysis-barnes-refinement)
  - [7. Space-Time Barnes Refinement](#7-space-time-barnes-refinement)
- [Implementation and Architecture](#implementation-and-architecture)
  - [Implementation Details](#implementation-details)
  - [Directory Structure](#directory-structure)
  - [Computational Dataflow](#computational-dataflow)
- [Tests](#tests)
- [Methodological References](#methodological-references)
- [License](#license)
- [Citation](#citation)

---

## Differences from DINEOF GHER and pyDINEOF

| Component | DINEOF GHER (reference) | Upstream `pyDINEOF` | DINEOF-HPC |
| :--- | :--- | :--- | :--- |
| **Spectral solver** | Restarted Lanczos via ARPACK. | Lanczos via `scipy.sparse.linalg.eigsh`. | Dense `scipy.linalg.eigh` with automatic crossover to Lanczos / Randomized SVD ([Implementation Details](#implementation-details)). |
| **Update and memory** | BLAS matrix-vector operations on 1D ocean vectors; double precision. | Explicit $U = A V \Sigma^{-1}$ and dense $U \Sigma V^{T}$ allocation. | In-place spatial chunking, multi-threaded Numba kernels, `float32`, 64-bit indexing, automatic thread selection. |
| **Covariance and missing data** | Direct products, 1D index arrays. | Full `np.dot`, global coordinate tuples. | $A^{T}A$ from row blocks with frozen spatial subsampling indices; boolean masks (1 byte per element). |
| **Temporal filter** | Explicit Laplacian diffusion (`dindiff.f`). | Diffusion with time delta cast to `uint16`. | Implicit, unconditionally stable diffusion, no CFL limit on irregular dates ([§1](#1-unconditionally-stable-1d-implicit-temporal-diffusion)); equivalent to GHER for the same `alpha`, `numit`. |
| **Cross-validation** | Isolated points or cloud-mask files; CV points left out of the final pass. | Random isolated pixels, restored for the final pass. | Isolated points as in GHER, CV points left out of the final pass (`exclude_cv_in_final=True`), size capped by `cv_max_points` ([§2](#2-isolated-point-hold-out--pooled-rmse)). |
| **Modes** | Singular values from the *filtered* covariance. | - | Same: modes scaled by $\sqrt{\lambda_{\text{filtered}}/v^{T}Bv}$; progressive mode-by-mode initialization, same `toliter` criterion and early stop. |
| **Uncertainty** | Companion routine (`dineof_error.f90`). | Not implemented. | Reduced-subspace OI error with automatic correlation length, and exact log-normal physical error ([§3](#3-in-stream-optimal-interpolation-oi-analysis-error), [§4](#4-exact-analytical-physical-error-for-log-normal-distributions)). |
| **Local structure** | Purely global EOF modes. | Purely global EOF modes. | Optional Barnes residual refinement, per date or space-time ([§6](#6-local-residual-objective-analysis-barnes-refinement), [§7](#7-space-time-barnes-refinement)). |
| **Quantile mapping** | Not implemented. | Not implemented. | Optional, off by default; no benefit demonstrated in our tests ([§5](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate)). |
| **Observations** | Substitution only. | Substitution only. | Strict gap-filling or seamless filtered EOF continuum (`keep_observations`). |

---

## Installation

**Requirements**: Python 3.9 to 3.14 (unit tests run on each; tested on Windows and Linux); `numpy`, `scipy`, `pandas`, `xarray`, `netCDF4`, `numba`, `joblib`, `scikit-learn`, `matplotlib`, `tqdm` (declared in `pyproject.toml`). `numba` compiles its kernels on first use, so the first call of `dineof(...)` in a session is slower.

```bash
git clone https://github.com/Julien-res/dineof_hpc.git
cd dineof_hpc

# conda / mamba (recommended): dependencies + editable install
conda env create -f environment.yaml
conda activate dineof_hpc

# or pip only, in a virtual environment
pip install -e .
```

Check the installation (21 tests, a few seconds; a harmless `RuntimeWarning` about `numpy.ndarray size changed` may appear):

```bash
python -c "import dineof_hpc; print(dineof_hpc.__version__)"
pip install pytest
pytest tests/
```

**Threads.** `dineof(...)` sizes its own thread pool (see `n_threads`) and keeps BLAS single-threaded, so `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` need not be set. Two optional environment variables are read: `NUMBA_NUM_THREADS` forces the thread count (and caps the pool); `DINEOF_ELEMS_PER_THREAD` sets the matrix elements per thread of the automatic rule (default `500000`).

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

Edit the paths in `src/dineof_hpc/example.py` (profiles `conservative` and `fast`) and submit it with your scheduler (OAR, Slurm...). The thread pool is sized from the problem and the CPUs available, so a plain call is enough; to impose a count, prefix it with `NUMBA_NUM_THREADS=32`:

```bash
python -u src/dineof_hpc/example.py
```

`JOBLIB_TEMP_FOLDER` can point to a fast scratch directory used by the parallel NetCDF reader. Keep `use_disk=False` (default) unless the matrix does not fit in RAM (`True` stores it in a `numpy.memmap`).

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


### Parameters

`dineof(...)` is defined in `core/main.py`, `run_reconstruction_pipeline(...)` in `core/io_utils.py`. Defaults are those of the function signatures. **Scope**: *both* = parameter of both functions (defaults differ where noted); *core* = parameter of `dineof`, reachable from the pipeline through `dineof_kwargs`; *pipeline* = pipeline only.

| Parameter | Scope | Type | Default | Role and advice |
| :--- | :--- | :--- | :--- | :--- |
| `data` | core | `ndarray` | *required* | 2D `(M, N)` or 3D `(time, lat, lon)` array with `NaN` for missing values. |
| `time_coords` | core | `ndarray` or `None` | `None` | Dates (datetime, timestamps or days). Needed for `alpha > 0` on irregular series. |
| `mask` | core | `ndarray` or `None` | `None` | 2D boolean domain mask `(lat, lon)`; required for 3D `data`. |
| `input_nc`, `mask_file`, `var_name`, `out_dir` | pipeline | `str` | *required* | Input NetCDF cube, NetCDF file with the variable `mask`, variable to reconstruct, output directory (created if needed). |
| `apply_log` | both | `bool` | core `False`, pipeline `True` | Reconstruct $\ln X$ and exponentiate afterwards; for strictly positive, log-normal tracers. |
| `use_climatology` | both | `bool` | core `False`, pipeline `True` | Remove the per-pixel temporal mean (instead of one global mean) before the decomposition. |
| `threshold` | both | `float` or `None` | `None` | Values $\le$ `threshold` become `NaN`. |
| `vmax` | pipeline | `float` or `None` | `None` | Upper physical bound: larger observations are set to `NaN` before the reconstruction and the output is bounded. |
| `min_valid_ratio` | pipeline | `float` | `0.01` | Dates with a smaller fraction of valid pixels are blanked and fully reconstructed (`0.0` keeps all dates). |
| `keep_observations` | core | `bool` | `True` | `True`: original valid values are kept, only gaps are filled. `False`: filtered low-rank field everywhere. |
| `exclude_cv_in_final` | core | `bool` | `True` | Leaves the CV points out of the final convergence pass, as DINEOF GHER does, so the fitted model is the one that was validated; observations are put back in the output when `keep_observations=True`. |
| `nev` | core | `int` or `None` | `None` | Ceiling of the rank search (must be `< N`). `None`: estimated from the explained variance and clipped to 6-30. When `meta['n_opt'] == nev` the CV optimum may be higher: increase `nev`. 30 is a common choice. |
| `toliter` | core | `float` | `1e-3` | Relative RMS convergence threshold, $\text{RMS}(dx)/\text{stdvini}$. |
| `nitemax` | core | `int` | `300` | Maximum iterations per mode. |
| `alpha` | core | `float` | `0.01` | Temporal Laplacian filter strength (days$^2$); `0` disables it. For `alpha > 0` each mode is scaled by $\sqrt{\lambda_{\text{filtered}}/v^{T}Bv}$, as in DINEOF GHER ([§1](#1-unconditionally-stable-1d-implicit-temporal-diffusion)). |
| `numit` | core | `int` | `3` | Number of diffusion passes on the covariance matrix. |
| `cv_ratio` | core | `float` | `0.02` | Fraction of valid observations withheld, as isolated points (capped at 5 % of them). |
| `cv_max_points` | core | `int` or `None` | `500_000` | Upper bound on the number of withheld points. |
| `cv_patience` | core | `int` | `3` | The rank search stops after `cv_patience` consecutive modes without a new CV minimum. DINEOF GHER needs three consecutive increases and resets its counter on a small dip, so it may go further on flat plateaus. |
| `min_improvement` | core | `float` | `0.0` | Minimum relative CV gain required to try another mode (`0` disables). |
| `cv_patch_size`, `n_cv_runs` | core | `int` or `None` | `None` | **Deprecated and ignored** (the 2D patch hold-out was removed). |
| `subsample_ratio` | core | `float` | `1.0` | Fraction of rows used to build the temporal covariance (`1.0` = exact). |
| `subsample_min` | core | `int` | `5_000_000` | Matrix size below which `subsample_ratio` is ignored. |
| `seed` | core | `int` or `None` | `42` | Seed of the hold-out selection and subsampling. |
| `verbose`, `debug` | core | `bool` | `True`, `False` | Print progress; collect timings in `meta['timings']`. |
| `use_disk` | both | `bool` | `False` | Store the working matrix in a temporary `numpy.memmap` (matrices larger than RAM). |
| `solver` | both | `str` | core `'dense'`, pipeline `'auto'` | `'dense'` (LAPACK), `'lanczos'`, `'randomized'`, or `'auto'` (dense up to $N^*(k) \approx \max(70, 10k)$ dates and for short series, Lanczos above). |
| `n_threads` | core | `int` or `None` | `None` | Threads of the updates and covariance product. `None`: `NUMBA_NUM_THREADS` if set, else one thread per `DINEOF_ELEMS_PER_THREAD` elements (default 500,000), capped by the larger of the physical cores and three quarters of the available CPUs. |
| `apply_qmap`, `qmap_nb_quantiles` | pipeline | `bool`, `int` | `False`, `100` | Quantile mapping (opt-in, no demonstrated benefit; [§5](#5-tracer-agnostic-quantile-mapping-with-kge-arbitration-gate)) and number of points of its transfer curve. |
| `apply_local_refinement` | pipeline | `bool` | `False` | Barnes refinement of the gaps ([§6](#6-local-residual-objective-analysis-barnes-refinement), [§7](#7-space-time-barnes-refinement)). |
| `local_refinement_mode` | pipeline | `str` | `'spatial'` | `'spatial'`: per-date 2D analysis. `'spacetime'`: 3D space-time analysis. |
| `local_refinement_sigma` | pipeline | `float`, list or `None` | `None` | Spatial bandwidth(s) in pixels. `None`: candidates arbitrated on the CV points ($[1.5, 2.5, 3.5, 5, 7]$ in `'spatial'`, $[1.5, 3.5, 7, 10, 15]$ in `'spacetime'` mode). |
| `local_refinement_sigma_t` | pipeline | `float`, list or `None` | `None` | Temporal bandwidth(s) in dates, `'spacetime'` only. `None`: $[0, 1, 2]$ arbitrated on the CV points. |
| `corr_length_km` | pipeline | `float`, `'auto'` or `None` | `'auto'` | Correlation length $L$ of the observation error in the OI error ([§3](#3-in-stream-optimal-interpolation-oi-analysis-error)). `'auto'` estimates it (50 km fallback); a number fixes it; `None` or `0` disables the correction (error far too optimistic on satellite data). |
| `dineof_kwargs` | pipeline | `dict` or `None` | `None` | Any `dineof(...)` parameter (`nev`, `toliter`, `alpha`, `cv_ratio`, `n_threads`, `keep_observations`...). Takes precedence over the pipeline's own `use_disk` and `solver`; refinement options given here also win. |

When the refinement is on, the core engine runs with `keep_observations=False` and the genuine observations are restored afterwards; the `keep_observations` value in `dineof_kwargs` applies to the final output.

---

## Recommendations and Known Limitations

| Situation | Recommendation |
| :--- | :--- |
| Daily or near-daily series (MODIS, OLCI, VIIRS...) | Keep `alpha=0.01`: without the temporal filter the cross-validation overfits high-order modes. |
| Sparse revisit (5 days or more) | The temporal filter has little effect; `alpha=0.01` can stay as a universal default. |
| Rank search | `nev=30` for series of a few hundred dates or more (`nev=None` estimates 6-30). Compare `meta['n_opt']` with `nev`: if equal, the ceiling cut the search. `cv_patience=3` is the default; a larger value explores flat CV plateaus further at the cost of more modes. |
| Gap structure | Consider the space-time Barnes refinement (`apply_local_refinement=True, local_refinement_mode='spacetime'`); off by default, adds computing time and a few `float32` copies of the grid on large cubes. |
| Very large grids | Leave `n_threads=None`; use `subsample_ratio` (e.g. `0.1`) only if the covariance product dominates the run time. |
| Uncertainty | Use `error` / `error_rel` as a map of where the data constrain the reconstruction, not as a calibrated error bar. |
| Quantile mapping | Keep `apply_qmap=False` unless validated on your own data. |

Known limitations:

* **Rank selection.** Isolated-point cross-validation does not target extrapolation inside large gaps: it can select a rank that is too high or too low for wide contiguous gaps, and on a flat CV curve $k$ depends on the hold-out sample, the seed and `cv_patience`. `nev` bounds $k$.
* **Variance loss.** Truncated EOF reconstructions retain less variance than the data inside large gaps, and the CV RMSE of isolated points is optimistic for wide cloud gaps.
* **OI error is an indicator**, not an error bar: on held-out pixels it is optimistic even with `corr_length_km='auto'`, and far too small without the correlation-length correction.
* **`alpha` is a constant** chosen by the user, not cross-validated; the best value depends on the temporal sampling.
* **Threads.** The automatic thread count was calibrated on one machine; other architectures may benefit from tuning `n_threads` or `NUMBA_NUM_THREADS`.
* **Scope of validation.** Tested on chlorophyll-*a* and turbidity series; no sea-surface temperature or multivariate case.

---

## Benchmarks

Comparisons with DINEOF GHER (equivalence, run times, accuracy on independent cloud masks, effect of the space-time Barnes refinement) will be added to this section after publication.

---

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

The correction is applied in the working space (natural-log space when `apply_log=True`, which keeps the tracer positive after exponentiation):

$$Y_{\text{final}}(m, t) = Y_{\text{rec}}(m, t) + r_t(m) \cdot \left[\mathcal{Q}_{\text{obs}}\left(\mathcal{F}_{\text{rec}}\left(Y_{\text{rec}}(m, t)\right)\right) - Y_{\text{rec}}(m, t)\right]$$

where $\mathcal{F}_{\text{rec}}$ is the empirical distribution of the reconstruction and $\mathcal{Q}_{\text{obs}}$ the quantile function of the observations.

**Operational Safeguards:**

* **Data-leakage-free calibration**: Transfer functions $\mathcal{Q}$<sub>obs</sub> and $\mathcal{F}$<sub>rec</sub> are fitted strictly on valid training observations; withheld cross-validation points are excluded.
* **KGE arbitration gate**: Out-of-sample predictions are checked against withheld observations via the Kling-Gupta Efficiency. If performance degrades:

$$\text{KGE}_{\text{QMap}} \lt \text{KGE}_{\text{raw}}$$

quantile mapping is automatically deactivated to prevent artificial variance over-inflation (in our tests this gate was not a reliable predictor of the independent-test benefit).
* **Bounded extrapolation slopes**: High-quantile extrapolation is bounded ($\text{slope}_{\text{right}} \in [0.0, 1.0]$) in log space to avoid runaway under persistent cloud cover while preserving physical flood peaks.

### 6. Local Residual Objective Analysis (Barnes Refinement)
Global low-rank EOF modes tend to smooth sharp local gradients (fronts, plumes, bloom boundaries). An optional post-reconstruction Barnes analysis propagates the residual of the genuine observations, $R = X_{\text{obs}} - X_{\text{rec}}$ (in $\ln$ space when `apply_log=True`), into adjacent gaps. The residual and the observation mask $M_{\text{obs}}$ are convolved with a 2D Gaussian $G_\sigma$ ($\sigma$ in pixels), and the correction is the normalized ratio, applied only where the support is large enough:

$$\Delta_{\text{local}} = \frac{R * G_\sigma}{M_{\text{obs}} * G_\sigma} \quad \text{where } M_{\text{obs}} * G_\sigma \ge \epsilon_{\text{cutoff}}, \qquad X_{\text{refined}} = X_{\text{rec}} + \Delta_{\text{local}} \;\; (\text{multiplicative, } X_{\text{rec}} e^{\Delta_{\text{local}}}, \text{ in log mode})$$

When `local_refinement_sigma=None`, a grid $\sigma \in [1.5, 2.5, 3.5, 5, 7]$ pixels is arbitrated on the withheld CV points, evaluated only where the support condition holds. The best $\sigma^*$ is kept only if it improves the reachable RMSE by at least 1 % without degrading the global out-of-sample RMSE; otherwise the filter is switched off. Genuine observations are never modified, and inside large gaps (beyond about $3\sigma$ from any observation) the weight vanishes and the plain EOF reconstruction is recovered.

### 7. Space-Time Barnes Refinement
The space-time variant (`local_refinement_mode='spacetime'`) smooths the residual over **space and time** with one normalized 3D Gaussian ($\sigma$ pixels, $\sigma_t$ dates):

$$\Delta(x, y, t) = \frac{(R * G_{\sigma, \sigma_t})(x, y, t)}{(M_{\text{obs}} * G_{\sigma, \sigma_t})(x, y, t)}, \qquad \Delta = 0 \text{ where } (M_{\text{obs}} * G_{\sigma, \sigma_t}) \le \epsilon_{\text{cutoff}}$$

A gap pixel borrows the residual of observations close in space *and* time, which matters when a cloud-covered pixel is clear the day before or after. $\sigma_t = 0$ is the purely spatial analysis.

* **Selection**: all pairs of $\sigma \in [1.5, 3.5, 7, 10, 15]$ pixels and $\sigma_t \in [0, 1, 2]$ dates are scored on the withheld CV points (the residual field is built *without* them); the pair with the lowest CV RMSE is kept, and only if it lowers the CV RMSE of the unrefined reconstruction. The scales are global (one pair per run). The choice is printed in the console log.
* **Use**: `run_reconstruction_pipeline(..., apply_local_refinement=True, local_refinement_mode='spacetime')`, optionally fixing `local_refinement_sigma` and `local_refinement_sigma_t` (a float or a list of candidates). Functions: `arbitrate_spacetime_refinement` and `apply_spacetime_refinement` in `dineof_hpc.core.io_utils`.
* **Cost and limits**: the whole `(time, lat, lon)` cube is filtered (a few `float32` copies of the grid, 15 candidate pairs). The gain is largest where gaps are small or short-lived; inside wide, long-lasting gaps the kernel weight falls under the cutoff and the correction vanishes.

---

## Implementation and Architecture

DINEOF-HPC isolates the array algebra (`main.py`) from NetCDF handling and post-processing (`io_utils.py`), so the mathematical core has no file-format dependency.

### Implementation Details

* **Chunked spatial projection.** Since $U = A V \Sigma^{-1}$, the reconstruction is $U \Sigma V^{T} = A V V^{T}$. It is evaluated in place over spatial slices, $X_{\text{chunk}} = A_{\text{chunk}} V_{k} V_{k}^{T}$, by a multi-threaded Numba kernel that updates only the missing entries and never allocates the full $U \in \mathbb{R}^{M \times k}$.
* **Progressive mode initialization.** Iterations start at $k=1$ and each rank is initialized from the converged state of the preceding one, so unconverged high modes do not inject noise into the gaps.
* **Observation preservation.** `keep_observations=True` keeps observed pixels unchanged (strict gap-filling; a gradient seam may appear at cloud edges). `keep_observations=False` returns the filtered projection $\sum_{k=1}^{k_{\text{opt}}} u_{k} s_{k} v_{k}^{T}$ everywhere, filtering noise for seamless gradients (useful for front detection).
* **Automatic rank ceiling.** With `nev=None`, the ceiling comes from the trace of the zero-filled covariance $B_0$: $\sum_{i=1}^{k_{\text{target}}} \lambda_{i} \ge 0.95\,\text{Tr}(B_{0})$ gives $k_{\text{est}} = \text{clip}(k_{\text{target}} + 3, 6, 30)$.
* **Spectral solver (`solver='auto'`).** Temporal modes come from diagonalizing $B = A^{T} A \in \mathbb{R}^{N \times N}$. Dense LAPACK (`syevr`) costs $\mathcal{O}(N^{3})$ whatever $k$; ARPACK Lanczos and Randomized SVD cost $\mathcal{O}(k N^{2})$. `'auto'` uses the empirical crossover $N^*(k) \approx \max(70, 10k)$: dense below, Lanczos above.
* **Covariance and threads.** $A^{T}A$ is accumulated from row blocks computed concurrently with single-threaded BLAS, which also stays single-threaded during the EM loop so it does not compete with Numba; the thread count follows the problem size. With `subsample_ratio < 1` the subsampling indices are drawn once and frozen. The OI background diagonal is computed once outside the temporal loop.

### Directory Structure

```text
dineof_hpc/
├── pyproject.toml            # Build metadata & editable installation
├── environment.yaml          # Conda environment (dependencies + editable install)
├── README.md
├── LICENSE                   # GPL-3.0
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
Input NetCDF (3D) + 2D water mask
  -> io_utils.load_netcdf_masked   : sub-region cropping, low-coverage dates masked, (M, N) matrix
  -> main.dineof                   : isolated CV points, rank ceiling, progressive SVD updates,
                                     out-of-sample RMSE convergence and early stop, final restoration
  -> io_utils.run_reconstruction_pipeline :
       1. stationary OI background variance  2. optional Barnes refinement + CV gate
       3. optional Quantile Mapping + KGE gate  4. streamed NetCDF export with OI error
       5. CSV metrics and 6-panel dashboard
Output: DINEOF_Reconstructed_<var>.nc (data, error, error_rel), CSV tables, Dashboard_<var>.png
```

---

## Tests

```bash
pip install pytest
pytest tests/
```

The 21 tests cover the isolated hold-out selection and deprecated arguments, the exclusion of the CV points from the final pass, the OI analysis error (including the correlation-length estimate) and the space-time Barnes refinement.

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

If you use DINEOF-HPC, please cite the archived release (version 1.0.0, Zenodo DOI [10.5281/zenodo.24083994](https://doi.org/10.5281/zenodo.24083994)). The metadata are also available in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{masson_dineof_hpc_2026,
  author    = {Masson, Julien and Coqué, Arthur and Verpoorter, Charles and Jamet, Cédric},
  title     = {{DINEOF-HPC}: a high-performance Python/Numba implementation of {DINEOF} with space-time Barnes residual refinement},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24083994},
  url       = {https://github.com/Julien-res/dineof_hpc}
}
```

The benchmark scripts and the data used for the evaluation against DINEOF GHER are archived separately on Zenodo (DOI [10.5281/zenodo.24084121](https://doi.org/10.5281/zenodo.24084121), CC BY 4.0; access restricted until the associated article is published). The article reference will be added here upon publication.
