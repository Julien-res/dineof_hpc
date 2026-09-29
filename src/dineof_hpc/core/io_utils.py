# -*- coding: utf-8 -*-
"""I/O, uncertainty estimation, and post-processing for DINEOF-HPC.

The module connects the array-based numerical core to geophysical NetCDF files.
It provides batched loading, streaming reconstruction with optimal-interpolation
(OI) uncertainty, leakage-free empirical quantile mapping, local residual
objective analysis with cross-validation arbitration, and diagnostic report
generation. The numerical core remains in :mod:`dineof_hpc.core.main`.

Unless stated otherwise, matrices use the convention ``(space, time)`` and
gridded NetCDF variables use ``(time, lat, lon)``. Missing values are represented
by ``NaN``.
"""

from __future__ import annotations

import os
import time
import numpy as np
import xarray as xr
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from dineof_hpc.core.main import compute_dineof_oi_error_step, dineof
except ImportError:
    from main import compute_dineof_oi_error_step, dineof


def _interp_extrap(
    x: np.ndarray,
    xp: np.ndarray,
    fp: np.ndarray,
    tail_fraction: float = 0.05
) -> np.ndarray:
    """Interpolate a calibration curve and extrapolate its two tails safely.

    Interior values use :func:`numpy.interp`. Outside the calibration interval,
    a slope estimated from the corresponding tail is used and clipped to avoid
    implausible corrections.

    Parameters
    ----------
    x : np.ndarray
        Query coordinates.
    xp : np.ndarray
        Monotonically increasing calibration points.
    fp : np.ndarray
        Function values at calibration points.
    tail_fraction : float
        Fraction of outer calibration points used to estimate each tail slope.

    Returns
    -------
    np.ndarray
        Interpolated and extrapolated values.
    """
    y = np.interp(x, xp, fp)
    n_q = len(xp)
    k = max(1, int(n_q * tail_fraction))

    # Estimate a bounded slope from the lowest calibration points.
    dx_left = xp[k] - xp[0]
    slope_left = (fp[k] - fp[0]) / dx_left if dx_left > 1e-6 else 1.0
    slope_left = max(0.0, min(slope_left, 1.5))
    m_left = x < xp[0]
    if np.any(m_left):
        y[m_left] = fp[0] + slope_left * (x[m_left] - xp[0])

    # Use a stricter upper bound on the high tail to limit amplification.
    dx_right = xp[-1] - xp[-1 - k]
    slope_right = (fp[-1] - fp[-1 - k]) / dx_right if dx_right > 1e-6 else 1.0
    slope_right = max(0.0, min(slope_right, 1.0))
    m_right = x > xp[-1]
    if np.any(m_right):
        y[m_right] = fp[-1] + slope_right * (x[m_right] - xp[-1])

    return y


def _read_and_extract_batch(
    file_path: str,
    var_name: str,
    t_start: int,
    t_end: int,
    y_min: int,
    y_max: int,
    x_min: int,
    x_max: int,
    water_flat_crop: np.ndarray,
    fill_value: float | None,
    threshold: float | None,
    mmap_path: str,
    n_water: int,
    n_time: int
):
    """Worker task extracting a spatial slice directly into a memory-mapped array."""
    # Open dataset independently in read-only mode to enable parallel decompression
    with nc.Dataset(file_path, 'r') as ds:
        var = ds.variables[var_name]
        var.set_auto_mask(False)

        # Slice restricted spatial bounding box to skip unneeded terrestrial chunks
        block = var[t_start:t_end, y_min:y_max, x_min:x_max]

    cur_time = block.shape[0]
    # Flatten spatial axes to accelerate 1D linear coordinate indexing
    block_2d = block.reshape(cur_time, -1)
    extracted = block_2d[:, water_flat_crop]

    # Handle NetCDF fill value replacement
    if fill_value is not None and not np.isnan(fill_value):
        extracted[extracted == fill_value] = np.nan

    # Apply physical lower bound threshold
    if threshold is not None:
        extracted[extracted <= threshold] = np.nan

    # Open target memory-mapped array and write slice directly without IPC serialization
    X_target = np.memmap(mmap_path, dtype=np.float32, mode='r+', shape=(n_water, n_time))
    X_target[:, t_start:t_end] = extracted.T
    X_target.flush()


def _extract_coord_1d(ds: nc.Dataset, names: list[str]) -> np.ndarray | None:
    """Extract a 1D coordinate array from a NetCDF dataset by checking potential variable names."""
    for name in names:
        if name in ds.variables:
            arr = np.asarray(ds.variables[name][:])
            if arr.ndim == 2:
                return arr[:, 0] if name in ['lat', 'latitude', 'y', 'Y'] else arr[0, :]
            return arr
    return None


def load_netcdf_masked(
    file_path: str,
    mask_file: str,
    var_name: str,
    threshold: float | None = None,
    batch_size: int = 50,
    n_workers: int = 6
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a masked NetCDF variable by batches cropped to the mask bounding box.

    Extracts active water coordinates using bounding box cropping, raw array
    slicing with auto-masking disabled, and multi-process batch execution via
    memory-mapped buffers. Coordinates metadata and the 2D mask are cropped
    to the bounding box.

    Parameters
    ----------
    file_path : str
        Path to the primary NetCDF dataset file.
    mask_file : str
        Path to the 2D water domain mask NetCDF file.
    var_name : str
        Target geophysical variable name within ``file_path``.
    threshold : float or None, default=None
        Physical lower bound. Values less than or equal to this threshold are
        set to ``NaN``.
    batch_size : int, default=50
        Number of time slices processed per batch.
    n_workers : int, default=6
        Number of parallel worker processes for concurrent decompression.

    Returns
    -------
    X : np.ndarray
        2D spatiotemporal array of shape ``(n_water, n_time)`` and dtype ``float32``.
    mask_2d : np.ndarray
        Cropped 2D boolean domain mask where ``True`` indicates valid water pixels.
    coords_meta : dict
        Extracted coordinates metadata cropped to bounding box.
    """
    # 1. Ingest mask data and spatial coordinates
    with nc.Dataset(mask_file, 'r') as m_ds:
        mask_raw = m_ds.variables['mask'][:]
        if np.ma.isMaskedArray(mask_raw):
            mask_raw = mask_raw.filled(0)
        mask_raw = np.asarray(mask_raw == 1, dtype=bool)

        mask_lat = _extract_coord_1d(m_ds, ['lat', 'latitude', 'y', 'Y'])
        mask_lon = _extract_coord_1d(m_ds, ['lon', 'longitude', 'x', 'X'])

    # 2. Extract reference coordinate metadata from input file
    with xr.open_dataset(file_path) as ds_meta:
        full_lat = ds_meta.lat.values
        full_lon = ds_meta.lon.values
        time_vals = ds_meta.time.values
        units_val = ds_meta[var_name].attrs.get('units', 'a.u.')

    with nc.Dataset(file_path, 'r') as ds_in:
        var = ds_in.variables[var_name]
        var_shape = var.shape
        n_time = var_shape[0]
        fill_val = getattr(var, '_FillValue', None)

        chunks = var.chunking()
        if chunks and chunks != 'contiguous':
            time_chunk = chunks[0]
            if time_chunk > 0:
                batch_size = max(batch_size, time_chunk * max(1, batch_size // time_chunk))

    # 3. Align mask spatially with the input dataset
    if mask_raw.shape == var_shape[1:]:
        # Case A: Mask is defined on the master grid
        water_y, water_x = np.where(mask_raw)
        n_water = len(water_y)
        if n_water == 0:
            raise ValueError(f"No valid water pixels (mask == 1) found in {mask_file}")

        y_min, y_max = int(np.min(water_y)), int(np.max(water_y)) + 1
        x_min, x_max = int(np.min(water_x)), int(np.max(water_x)) + 1
        mask_2d = mask_raw[y_min:y_max, x_min:x_max]
        water_flat_crop = (water_y - y_min) * (x_max - x_min) + (water_x - x_min)
    else:
        # Case B: Mask is already a sub-regional crop (e.g., Tonle Sap 7540x6114)
        if mask_lat is None or mask_lon is None:
            raise ValueError(
                f"Shape mismatch: mask {mask_raw.shape} vs input {var_shape[1:]}, "
                f"and mask file '{mask_file}' lacks lat/lon coordinates for geographic alignment."
            )

        ref_lat = full_lat[:, 0] if full_lat.ndim == 2 else full_lat
        ref_lon = full_lon[0, :] if full_lon.ndim == 2 else full_lon

        # Locate latitude boundaries
        idx_y0 = int(np.argmin(np.abs(ref_lat - mask_lat[0])))
        idx_y1 = int(np.argmin(np.abs(ref_lat - mask_lat[-1])))
        if idx_y0 > idx_y1:
            mask_raw = np.flipud(mask_raw)
            y_min = idx_y1
        else:
            y_min = idx_y0

        # Locate longitude boundaries
        idx_x0 = int(np.argmin(np.abs(ref_lon - mask_lon[0])))
        idx_x1 = int(np.argmin(np.abs(ref_lon - mask_lon[-1])))
        if idx_x0 > idx_x1:
            mask_raw = np.fliplr(mask_raw)
            x_min = idx_x1
        else:
            x_min = idx_x0

        y_max = y_min + mask_raw.shape[0]
        x_max = x_min + mask_raw.shape[1]

        if y_max > var_shape[1] or x_max > var_shape[2]:
            raise ValueError(
                f"Aligned bounding box [{y_min}:{y_max}, {x_min}:{x_max}] exceeds "
                f"input grid shape {var_shape[1:]}."
            )

        mask_2d = mask_raw
        water_y, water_x = np.where(mask_2d)
        n_water = len(water_y)
        if n_water == 0:
            raise ValueError(f"No valid water pixels (mask == 1) found in {mask_file}")

        water_flat_crop = water_y * mask_2d.shape[1] + water_x

    coords_meta = {
        'time': time_vals,
        'lat': full_lat[y_min:y_max] if full_lat.ndim == 1 else full_lat[y_min:y_max, x_min:x_max],
        'lon': full_lon[x_min:x_max] if full_lon.ndim == 1 else full_lon[y_min:y_max, x_min:x_max],
        'units': units_val
    }

    # 4. Allocate shared temporary memory map
    scratch_dir = os.environ.get('JOBLIB_TEMP_FOLDER', '/dev/shm' if os.path.exists('/dev/shm') else '/tmp')
    mmap_path = os.path.join(scratch_dir, f"load_X_{os.getpid()}_{int(time.time() * 1000)}.dat")
    X_mmap = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=(n_water, n_time))

    batch_ranges = [
        (t, min(t + batch_size, n_time))
        for t in range(0, n_time, batch_size)
    ]

    # 5. Parallel extraction across batches
    Parallel(n_jobs=n_workers, backend='loky')(
        delayed(_read_and_extract_batch)(
            file_path=file_path,
            var_name=var_name,
            t_start=t_start,
            t_end=t_end,
            y_min=y_min,
            y_max=y_max,
            x_min=x_min,
            x_max=x_max,
            water_flat_crop=water_flat_crop,
            fill_value=fill_val,
            threshold=threshold,
            mmap_path=mmap_path,
            n_water=n_water,
            n_time=n_time
        )
        for t_start, t_end in tqdm(batch_ranges, desc="Loading NetCDF (parallel)", unit="batch")
    )

    X = np.array(X_mmap, dtype=np.float32)
    if hasattr(X_mmap, '_mmap') and X_mmap._mmap is not None:
        X_mmap._mmap.close()
    del X_mmap

    if os.path.exists(mmap_path):
        try:
            os.remove(mmap_path)
        except OSError:
            pass

    return X, mask_2d, coords_meta


def arbitrate_local_residual_refinement(
    X_rec: np.ndarray,
    X_raw_obs: np.ndarray,
    mask_2d: np.ndarray,
    dineof_meta: dict,
    candidate_sigmas: list[float] | None = None,
    decorrelation_cutoff: float = 0.05,
    is_log_scale: bool = False
) -> tuple[bool, float | None, np.ndarray]:
    """Arbitrate local residual refinement against reachable cross-validation points.

    Evaluates candidate spatial kernel bandwidths on withheld evaluation points
    that fall within the observational support of the Gaussian kernel (W >= cutoff),
    preventing large-patch geometric artifacts from diluting local filter skill.

    Parameters
    ----------
    X_rec : np.ndarray
        Reconstructed matrix of shape (n_water, n_time).
    X_raw_obs : np.ndarray
        Raw un-imputed observation matrix of shape (n_water, n_time) containing NaNs.
    mask_2d : np.ndarray
        2D boolean domain mask where True indicates valid water pixels.
    dineof_meta : dict
        Metadata dictionary containing 'cv_rows', 'cv_cols', 'cv_truth', 'cv_pred'.
    candidate_sigmas : list of float or None, default=None
        List of Gaussian sigma candidates (in pixels) to evaluate.
    decorrelation_cutoff : float, default=0.05
        Observation density threshold below which corrections are ignored.
    is_log_scale : bool, default=False
        Whether errors and residuals are evaluated in log-transformed space.

    Returns
    -------
    validated : bool
        True if local refinement improves out-of-sample RMSE on supported points.
    best_sigma : float or None
        Optimal spatial bandwidth in pixels, or None if arbitration failed.
    pred_cv_refined : np.ndarray
        Updated cross-validation predictions reflecting the optimal refinement.
    """
    cv_r = dineof_meta.get('cv_rows')
    cv_c = dineof_meta.get('cv_cols')
    truth_cv = dineof_meta.get('cv_truth')
    pred_cv = dineof_meta.get('cv_pred')

    if cv_r is None or len(cv_r) == 0 or truth_cv is None or pred_cv is None:
        print("   [Refinement Gate] No cross-validation points available. Skipping refinement.")
        return False, None, pred_cv

    if candidate_sigmas is None:
        candidate_sigmas = [1.5, 2.5, 3.5, 5.0, 7.0]

    # Clean finite held-out evaluation pairs
    if is_log_scale:
        valid_cv = np.isfinite(truth_cv) & np.isfinite(pred_cv) & (truth_cv > 0) & (pred_cv > 0)
        target_truth = np.log(truth_cv[valid_cv])
        target_raw = np.log(pred_cv[valid_cv])
    else:
        valid_cv = np.isfinite(truth_cv) & np.isfinite(pred_cv)
        target_truth = truth_cv[valid_cv]
        target_raw = pred_cv[valid_cv]

    if np.sum(valid_cv) < 30:
        print("   [Refinement Gate] Insufficient valid CV points (< 30). Skipping refinement.")
        return False, None, pred_cv

    sub_cv_r = cv_r[valid_cv]
    sub_cv_c = cv_c[valid_cv]
    unique_dates = np.unique(sub_cv_c)

    n_lat, n_lon = mask_2d.shape
    water_y, water_x = np.where(mask_2d)

    # Precompute per-date residual grids with held-out CV points masked out
    date_data = {}
    for t in unique_dates:
        raw_t = X_raw_obs[:, t].copy()
        rec_t = X_rec[:, t]

        cv_mask_t = (sub_cv_c == t)
        pts_cv_t = sub_cv_r[cv_mask_t]
        raw_t[pts_cv_t] = np.nan

        valid_obs = np.isfinite(raw_t)
        if np.count_nonzero(valid_obs) < 5:
            continue

        buf_obs = np.zeros((n_lat, n_lon), dtype=np.float32)
        buf_rec = np.zeros((n_lat, n_lon), dtype=np.float32)
        buf_mask = np.zeros((n_lat, n_lon), dtype=np.float32)

        buf_rec[water_y, water_x] = rec_t

        obs_y = water_y[valid_obs]
        obs_x = water_x[valid_obs]
        buf_obs[obs_y, obs_x] = raw_t[valid_obs]
        buf_mask[obs_y, obs_x] = 1.0

        res_2d = np.zeros((n_lat, n_lon), dtype=np.float32)
        if is_log_scale:
            obs_log = np.log(np.maximum(buf_obs[obs_y, obs_x], 1e-7))
            rec_log = np.log(np.maximum(buf_rec[obs_y, obs_x], 1e-7))
            res_2d[obs_y, obs_x] = obs_log - rec_log
        else:
            res_2d[obs_y, obs_x] = buf_obs[obs_y, obs_x] - buf_rec[obs_y, obs_x]

        date_data[t] = {
            'res_2d': res_2d,
            'mask_2d': buf_mask,
            'cv_mask_t': cv_mask_t,
            'pts_cv_t': pts_cv_t
        }

    if len(date_data) == 0:
        print("   [Refinement Gate] No evaluable CV dates with sufficient observations.")
        return False, None, pred_cv

    scale_label = "ln" if is_log_scale else "phys"
    print(f"\n   [Refinement Gate] Evaluating Barnes residual filter on active CV support...")

    best_gain_pct = 0.0
    best_sigma = None
    best_pred_cand = target_raw.copy()

    for sigma in candidate_sigmas:
        pred_cand = target_raw.copy()
        has_support = np.zeros(len(target_raw), dtype=bool)

        for t, d_info in date_data.items():
            res_conv = gaussian_filter(d_info['res_2d'], sigma=sigma, mode='constant', cval=0.0)
            w_conv = gaussian_filter(d_info['mask_2d'], sigma=sigma, mode='constant', cval=0.0)

            valid_w = w_conv > decorrelation_cutoff
            corr_2d = np.zeros((n_lat, n_lon), dtype=np.float32)
            corr_2d[valid_w] = res_conv[valid_w] / w_conv[valid_w]

            corr_1d = corr_2d[mask_2d]
            pts_t = d_info['pts_cv_t']
            cv_indices = np.where(d_info['cv_mask_t'])[0]

            support_1d = valid_w[mask_2d]
            has_support[cv_indices] = support_1d[pts_t]

            if is_log_scale:
                pred_cand[cv_indices] += np.clip(corr_1d[pts_t], -3.0, 3.0)
            else:
                pred_cand[cv_indices] += corr_1d[pts_t]

        n_supported = int(np.sum(has_support))
        if n_supported < 20:
            print(f"      Candidate sigma = {sigma:4.1f} px | Reachable CV points: {n_supported} (< 20, skipped)")
            continue

        # Evaluate strictly on points within reach of the Barnes filter
        rmse_base_sub = float(np.sqrt(mean_squared_error(target_truth[has_support], target_raw[has_support])))
        rmse_cand_sub = float(np.sqrt(mean_squared_error(target_truth[has_support], pred_cand[has_support])))
        gain_pct = ((rmse_base_sub - rmse_cand_sub) / rmse_base_sub) * 100.0

        pct_patch = (n_supported / len(target_raw)) * 100.0
        delta_str = f"{rmse_cand_sub - rmse_base_sub:+.4f}"
        print(
            f"      Candidate sigma = {sigma:4.1f} px | Supported points: {n_supported:5d} ({pct_patch:4.1f}%) | "
            f"RMSE: {rmse_base_sub:.4f} -> {rmse_cand_sub:.4f} ({delta_str}, Gain: {gain_pct:+.2f}%)"
        )

        if gain_pct > best_gain_pct:
            best_gain_pct = gain_pct
            best_sigma = sigma
            best_pred_cand = pred_cand.copy()

    # Validation criteria: must improve reachable RMSE by at least 1.0% without degrading global RMSE
    if best_sigma is not None and best_gain_pct >= 1.0:
        rmse_global_raw = float(np.sqrt(mean_squared_error(target_truth, target_raw)))
        rmse_global_refined = float(np.sqrt(mean_squared_error(target_truth, best_pred_cand)))

        if rmse_global_refined <= rmse_global_raw:
            print(
                f"   [Refinement Gate VALIDATED] Selected sigma = {best_sigma:.1f} px "
                f"(Local gain: +{best_gain_pct:.2f}%, Global RMSE: {rmse_global_raw:.4f} -> {rmse_global_refined:.4f})."
            )

            final_pred_cv = pred_cv.copy()
            if is_log_scale:
                final_pred_cv[valid_cv] = np.exp(best_pred_cand)
            else:
                final_pred_cv[valid_cv] = best_pred_cand

            return True, best_sigma, final_pred_cv

    print("   [Refinement Gate REJECTED] No candidate scale demonstrated robust improvement on reachable points.")
    return False, None, pred_cv


def apply_local_residual_refinement_2d(
    X_rec: np.ndarray,
    X_raw_obs: np.ndarray,
    mask_2d: np.ndarray,
    sigma_spatial_pixels: float = 3.0,
    decorrelation_cutoff: float = 0.05,
    is_log_scale: bool = False
) -> np.ndarray:
    """Refine DINEOF reconstruction with local residual objective analysis (Barnes filter).

    Propagates high-frequency spatial residuals from genuine satellite observations
    to adjacent unobserved pixels using a normalized Gaussian spatial kernel.
    The local correction smoothly decays to zero as distance from observations
    exceeds the spatial decorrelation length.

    Parameters
    ----------
    X_rec : np.ndarray
        Reconstructed matrix of shape ``(n_water, n_time)`` and dtype ``float32``.
    X_raw_obs : np.ndarray
        Raw un-imputed observation matrix of shape ``(n_water, n_time)`` containing NaNs.
    mask_2d : np.ndarray
        2D boolean domain mask where ``True`` indicates valid water pixels.
    sigma_spatial_pixels : float, default=3.0
        Standard deviation of the Gaussian spatial smoothing kernel (in pixels).
    decorrelation_cutoff : float, default=0.05
        Observation density threshold below which local residual corrections are ignored.
    is_log_scale : bool, default=False
        Whether to compute multiplicative residuals in natural-log space to preserve
        positivity of biogeochemical tracers.

    Returns
    -------
    X_refined : np.ndarray
        Refined spatiotemporal reconstruction array of shape ``(n_water, n_time)``.
    """
    n_water, n_time = X_rec.shape
    n_lat, n_lon = mask_2d.shape

    X_refined = X_rec.copy()

    buf_obs = np.zeros((n_lat, n_lon), dtype=np.float32)
    buf_rec = np.zeros((n_lat, n_lon), dtype=np.float32)
    buf_mask_obs = np.zeros((n_lat, n_lon), dtype=np.float32)

    water_indices = np.where(mask_2d)

    for t in tqdm(range(n_time), desc="Applying Local Spatial Refinement", unit="date"):
        raw_t = X_raw_obs[:, t]
        rec_t = X_rec[:, t]

        valid_obs = np.isfinite(raw_t)
        n_obs = int(np.count_nonzero(valid_obs))

        if n_obs < 10:
            continue

        buf_obs.fill(0.0)
        buf_rec.fill(0.0)
        buf_mask_obs.fill(0.0)

        buf_rec[water_indices] = rec_t

        obs_y = water_indices[0][valid_obs]
        obs_x = water_indices[1][valid_obs]
        obs_indices = (obs_y, obs_x)

        buf_obs[obs_indices] = raw_t[valid_obs]
        buf_mask_obs[obs_indices] = 1.0

        residual_2d = np.zeros((n_lat, n_lon), dtype=np.float32)
        if is_log_scale:
            buf_obs_log = np.log(np.maximum(buf_obs[obs_indices], 1e-7))
            buf_rec_log = np.log(np.maximum(buf_rec[obs_indices], 1e-7))
            residual_2d[obs_indices] = buf_obs_log - buf_rec_log
        else:
            residual_2d[obs_indices] = buf_obs[obs_indices] - buf_rec[obs_indices]

        res_conv = gaussian_filter(residual_2d, sigma=sigma_spatial_pixels, mode='constant', cval=0.0)
        weight_conv = gaussian_filter(buf_mask_obs, sigma=sigma_spatial_pixels, mode='constant', cval=0.0)

        valid_weight = weight_conv > decorrelation_cutoff
        local_correction = np.zeros((n_lat, n_lon), dtype=np.float32)
        local_correction[valid_weight] = res_conv[valid_weight] / weight_conv[valid_weight]

        corr_1d = local_correction[mask_2d]

        unobserved = ~valid_obs
        if is_log_scale:
            X_refined[unobserved, t] = X_rec[unobserved, t] * np.exp(np.clip(corr_1d[unobserved], -3.0, 3.0))
        else:
            X_refined[unobserved, t] += corr_1d[unobserved]

    return X_refined


def fit_quantile_transfer_training(
    X_obs_train: np.ndarray,
    X_rec_train: np.ndarray,
    out_dir: str,
    var_name: str,
    nb_quantiles: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an empirical quantile transfer using training observations only.

    The transfer maps reconstructed quantiles to observed quantiles. The caller
    is responsible for supplying values in the desired working space, normally
    natural-log space for positive geophysical variables. The calibration curve
    is also saved as a PNG diagnostic in ``out_dir``.

    Parameters
    ----------
    X_obs_train : np.ndarray
        Training observations in native log space.
    X_rec_train : np.ndarray
        Reconstructed values at training observation positions in native log space.
    out_dir : str
        Output directory for diagnostic plots.
    var_name : str
        Target geophysical variable name.
    nb_quantiles : int
        Number of quantile pairs used to represent the transfer curve.

    Returns
    -------
    q_rec : np.ndarray
        Quantiles of reconstructed values.
    q_obs : np.ndarray
        Quantiles of observed values.

    Raises
    ------
    ValueError
        If no finite paired observations are available for calibration.
    """
    print("   Calibrating Quantile Mapping strictly on training observations...")
    valid = np.isfinite(X_obs_train) & np.isfinite(X_rec_train)
    obs_v = X_obs_train[valid]
    rec_v = X_rec_train[valid]

    quantiles = np.linspace(0.001, 0.999, nb_quantiles)
    q_obs = np.quantile(obs_v, quantiles)
    q_rec = np.quantile(rec_v, quantiles)

    plt.figure(figsize=(6, 6))
    plt.plot(q_rec, q_rec, 'k--', label="Identity")
    plt.plot(q_rec, q_obs, 'r-', lw=2, label="Q-Map calibration (Training)")
    plt.xlabel("EOF Reconstruction (ln)")
    plt.ylabel("Training Observations (ln)")
    plt.title(f"Q-Map Transfer Curve: {var_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(out_dir, f"QMap_Transfer_Curve_{var_name}.png"), dpi=150)
    plt.close()

    return q_rec.astype(np.float32), q_obs.astype(np.float32)


def stream_reconstruction_and_oi_netcdf(
    out_path: str,
    X_rec: np.ndarray,
    dineof_meta: dict,
    mask_2d: np.ndarray,
    coords_meta: dict,
    var_name: str,
    is_log_scale: bool = True,
    apply_qmap: bool = False,
    q_rec: np.ndarray | None = None,
    q_obs: np.ndarray | None = None,
    X_raw_obs: np.ndarray | None = None,
    vmax: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write a reconstruction to NetCDF while computing OI diagnostics.

    Processing is performed one date at a time. The function optionally applies
    uncertainty-weighted quantile mapping, writes reconstructed values and
    errors to NetCDF cropped to the bounding box, and accumulates summaries.

    Parameters
    ----------
    out_path : str
        Destination NetCDF file path.
    X_rec : np.ndarray
        Reconstructed matrix of shape (n_water, n_time).
    dineof_meta : dict
        Metadata returned by :func:`dineof`; it must contain ``u``, ``s``,
        ``sigma_o`` and ``obs_mask``.
    mask_2d : np.ndarray
        Cropped 2D water mask array matching coords_meta spatial dimensions.
    coords_meta : dict
        Coordinate dictionary returned by :func:`load_netcdf_masked`.
    var_name : str
        Variable name.
    is_log_scale : bool
        Whether the reconstruction and OI variance are expressed in natural-log
        space before conversion to physical error units.
    apply_qmap : bool
        Whether to apply error-modulated Quantile Mapping.
    q_rec : np.ndarray or None
        Reconstructed quantiles returned by
        :func:`fit_quantile_transfer_training`.
    q_obs : np.ndarray or None
        Observed quantiles returned by
        :func:`fit_quantile_transfer_training`.
    X_raw_obs : np.ndarray or None
        Raw observation matrix for validation or preservation.
    vmax : float or None
        Optional upper bound applied to reconstructed values before writing.

    Returns
    -------
    mean_spatial_err : np.ndarray
        Mean physical analysis error map (n_water,).
    mean_spatial_rel : np.ndarray
        Mean relative error map (n_water,).
    temporal_err_mean : np.ndarray
        Domain-wide mean physical analysis error time series (n_time,).
    """
    n_time = len(coords_meta['time'])
    n_lat = len(coords_meta['lat'])
    n_lon = len(coords_meta['lon'])
    n_water = X_rec.shape[0]

    U_k = dineof_meta['u']
    s_k = dineof_meta['s']
    sigma_o = dineof_meta['sigma_o']
    obs_mask = dineof_meta['obs_mask']
    keep_obs = dineof_meta.get('keep_observations', True)

    buf_Uk_invA = np.empty_like(U_k)
    spatial_err_sum = np.zeros(n_water, dtype=np.float64)
    spatial_err_count = np.zeros(n_water, dtype=np.int32)
    spatial_rel_sum = np.zeros(n_water, dtype=np.float64)
    temporal_err_mean = np.zeros(n_time, dtype=np.float32)

    with nc.Dataset(out_path, 'w', format='NETCDF4') as ds_out:
        ds_out.createDimension('time', n_time)
        ds_out.createDimension('lat', n_lat)
        ds_out.createDimension('lon', n_lon)

        var_time = ds_out.createVariable('time', np.float64, ('time',))
        var_time.units = "seconds since 1970-01-01 00:00:00"
        var_lat = ds_out.createVariable('lat', coords_meta['lat'].dtype, ('lat',))
        var_lon = ds_out.createVariable('lon', coords_meta['lon'].dtype, ('lon',))

        var_data = ds_out.createVariable('data', np.float32, ('time', 'lat', 'lon'),
                                         zlib=True, complevel=1, fill_value=np.nan,
                                         chunksizes=(1, n_lat, n_lon))
        var_data.units = coords_meta['units']

        var_err = ds_out.createVariable('error', np.float32, ('time', 'lat', 'lon'),
                                        zlib=True, complevel=1, fill_value=np.nan,
                                        chunksizes=(1, n_lat, n_lon))
        var_err.units = coords_meta['units']

        var_rel = ds_out.createVariable('error_rel', np.float32, ('time', 'lat', 'lon'),
                                        zlib=True, complevel=1, fill_value=np.nan,
                                        chunksizes=(1, n_lat, n_lon))

        var_time[:] = coords_meta['time'].astype('datetime64[s]').astype(np.float64)
        var_lat[:] = coords_meta['lat']
        var_lon[:] = coords_meta['lon']

        # Precompute background covariance diagonal on the cropped points
        lambda_k = (s_k ** 2) / float(n_time)
        b_diag_global = np.sum((U_k * lambda_k) * U_k, axis=1).astype(np.float32)

        buf_data = np.empty((n_lat, n_lon), dtype=np.float32)
        buf_err = np.empty((n_lat, n_lon), dtype=np.float32)
        buf_rel = np.empty((n_lat, n_lon), dtype=np.float32)

        for t in tqdm(range(n_time), desc="Streaming NetCDF + OI Error + Q-Map", unit="date"):
            rec_t = X_rec[:, t]
            obs_mask_t = obs_mask[:, t]

            err_var_t, rel_err_t = compute_dineof_oi_error_step(
                U_k=U_k, s_k=s_k, N_time=n_time, sigma_o=sigma_o,
                obs_mask_t=obs_mask_t, buf_Uk_invA=buf_Uk_invA,
                b_diag=b_diag_global
            )

            if apply_qmap and (q_rec is not None) and (q_obs is not None):
                if is_log_scale:
                    ln_rec = np.log(np.maximum(rec_t, 1e-7))
                    delta_qmap = _interp_extrap(ln_rec, q_rec, q_obs) - ln_rec
                    rec_final_t = np.exp(ln_rec + rel_err_t * delta_qmap)
                else:
                    delta_qmap = _interp_extrap(rec_t, q_rec, q_obs) - rec_t
                    rec_final_t = rec_t + rel_err_t * delta_qmap
            else:
                rec_final_t = rec_t.copy()



            if keep_obs:
                rec_final_t[obs_mask_t] = rec_t[obs_mask_t]

            if vmax is not None:
                print(f"   [NetCDF Streaming] Applying upper bound {vmax} to reconstructed values.")
                rec_final_t = np.minimum(rec_final_t, vmax)

            buf_data.fill(np.nan)
            buf_data[mask_2d] = rec_final_t
            var_data[t, :, :] = buf_data

            sigma_o2 = float(sigma_o ** 2)

            if is_log_scale:
                var_log_t = sigma_o2 * rel_err_t
                valid_err = np.isfinite(var_log_t)
                err_phys_t = np.full(n_water, np.nan, dtype=np.float32)
                v_t = var_log_t[valid_err]
                err_phys_t[valid_err] = (
                    rec_final_t[valid_err]
                    * np.exp(0.5 * v_t)
                    * np.sqrt(np.expm1(v_t))
                )
            else:
                err_phys_t = np.sqrt(sigma_o2 * rel_err_t)

            buf_err.fill(np.nan)
            buf_err[mask_2d] = err_phys_t
            var_err[t, :, :] = buf_err

            buf_rel.fill(np.nan)
            buf_rel[mask_2d] = rel_err_t
            var_rel[t, :, :] = buf_rel

            valid_phys = np.isfinite(err_phys_t)
            spatial_err_sum[valid_phys] += err_phys_t[valid_phys]
            spatial_err_count[valid_phys] += 1
            spatial_rel_sum += rel_err_t
            temporal_err_mean[t] = float(np.nanmean(err_phys_t))

    disk_file = dineof_meta.get('disk_file')
    if hasattr(X_rec, '_mmap') and X_rec._mmap is not None:
        X_rec._mmap.close()
    if disk_file and os.path.exists(disk_file):
        try:
            os.remove(disk_file)
        except OSError:
            pass

    mean_spatial_err = np.full(n_water, np.nan, dtype=np.float32)
    has_err = spatial_err_count > 0
    mean_spatial_err[has_err] = (spatial_err_sum[has_err] / spatial_err_count[has_err]).astype(np.float32)
    mean_spatial_rel = (spatial_rel_sum / n_time).astype(np.float32)

    return mean_spatial_err, mean_spatial_rel, temporal_err_mean


def export_diagnostics_and_plots(
    out_dir: str,
    var_name: str,
    mask_2d: np.ndarray,
    coords_meta: dict,
    truth_cv: np.ndarray,
    pred_cv: np.ndarray,
    dineof_meta: dict,
    mean_spatial_err: np.ndarray,
    mean_spatial_rel: np.ndarray,
    temporal_err_mean: np.ndarray,
    n_bins: int = 20
):
    """Write validation CSV reports and a six-panel diagnostic dashboard."""
    is_log = dineof_meta.get('apply_log', True)

    if is_log:
        valid = np.isfinite(pred_cv) & np.isfinite(truth_cv) & (pred_cv > 0) & (truth_cv > 0)
        p_cln, t_cln = pred_cv[valid], truth_cv[valid]
        v_true = np.log10(t_cln)
        v_pred = np.log10(p_cln)
        scale_str = "Log10"
        sigma_o_ln = float(dineof_meta['sigma_o'])
        sigma_o_log10 = float(sigma_o_ln / np.log(10.0))
    else:
        valid = np.isfinite(pred_cv) & np.isfinite(truth_cv)
        p_cln, t_cln = pred_cv[valid], truth_cv[valid]
        v_true = t_cln
        v_pred = p_cln
        scale_str = "Linear"
        sigma_o_ln = np.nan
        sigma_o_log10 = float(dineof_meta['sigma_o'])

    rmse_trans = float(np.sqrt(mean_squared_error(v_true, v_pred)))
    bias_trans = float(np.mean(v_pred - v_true))
    r2_trans = float(r2_score(v_true, v_pred))

    rmse_phys = float(np.sqrt(mean_squared_error(t_cln, p_cln)))
    bias_phys = float(np.mean(p_cln - t_cln))

    pd.DataFrame([{
        'Variable': var_name,
        'Modes_Optimum': dineof_meta['n_opt'],
        'Variance_Initial': dineof_meta['varini'],
        'Variance_Reconstructed': dineof_meta['varend'],
        'Variance_Ratio': dineof_meta['var_ratio'],
        'Scale_Mode': scale_str,
        'Sigma_o_ln': sigma_o_ln,
        'Sigma_o_log10': sigma_o_log10,
        'RMSE_Transformed': rmse_trans,
        'Bias_Transformed': bias_trans,
        'R2_Transformed': r2_trans,
        'RMSE_Physical': rmse_phys,
        'Bias_Physical': bias_phys,
        'OI_Mean_Error': float(np.nanmean(mean_spatial_err))
    }]).to_csv(os.path.join(out_dir, f"Error_Matrix_{var_name}.csv"), index=False)

    bins = np.linspace(v_true.min(), v_true.max(), n_bins + 1)
    b_idx = np.digitize(v_true, bins)
    stats_spec = []
    for i in range(1, n_bins + 1):
        m = (b_idx == i)
        if np.sum(m) < 5:
            continue
        stats_spec.append({
            'bin_center': float((bins[i-1] + bins[i]) / 2),
            'rmse': float(np.sqrt(mean_squared_error(v_true[m], v_pred[m]))),
            'bias': float(np.mean(v_pred[m] - v_true[m])),
            'count': int(np.sum(m))
        })
    df_spec = pd.DataFrame(stats_spec)
    df_spec.to_csv(os.path.join(out_dir, f"Error_Profile_{var_name}.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    hb = axes[0, 0].hexbin(v_true, v_pred, gridsize=45, cmap='inferno', mincnt=1, bins='log')
    plt.colorbar(hb, ax=axes[0, 0], label='Pixel Count (Log)')
    axes[0, 0].plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'w--')
    axes[0, 0].set_title(f"Out-of-Sample Evaluation: {var_name}\nRMSE={rmse_trans:.3f} | R²={r2_trans:.3f}")
    axes[0, 0].set_xlabel(f"Ground Truth ({scale_str})")
    axes[0, 0].set_ylabel(f"Model Prediction ({scale_str})")

    if not df_spec.empty:
        axes[0, 1].plot(df_spec['bin_center'], df_spec['rmse'], 'r-o', label='RMSE')
        axes[0, 1].plot(df_spec['bin_center'], df_spec['bias'], 'b-s', label='Bias')
        axes[0, 1].axhline(0, color='k', ls='--')
        axes[0, 1].set_title(f"Spectral Error Profile ({scale_str})")
        axes[0, 1].set_xlabel(f"Intensity ({scale_str})")
        axes[0, 1].legend()

    axes[0, 2].plot(range(1, len(dineof_meta['valc']) + 1), dineof_meta['valc'], 'k-o')
    axes[0, 2].axvline(dineof_meta['n_opt'], color='g', ls='--', label=f"Optimum k={dineof_meta['n_opt']}")
    axes[0, 2].set_title("Mode Selection Convergence")
    axes[0, 2].set_xlabel("EOF Mode")
    axes[0, 2].legend()

    map_err = np.full(mask_2d.shape, np.nan, dtype=np.float32)
    map_err[mask_2d] = mean_spatial_err
    im1 = axes[1, 0].imshow(map_err, cmap='turbo')
    plt.colorbar(im1, ax=axes[1, 0])
    axes[1, 0].set_title(f"OI Mean Analysis Error ({coords_meta['units']})")
    axes[1, 0].axis('off')

    map_rel = np.full(mask_2d.shape, np.nan, dtype=np.float32)
    map_rel[mask_2d] = mean_spatial_rel
    im2 = axes[1, 1].imshow(map_rel, cmap='magma', vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1, 1])
    axes[1, 1].set_title("OI Mean Relative Error (P / B)")
    axes[1, 1].axis('off')

    axes[1, 2].plot(pd.to_datetime(coords_meta['time']), temporal_err_mean, 'm-')
    axes[1, 2].set_title("Temporal Error Evolution")
    axes[1, 2].set_xlabel("Date")
    plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"Dashboard_{var_name}.png"), dpi=150)
    plt.close()


def run_reconstruction_pipeline(
    input_nc: str,
    mask_file: str,
    var_name: str,
    out_dir: str,
    threshold: float | None = None,
    vmax: float | None = None,
    min_valid_ratio: float = 0.01,
    apply_log: bool = True,
    use_climatology: bool = True,
    apply_qmap: bool = True,
    qmap_nb_quantiles: int = 100,
    apply_local_refinement: bool = False,
    local_refinement_sigma: float | None = None,
    use_disk: bool = False,
    solver: str = 'auto',
    dineof_kwargs: dict | None = None
):
    """Run the complete NetCDF-to-diagnostics reconstruction workflow.

    The workflow loads and masks the input field, removes dates with insufficient
    spatial coverage, runs the numerical DINEOF core, optionally refines the
    reconstruction with local spatial residuals (Barnes objective analysis filter)
    arbitrated via held-out cross-validation, calibrates a leakage-free quantile
    transfer, streams the reconstruction to NetCDF, and writes validation reports.

    Parameters
    ----------
    input_nc : str
        Source NetCDF file containing the time-varying geophysical variable.
    mask_file : str
        NetCDF file containing the binary variable named ``mask``.
    var_name : str
        Name of the source variable and the variable used in output labels.
    out_dir : str
        Directory receiving the reconstructed NetCDF file, CSV reports and PNGs.
    threshold : float or None
        Lower physical bound passed to :func:`load_netcdf_masked`.
    vmax : float or None
        Optional upper bound applied before writing reconstructed values.
    min_valid_ratio : float
        Minimum fraction of valid spatial pixels required for a date to remain.
    apply_log : bool
        Reconstruct in natural-log space and convert back to physical units.
    use_climatology : bool
        Remove a spatial climatology before reconstruction.
    apply_qmap : bool
        Enable uncertainty-weighted empirical quantile mapping when calibration
        passes its out-of-sample quality check.
    qmap_nb_quantiles : int
        Number of points used to represent the quantile transfer curve.
    apply_local_refinement : bool, default=False
        Whether to refine unobserved pixels with local residual objective analysis.
    local_refinement_sigma : float or None, default=None
        Standard deviation in pixels of the Gaussian kernel for residual refinement.
        If None, tests a candidate grid [1.5, 2.5, 3.5, 5.0, 7.0] and auto-selects.
    use_disk : bool
        Store DINEOF work arrays in temporary memory-mapped files when possible.
    solver : str
        Spectral solver passed to :func:`dineof`.
    dineof_kwargs : dict or None
        Additional keyword arguments forwarded to :func:`dineof`. Values in this
        dictionary take precedence over ``use_disk`` and ``solver``.

    Returns
    -------
    None
        Results are written to ``out_dir``. The main output is
        ``DINEOF_Reconstructed_<var_name>.nc``.
    """
    os.makedirs(out_dir, exist_ok=True)
    opts = dineof_kwargs.copy() if dineof_kwargs else {}
    if 'use_disk' not in opts:
        opts['use_disk'] = use_disk
    if 'solver' not in opts:
        opts['solver'] = solver

    do_refine = opts.pop('apply_local_refinement', apply_local_refinement)
    refine_sigma = opts.pop('local_refinement_sigma', local_refinement_sigma)

    X_input, mask_2d, coords_meta = load_netcdf_masked(
        file_path=input_nc,
        mask_file=mask_file,
        var_name=var_name,
        threshold=threshold
    )
    n_water, n_time = X_input.shape

    print(f"    [Domain] Sub-region bounding box: {len(coords_meta['lat'])} lat x {len(coords_meta['lon'])} lon ({n_water} active water pixels).")

    # Preserve all calendar dates: blank low-coverage dates to NaN for reconstruction
    if min_valid_ratio > 0.0:
        valid_per_date = np.count_nonzero(~np.isnan(X_input), axis=0)
        low_coverage = (valid_per_date / float(n_water)) < min_valid_ratio
        n_blanked = int(np.sum(low_coverage))
        if n_blanked > 0:
            print(f"    [Date Filter] Setting {n_blanked} dates below {min_valid_ratio:.1%} coverage to NaN (preserved for reconstruction).")
            X_input[:, low_coverage] = np.nan

    X_raw_observations = X_input.copy()
    if vmax is not None:
        print(f"    [Pre-Process] Applying upper bound of {vmax} to raw observations...")
        X_raw_observations[X_raw_observations > vmax] = np.nan
        X_input[X_input > vmax] = np.nan

    user_keep_obs = opts.get('keep_observations', True)
    opts_core = opts.copy()
    if do_refine:
        opts_core['keep_observations'] = False

    print("\n-> Launching DINEOF core engine...")

    X_final, meta = dineof(
        data=X_input,
        time_coords=coords_meta['time'],
        mask=mask_2d,
        apply_log=apply_log,
        use_climatology=use_climatology,
        **opts_core
    )

    truth_cv = meta['cv_truth']
    pred_cv = meta['cv_pred'].copy()

    # --- Cross-Validation Arbitration & Local Spatial Refinement ---
    if do_refine:
        sigmas_to_test = [refine_sigma] if (refine_sigma is not None) else [1.5, 2.5, 3.5, 5.0, 7.0]
        validated, opt_sigma, pred_cv_refined = arbitrate_local_residual_refinement(
            X_rec=X_final,
            X_raw_obs=X_raw_observations,
            mask_2d=mask_2d,
            dineof_meta=meta,
            candidate_sigmas=sigmas_to_test,
            is_log_scale=apply_log
        )

        if validated and opt_sigma is not None:
            pred_cv = pred_cv_refined
            print(f"   [Post-Process] Applying local residual refinement with validated sigma={opt_sigma:.1f} px...")
            X_final = apply_local_residual_refinement_2d(
                X_rec=X_final,
                X_raw_obs=X_raw_observations,
                mask_2d=mask_2d,
                sigma_spatial_pixels=opt_sigma,
                is_log_scale=apply_log
            )
        else:
            print("   [Post-Process] Local residual refinement deactivated by CV arbitration gate.")

    # Strictly enforce user observation preference on the final reconstructed matrix
    if user_keep_obs:
        print("   [Post-Process] Restoring genuine raw satellite observations to valid pixels...")
        valid_obs_mask = np.isfinite(X_raw_observations)
        X_final[valid_obs_mask] = X_raw_observations[valid_obs_mask]
    
    meta['keep_observations'] = user_keep_obs

    cv_r = meta['cv_rows']
    cv_c = meta['cv_cols']
    U_k = meta['u']
    s_k = meta['s']
    sigma_o = meta['sigma_o']
    obs_mask = meta['obs_mask']

    inv_lambda = 1.0 / np.maximum((s_k ** 2) / float(n_time), 1e-12)
    inv_sigma2 = 1.0 / max(sigma_o ** 2, 1e-12)
    lambda_k = (s_k ** 2) / float(n_time)

    unique_dates = np.unique(cv_c)
    r_cv = np.empty(len(cv_r), dtype=np.float32)

    for t in unique_dates:
        mask_t = (cv_c == t)
        pts_t = cv_r[mask_t]

        obs_mask_t = obs_mask[:, t].copy()
        obs_mask_t[pts_t] = False

        U_pts = U_k[pts_t, :]
        b_diag_pts = np.sum((U_pts * lambda_k) * U_pts, axis=1)
        valid_b = b_diag_pts > 1e-9

        if np.any(obs_mask_t):
            U_obs = U_k[obs_mask_t, :]
            G = U_obs.T @ U_obs
            A = np.diag(inv_lambda) + inv_sigma2 * G
            inv_A = np.linalg.inv(A)
            err_var_pts = np.sum((U_pts @ inv_A) * U_pts, axis=1)
        else:
            err_var_pts = b_diag_pts.copy()

        rel_pts = np.ones(len(pts_t), dtype=np.float32)
        rel_pts[valid_b] = np.clip(err_var_pts[valid_b] / b_diag_pts[valid_b], 0.0, 1.0)
        r_cv[mask_t] = rel_pts

    q_rec, q_obs = None, None
    if apply_qmap:
        obs_mask_train = np.isfinite(X_input)
        cv_r = meta['cv_rows']
        cv_c = meta['cv_cols']
        obs_mask_train[cv_r, cv_c] = False

        train_r, train_c = np.where(obs_mask_train)
        n_train_obs = len(train_r)

        if n_train_obs > 100:
            max_qmap_train = 500_000
            if n_train_obs > max_qmap_train:
                rng_qmap = np.random.default_rng(42)
                idx_sub = rng_qmap.choice(n_train_obs, size=max_qmap_train, replace=False)
                sub_r = train_r[idx_sub]
                sub_c = train_c[idx_sub]
            else:
                sub_r = train_r
                sub_c = train_c

            v = meta['v']
            s_k = meta['s']
            clim = meta.get('clim', 0.0)
            if isinstance(clim, np.ndarray):
                c_train = clim[sub_r, 0] if clim.ndim > 1 else clim[sub_r]
            else:
                c_train = float(clim)

            rec_smooth_sample = np.sum(U_k[sub_r, :] * (s_k * v[sub_c, :]), axis=1) + c_train

            if apply_log:
                obs_sample = np.log(np.maximum(X_input[sub_r, sub_c], 1e-7))
            else:
                obs_sample = X_input[sub_r, sub_c]

            q_rec_cand, q_obs_cand = fit_quantile_transfer_training(
                X_obs_train=obs_sample,
                X_rec_train=rec_smooth_sample,
                out_dir=out_dir,
                var_name=var_name,
                nb_quantiles=qmap_nb_quantiles
            )

            def compute_kge(yt, yp):
                m = np.isfinite(yt) & np.isfinite(yp)
                if np.sum(m) < 10:
                    return -np.inf
                yt, yp = yt[m], yp[m]
                r = float(np.corrcoef(yt, yp)[0, 1])
                alpha = float(np.std(yp) / max(np.std(yt), 1e-6))
                beta = float(np.mean(yp) / max(np.mean(yt), 1e-6))
                return float(1.0 - np.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2))

            kge_raw = compute_kge(truth_cv, pred_cv)

            if apply_log:
                ln_pred = np.log(np.maximum(pred_cv, 1e-7))
                delta_q = _interp_extrap(ln_pred, q_rec_cand, q_obs_cand) - ln_pred
                pred_cv_qmap = np.exp(ln_pred + r_cv * delta_q)
            else:
                delta_q = _interp_extrap(pred_cv, q_rec_cand, q_obs_cand) - pred_cv
                pred_cv_qmap = pred_cv + r_cv * delta_q

            kge_qmap = compute_kge(truth_cv, pred_cv_qmap)

            if kge_qmap < kge_raw:
                print(
                    f"\n[WARNING Q-MAP] Quantile mapping led to a degradation in performance (KGE: {kge_raw:.3f} -> {kge_qmap:.3f}).\n"
                    "Deactivation of quantile mapping to preserve reconstruction quality."
                )
                apply_qmap = False
                q_rec, q_obs = None, None
            else:
                print(f"[Q-MAP VALIDATED] Quantile mapping improved performance (KGE: {kge_raw:.3f} -> {kge_qmap:.3f}).")
                q_rec, q_obs = q_rec_cand, q_obs_cand
                pred_cv = pred_cv_qmap

    out_nc = os.path.join(out_dir, f"DINEOF_Reconstructed_{var_name}.nc")
    X_stream_obs = X_raw_observations if opts.get('keep_observations', True) else None

    mean_spatial_err, mean_spatial_rel, temporal_err_mean = stream_reconstruction_and_oi_netcdf(
        out_path=out_nc,
        X_rec=X_final,
        dineof_meta=meta,
        mask_2d=mask_2d,
        coords_meta=coords_meta,
        var_name=var_name,
        is_log_scale=apply_log,
        apply_qmap=apply_qmap,
        q_rec=q_rec,
        q_obs=q_obs,
        X_raw_obs=X_stream_obs,
        vmax=vmax
    )

    export_diagnostics_and_plots(
        out_dir=out_dir,
        var_name=var_name,
        mask_2d=mask_2d,
        coords_meta=coords_meta,
        truth_cv=truth_cv,
        pred_cv=pred_cv,
        dineof_meta=meta,
        mean_spatial_err=mean_spatial_err,
        mean_spatial_rel=mean_spatial_rel,
        temporal_err_mean=temporal_err_mean
    )