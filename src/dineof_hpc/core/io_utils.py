# -*- coding: utf-8 -*-
"""I/O, uncertainty estimation, and post-processing for DINEOF-HPC.

The module connects the array-based numerical core to geophysical NetCDF files.
It provides batched loading, streaming reconstruction with optimal-interpolation
(OI) uncertainty, leakage-free empirical quantile mapping, and diagnostic report
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


def load_netcdf_masked(
    file_path: str,
    mask_file: str,
    var_name: str,
    threshold: float | None = None,
    batch_size: int = 50,
    n_workers: int = 6
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load a masked NetCDF variable by batches using parallel decompression.

    Extracts active water coordinates using bounding box cropping, raw array
    slicing with auto-masking disabled, and multi-process batch execution via
    memory-mapped buffers.

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
        2D boolean domain mask where ``True`` indicates valid water pixels.
    coords_meta : dict
        Extracted coordinates metadata (``'time'``, ``'lat'``, ``'lon'``, ``'units'``).
    """
    # 1. Ingest mask and calculate spatial bounding box
    with nc.Dataset(mask_file, 'r') as m_ds:
        mask_raw = m_ds.variables['mask'][:]
        if np.ma.isMaskedArray(mask_raw):
            mask_raw = mask_raw.filled(0)
        mask_2d = np.asarray(mask_raw == 1, dtype=bool)

    water_y, water_x = np.where(mask_2d)
    n_water = len(water_y)

    y_min, y_max = int(np.min(water_y)), int(np.max(water_y)) + 1
    x_min, x_max = int(np.min(water_x)), int(np.max(water_x)) + 1
    width_crop = x_max - x_min

    # Precalculate 1D flattened coordinates inside the cropped spatial window
    water_y_crop = water_y - y_min
    water_x_crop = water_x - x_min
    water_flat_crop = water_y_crop * width_crop + water_x_crop

    # 2. Extract coordinate metadata lazily
    with xr.open_dataset(file_path) as ds_meta:
        coords_meta = {
            'time': ds_meta.time.values,
            'lat': ds_meta.lat.values,
            'lon': ds_meta.lon.values,
            'units': ds_meta[var_name].attrs.get('units', 'a.u.')
        }

    # 3. Read dataset dimensions and chunk configuration
    with nc.Dataset(file_path, 'r') as ds_in:
        var = ds_in.variables[var_name]
        n_time = var.shape[0]
        fill_val = getattr(var, '_FillValue', None)

        chunks = var.chunking()
        if chunks and chunks != 'contiguous':
            time_chunk = chunks[0]
            if time_chunk > 0:
                batch_size = max(batch_size, time_chunk * max(1, batch_size // time_chunk))

    # Allocate shared temporary memory map on scratch space or /dev/shm
    scratch_dir = os.environ.get('JOBLIB_TEMP_FOLDER', '/dev/shm' if os.path.exists('/dev/shm') else '/tmp')
    mmap_path = os.path.join(scratch_dir, f"load_X_{os.getpid()}_{int(time.time() * 1000)}.dat")
    X_mmap = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=(n_water, n_time))

    batch_ranges = [
        (t, min(t + batch_size, n_time))
        for t in range(0, n_time, batch_size)
    ]

    # 4. Parallel extraction across batches
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

    # 5. Load into RAM array and clean temporary buffer
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
    errors to NetCDF, and accumulates spatial and temporal error summaries.

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
        2D water mask array.
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

    Notes
    -----
    The output file contains ``data``, ``error`` and ``error_rel`` variables,
    each with dimensions ``(time, lat, lon)``. Any memmap referenced by
    ``dineof_meta['disk_file']`` is closed and removed after streaming.
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

        # This diagonal is reused for every date in the OI calculation.
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

            # Move the reconstruction toward observations according to OI
            # confidence: observed locations receive little or no correction.
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

            if vmax is not None:
                rec_final_t = np.minimum(rec_final_t, vmax)

            if keep_obs:
                rec_final_t[obs_mask_t] = rec_t[obs_mask_t]

            buf_data.fill(np.nan)
            buf_data[mask_2d] = rec_final_t
            var_data[t, :, :] = buf_data

            # Convert relative OI uncertainty into an absolute error estimate.
            sigma_o2 = float(sigma_o ** 2)

            if is_log_scale:
                # In log space, propagate variance through the exponential map.
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
    """Write validation CSV reports and a six-panel diagnostic dashboard.

    Two CSV files are produced in ``out_dir``: a global error summary and an
    intensity-binned error profile. A PNG dashboard combines cross-validation,
    mode-selection, spatial OI, and temporal error plots.
    """
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
    use_disk: bool = False,
    solver: str = 'auto',
    dineof_kwargs: dict | None = None
):
    """Run the complete NetCDF-to-diagnostics reconstruction workflow.

    The workflow loads and masks the input field, removes dates with insufficient
    spatial coverage, runs the numerical DINEOF core, optionally calibrates a
    leakage-free quantile transfer, streams the reconstruction to NetCDF, and
    writes validation reports and figures.

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

    Notes
    -----
    Quantile mapping is calibrated on training observations only; the withheld
    cross-validation points are excluded from calibration and used as a quality
    gate before the correction is enabled.
    """
    os.makedirs(out_dir, exist_ok=True)
    opts = dineof_kwargs.copy() if dineof_kwargs else {}
    if 'use_disk' not in opts:
            opts['use_disk'] = use_disk
    if 'solver' not in opts:
        opts['solver'] = solver
    
    X_input, mask_2d, coords_meta = load_netcdf_masked(
        file_path=input_nc,
        mask_file=mask_file,
        var_name=var_name,
        threshold=threshold
    )
    n_water, n_time = X_input.shape

    # Remove dates whose valid spatial coverage is below the requested minimum.
    if min_valid_ratio > 0.0:
        valid_per_date = np.count_nonzero(~np.isnan(X_input), axis=0)
        keep_dates = (valid_per_date / float(n_water)) >= min_valid_ratio
        n_dropped = n_time - int(np.sum(keep_dates))
        if n_dropped > 0:
            print(f"    [Date Filter] Dropping {n_dropped} dates below {min_valid_ratio:.1%} coverage.")
            X_input = X_input[:, keep_dates]
            coords_meta['time'] = coords_meta['time'][keep_dates]
            n_time = X_input.shape[1]

    X_raw_observations = X_input.copy() if opts.get('keep_observations', True) else None

    print("\n-> Launching DINEOF core engine...")
    
    X_final, meta = dineof(
        data=X_input,
        time_coords=coords_meta['time'],
        mask=mask_2d,
        apply_log=apply_log,
        use_climatology=use_climatology,
        **opts
    )

    truth_cv = meta['cv_truth']
    pred_cv = meta['cv_pred'].copy()

    # Recompute OI relative uncertainty at the withheld evaluation coordinates.
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

    # Calibrate quantile mapping strictly on observations not withheld for CV.
    q_rec, q_obs = None, None
    if apply_qmap:
        # Start from the finite training-observation mask.
        obs_mask_train = np.isfinite(X_input)

        # Prevent validation data from leaking into the calibration sample.
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

            # Accept the transfer only when it improves the held-out KGE score.
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
                    f"Deactivation of quantile mapping to preserve reconstruction quality."
                )
                apply_qmap = False
                q_rec, q_obs = None, None
            else:
                print(f"[Q-MAP VALIDATED] Quantile mapping improved performance (KGE: {kge_raw:.3f} -> {kge_qmap:.3f}).")
                q_rec, q_obs = q_rec_cand, q_obs_cand
                pred_cv = pred_cv_qmap

    out_nc = os.path.join(out_dir, f"DINEOF_Reconstructed_{var_name}.nc")
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
        X_raw_obs=X_raw_observations,
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