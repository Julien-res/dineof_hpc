# -*- coding: utf-8 -*-
"""Numerical core of the Data Interpolating Empirical Orthogonal Functions method.

The module operates on NumPy arrays and deliberately contains no file-format
logic. Missing values must be represented by ``NaN`` on input. The main public
entry point is :func:`dineof`; the other public helper computes the analytical
optimal-interpolation error associated with the retained EOF modes.

The implementation combines iterative DINEOF reconstruction with spatial
cross-validation, optional temporal diffusion, chunked updates, and several
eigensolver strategies. Array conventions are ``(space, time)`` internally and
``(time, y, x)`` for gridded input.

Mathematical & Methodological References:
    - Beckers, J. M., & Rixen, M. (2003). EOF calculations and data filling from
      incomplete oceanographic datasets. J. Atmos. Oceanic Technol., 20(12), 1839-1856.
    - Beckers, J. M., et al. (2006). Error calculation on DINEOF data reconstruction
      using an optimal interpolation approach. Ocean Science, 2(2), 79-99.
    - Alvera-Azcarate, A., et al. (2009). The use of the temporal filter in DINEOF:
      analysis of a Gulf of Mexico SST data set. Ocean Science, 5(4), 475-485.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.linalg import eigh
from joblib import cpu_count
from scipy.sparse.linalg import eigsh

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
#                   TEMPORAL DIFFUSION AND SPECTRAL OPERATORS
# =============================================================================

def _diff_implicit_1d_python(
    x: NDArray[np.float32],
    B: NDArray[np.float32],
    alpha: float,
    numit: int
) -> NDArray[np.float32]:
    """Apply an unconditionally stable 1D implicit Laplacian diffusion filter.

    Solves the backward Euler system (I - alpha * L) B^{n+1} = B^n on non-uniform
    temporal grids using finite volumes with half-cell boundary control volumes
    and zero-flux (Neumann) boundary conditions. The resulting strictly diagonally
    dominant tridiagonal system is solved directly via the Thomas algorithm in O(N).

    Parameters
    ----------
    x : NDArray[np.float32]
        1D temporal coordinates in decimal days (length N).
    B : NDArray[np.float32]
        1D array or matrix row representing the temporal covariance slice.
    alpha : float
        Laplacian temporal diffusion coefficient (dimension: time^2).
    numit : int
        Number of filtering iterations.

    Notes
    -----
    The boundary control volumes are half-sized. This is the finite-volume
    equivalent of a zero normal derivative at both ends of the time axis.

    Returns
    -------
    NDArray[np.float32]
        Smoothed array of length N.
    """
    N = len(B)
    if N < 2 or alpha <= 0.0 or numit <= 0:
        return B.copy()

    # Time step increments between successive observations
    dx = np.diff(x).astype(np.float32)
    dx = np.maximum(dx, 1e-7)

    # Control volume cell widths: boundary cells are defined with half-width
    dxe = np.empty(N, dtype=np.float32)
    dxe[0] = 0.5 * dx[0]
    dxe[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    dxe[-1] = 0.5 * dx[-1]

    # Tridiagonal coefficients: a_i * y_{i-1} + b_i * y_i + c_i * y_{i+1} = d_i
    a = np.zeros(N, dtype=np.float32)
    b = np.ones(N, dtype=np.float32)
    c = np.zeros(N, dtype=np.float32)

    # Left boundary node i = 0 (zero-flux)
    c[0] = -alpha / (dx[0] * dxe[0])
    b[0] = 1.0 - c[0]

    # Interior nodes 1 <= i <= N - 2
    for i in range(1, N - 1):
        a[i] = -alpha / (dx[i - 1] * dxe[i])
        c[i] = -alpha / (dx[i] * dxe[i])
        b[i] = 1.0 - a[i] - c[i]

    # Right boundary node i = N - 1 (zero-flux)
    a[-1] = -alpha / (dx[-1] * dxe[-1])
    b[-1] = 1.0 - a[-1]

    res = B.copy()
    cp = np.empty(N, dtype=np.float32)
    dp = np.empty(N, dtype=np.float32)

    for _ in range(numit):
        # Forward elimination sweep
        cp[0] = c[0] / b[0]
        dp[0] = res[0] / b[0]
        for i in range(1, N):
            denom = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / denom
            dp[i] = (res[i] - a[i] * dp[i - 1]) / denom

        # Backward substitution sweep
        res[-1] = dp[-1]
        for i in range(N - 2, -1, -1):
            res[i] = dp[i] - cp[i] * res[i + 1]

    return res

if HAS_NUMBA:
    _dindiff = jit(nopython=True)(_diff_implicit_1d_python)
else:
    _dindiff = _diff_implicit_1d_python


def _get_diff(
    B: NDArray[np.float32],
    arr_time: NDArray[np.float32],
    alpha: float,
    numit: int
) -> NDArray[np.float32]:
    """Apply the 1D diffusion filter to each covariance row.

    Parameters
    ----------
    B : NDArray[np.float32]
        Square temporal covariance matrix with shape ``(N, N)``.
    arr_time : NDArray[np.float32]
        Temporal coordinates in decimal days.
    alpha : float
        Diffusion coefficient passed to :func:`_diff_implicit_1d_python`.
    numit : int
        Number of implicit filter passes per row.

    Returns
    -------
    NDArray[np.float32]
        Filtered covariance matrix, transposed to preserve the operator's
        historical orientation.
    """
    Bp = np.zeros_like(B)
    for i in range(B.shape[1]):
        Bp[i] = _dindiff(arr_time, B[i], alpha, numit)
    return np.transpose(Bp)


def _resolve_solver(solver: str, N: int, nev: int) -> str:
    """Resolve spectral solver selection based on temporal dimension and mode rank.

    Parameters
    ----------
    solver : str
        Specified solver mode ('auto', 'dense', 'lanczos', 'randomized').
    N : int
        Temporal dimension (number of dates).
    nev : int
        Truncation rank (number of EOF modes).

    Returns
    -------
    str
        Resolved solver name ('dense', 'lanczos', or 'randomized').
    """
    if solver != 'auto':
        return solver

    # Security check: ARPACK needs N > nev with a minimal Krylov subspace
    if N <= nev + 5 or N < 50:
        return 'dense'

    # Benchmark: N*(k) ≈ max(70, 10 * k)
    n_crossover = max(70, 10 * nev)

    return 'dense' if N <= n_crossover else 'lanczos'


def _parallel_AtA(
    A: NDArray[np.float32],
    idx_sub: NDArray[np.int64] | None = None,
    scale: float = 1.0,
    subsample_ratio: float = 1.0,
    subsample_min: int = 5_000_000,
    seed: int | None = 42
) -> NDArray[np.float32]:
    """Compute the temporal covariance matrix ``B = A.T @ A``.

    A precomputed ``idx_sub`` takes precedence over random subsampling. When
    rows are sampled, ``scale`` rescales the estimate to the full spatial
    population so that its expected covariance remains unchanged.
    """
    M, N = A.shape
    if idx_sub is not None:
        A_work = A[idx_sub]
    elif (subsample_ratio < 1.0) and (M > subsample_min):
        target = min(max(int(M * subsample_ratio), subsample_min), M)
        rng = np.random.default_rng(seed)
        idx_calc = np.sort(rng.choice(M, size=target, replace=False))
        A_work = A[idx_calc]
        scale = float(M / target)
    else:
        A_work = A
        scale = 1.0

    return (A_work.T @ A_work) * scale


def _randomized_eigh(
    B: NDArray[np.float32],
    nev: int,
    n_oversamples: int = 10,
    n_iter: int = 2,
    seed: int | None = 42
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute truncated eigendecomposition of a symmetric matrix via randomized SVD.

    Parameters
    ----------
    B : NDArray[np.float32]
        Symmetric covariance matrix of shape (N, N).
    nev : int
        Target number of leading eigenvectors to extract.
    n_oversamples : int, default=10
        Additional random test vectors to improve spectral subspace capture.
    n_iter : int, default=2
        Number of power iterations to accelerate singular value decay.
    seed : int or None, default=42
        Seed for Gaussian matrix generator ensuring iteration stability.

    Returns
    -------
    eigenvalues : NDArray[np.float32]
        Truncated eigenvalues of length nev (sorted descending).
    eigenvectors : NDArray[np.float32]
        Truncated eigenvectors of shape (N, nev).
    """
    N = B.shape[0]
    l = min(nev + n_oversamples, N)

    rng = np.random.default_rng(seed)
    Omega = rng.standard_normal((N, l), dtype=np.float32)

    # Subspace sampling with power iterations
    Y = B @ Omega
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y, mode='reduced')
        Y = B @ Q

    Q, _ = np.linalg.qr(Y, mode='reduced')

    # Projection into reduced subspace (dimension l x l)
    B_small = Q.T @ (B @ Q)
    eigenvalues_small, V_small = eigh(B_small)

    # Sort descending eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues_small)[::-1][:nev]
    eigenvalues = eigenvalues_small[idx]
    eigenvectors = Q @ V_small[:, idx]

    return eigenvalues.astype(np.float32), eigenvectors.astype(np.float32)


def _ssvd_spectral(
    A: NDArray[np.float32],
    nev: int,
    arr_time: NDArray[np.float32],
    alpha: float,
    numit: int,
    u: NDArray[np.float32] | None,
    s: NDArray[np.float32],
    v: NDArray[np.float32],
    compute_u: bool = True,
    idx_sub: NDArray[np.int64] | None = None,
    scale: float = 1.0,
    solver: str = 'auto',
    seed: int | None = 42
) -> tuple[NDArray[np.float32] | None, NDArray[np.float32], NDArray[np.float32]]:
    """Compute leading EOF modes from the temporal covariance matrix.

    The decomposition is performed on ``A.T @ A`` rather than on the usually
    much larger spatial covariance matrix. Temporal diffusion is applied before
    the eigensolve when ``alpha > 0``.

    Parameters
    ----------
    A : NDArray[np.float32]
        Data matrix of shape (M, N).
    nev : int
        Number of EOF modes to retain.
    arr_time : NDArray[np.float32]
        1D temporal coordinates.
    alpha : float
        Temporal diffusion filter parameter.
    numit : int
        Number of diffusion filter passes.
    u : NDArray[np.float32] or None
        Pre-allocated spatial modes array of shape (M, nev).
    s : NDArray[np.float32]
        Pre-allocated singular values array of shape (nev,).
    v : NDArray[np.float32]
        Pre-allocated temporal modes array of shape (N, nev).
    compute_u : bool
        Whether to compute and normalize spatial modes U.
    idx_sub : NDArray[np.int64] or None
        Optional spatial row indices used for a precomputed covariance sample.
    scale : float
        Covariance scaling factor associated with ``idx_sub``.
    seed : int or None
        Random seed for subsampling reproducibility.
    solver : str, default='dense'
        Eigensolver algorithm:
        - 'dense' : LAPACK syevr with subset_by_index (exact MRRR solver).
        - 'lanczos' : ARPACK restarted Lanczos with warm-start (eigsh).
        - 'randomized' : Randomized SVD (Halko et al., 2011) on covariance.

    Notes
    -----
    ``v`` contains normalized temporal EOFs. When ``compute_u`` is true,
    ``u`` receives normalized spatial EOFs and ``s`` their corresponding
    singular values. Otherwise only ``s`` and ``v`` are updated.

    Returns
    -------
    tuple[NDArray[np.float32] | None, NDArray[np.float32], NDArray[np.float32]]
        Updated tuple containing (u, s, v).
    """
    N = A.shape[1]
    B = _parallel_AtA(A, idx_sub=idx_sub, scale=scale)

    if alpha > 0.0:
        B = _get_diff(B, arr_time, alpha, numit)
        B = _get_diff(B, arr_time, alpha, numit)
        B = 0.5 * (B + B.T)

    if solver == 'randomized' and nev < (N - 15):
        eigenvalues_active, eigenvectors = _randomized_eigh(
            B, nev=nev, n_oversamples=10, n_iter=2, seed=seed
        )
        v[:, :nev] = eigenvectors
    elif solver == 'lanczos' and nev < (N - 1):
        ncv = min(max(2 * nev + 1, 20), N - 1)
        v0 = v[:, 0].copy() if np.any(v[:, 0]) else None
        try:
            eigenvalues, eigenvectors = eigsh(
                B, k=nev, which='LA', ncv=ncv, v0=v0, tol=1e-6, maxiter=N * 3
            )
            v[:, :nev] = np.flip(eigenvectors, axis=1)
            eigenvalues_active = np.flip(eigenvalues)
        except Exception:
            eigenvalues_active, eigenvectors = eigh(
                B, subset_by_index=[N - nev, N - 1], driver='evr'
            )
            v[:, :nev] = np.flip(eigenvectors, axis=1)
    else:
        eigenvalues, eigenvectors = eigh(
            B, subset_by_index=[N - nev, N - 1], driver='evr'
        )
        v[:, :nev] = np.flip(eigenvectors, axis=1)
        eigenvalues_active = np.flip(eigenvalues)

    if compute_u and (u is not None):
        U_unscaled = A @ v[:, :nev]
        u[:, :nev] = 0.0
        for idx in range(nev):
            norm_u = float(np.linalg.norm(U_unscaled[:, idx]))
            if norm_u > 1e-9:
                u[:, idx] = U_unscaled[:, idx] / norm_u
                s[idx] = norm_u
            else:
                s[idx] = 0.0
    else:
        s[:nev] = np.sqrt(np.maximum(eigenvalues_active[:nev], 0.0))

    return u, s, v


def _estimate_nev(
    Xa: NDArray[np.float32],
    N: int,
    target_var: float = 0.95,
    max_nev: int = 30,
    min_nev: int = 6,
    subsample_ratio: float = 1.0,
    subsample_min: int = 5_000_000,
    seed: int | None = 42
) -> int:
    """Estimate an EOF rank from the cumulative explained variance.

    The estimate is intentionally conservative: three extra modes are added so
    that cross-validation can inspect a small neighborhood around the variance
    target. The returned rank is always strictly smaller than ``N``.
    """
    if N <= min_nev + 1:
        return max(1, N - 1)

    k_search = min(max_nev + 5, N - 1)

    # Replace cloud gaps (NaN) with 0.0 for initial covariance estimation
    Xa_clean = np.nan_to_num(Xa, nan=0.0, copy=True)
    B0 = _parallel_AtA(Xa_clean, subsample_ratio=subsample_ratio, subsample_min=subsample_min, seed=seed)

    # Numerical safety check
    B0 = np.nan_to_num(B0, nan=0.0)
    total_var = float(np.trace(B0))

    if total_var <= 1e-12:
        return min(min_nev, N - 1)

    # Spectral extraction without check_finite blocking
    w_all = eigh(
        B0,
        subset_by_index=[N - k_search, N - 1],
        eigvals_only=True,
        check_finite=False
    )
    w = np.flip(np.maximum(w_all, 0.0))

    # Cumulative variance ratio
    cum_var = np.cumsum(w) / total_var
    idx_target = np.where(cum_var >= target_var)[0]
    k_target = int(idx_target[0] + 1) if len(idx_target) > 0 else k_search

    # Safety margin (+3 modes) for cv_patience exploration
    k_est = k_target + 3

    return int(min(max(k_est, min_nev), max_nev, N - 1))


@jit(nopython=True, parallel=True, fastmath=True, nogil=True)
def _jit_reconstruct_and_update(
    chunk: NDArray[np.float32],
    b_proj: NDArray[np.float32],
    v_T: NDArray[np.float32],
    mask: NDArray[np.bool_]
) -> float:
    """Reconstruct and replace only masked values in one spatial chunk.

    The squared change is accumulated for the convergence criterion. Observed
    values are read but never modified.
    """
    M_c, N_c = chunk.shape
    nev = b_proj.shape[1]
    diff_sq = 0.0

    # Distribute rows across all allocated CPU threads
    for i in prange(M_c):
        for t in range(N_c):
            if mask[i, t]:
                val = 0.0
                for l in range(nev):
                    val += b_proj[i, l] * v_T[l, t]
                diff = val - chunk[i, t]
                diff_sq += diff * diff
                chunk[i, t] = val

    return diff_sq


def _parallel_update_orchestrator(
    Xa: NDArray[np.float32],
    v: NDArray[np.float32],
    nev: int,
    mask_miss: NDArray[np.bool_],
    buf_proj: NDArray[np.float32],
    buf_rec: NDArray[np.float32],
) -> float:
    """Project and reconstruct missing values chunk by chunk.

    Chunking bounds the temporary projection buffer by the configured chunk
    size instead of allocating a full ``(M, nev)`` work array.
    """
    M, N = Xa.shape
    v_active = v[:, :nev]
    v_active_T = np.ascontiguousarray(v_active.T)

    chunk_size = buf_proj.shape[0]
    diff_sq_total = 0.0

    for i in range(0, M, chunk_size):
        sl = slice(i, min(i + chunk_size, M))
        m = mask_miss[sl]

        if not np.any(m):
            continue

        chunk = Xa[sl]
        cur_size = chunk.shape[0]
        b_proj = buf_proj[:cur_size, :nev]

        # 1. Multi-threaded BLAS Level-3 projection (very small: cur_size x nev)
        np.matmul(chunk, v_active, out=b_proj)

        # 2. Multi-threaded JIT in-place update over missing coordinates only
        diff_sq_total += _jit_reconstruct_and_update(chunk, b_proj, v_active_T, m)

    return diff_sq_total


def compute_dineof_oi_error_step(
    U_k: NDArray[np.float32],
    s_k: NDArray[np.float32],
    N_time: int,
    sigma_o: float,
    obs_mask_t: NDArray[np.bool_],
    buf_Uk_invA: NDArray[np.float32] | None = None,
    b_diag: NDArray[np.float32] | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute the optimal-interpolation analysis error for one time step.

    The calculation uses the dual formulation in the reduced EOF subspace
    described by Beckers et al. (2006). The returned error is computed for all
    spatial points, including points that were not observed at this time step.

    Parameters
    ----------
    U_k : NDArray[np.float32]
        Spatial EOF modes of shape (M, k).
    s_k : NDArray[np.float32]
        Singular values of length k.
    N_time : int
        Total number of temporal steps in the dataset.
    sigma_o : float
        Estimated observation noise standard deviation (optimal CV RMSE).
    obs_mask_t : NDArray[np.bool_]
        Boolean observation mask at time step t (True for valid physical observations).
    buf_Uk_invA : NDArray[np.float32] or None
        Optional pre-allocated memory buffer of shape (M, k).
    b_diag : NDArray[np.float32] or None
        Optional precomputed background covariance diagonal (length M).

    Notes
    -----
    ``obs_mask_t`` refers to physical observations, not to the cross-validation
    points temporarily hidden during the DINEOF search. If no observations are
    available, the analysis error equals the background error.

    Returns
    -------
    err_var : NDArray[np.float32]
        Analysis error variance array of length M ([P_t]_{mm}).
    rel_err : NDArray[np.float32]
        Relative error ratio array of length M (r_t(m) = [P_t]_{mm} / [B]_{mm} in [0, 1]).
    """
    M = U_k.shape[0]
    lambda_k = (s_k ** 2) / float(N_time)
    inv_lambda = 1.0 / np.maximum(lambda_k, 1e-12)
    inv_sigma2 = 1.0 / max(sigma_o ** 2, 1e-12)

    # Prior background covariance diagonal: B_diag = sum(lambda_k * U_k^2)
    if b_diag is None:
        b_diag = np.sum((U_k * lambda_k) * U_k, axis=1)
    valid_b = b_diag > 1e-9

    if not np.any(obs_mask_t):
        err_var = b_diag.astype(np.float32)
        err_var[~valid_b] = np.nan
        rel_err = np.ones(M, dtype=np.float32)
        return err_var, rel_err

    # Solve the reduced dual system in k x k subspace
    U_obs = U_k[obs_mask_t, :]
    G = U_obs.T @ U_obs
    A = np.diag(inv_lambda) + inv_sigma2 * G
    inv_A = np.linalg.inv(A)

    if buf_Uk_invA is not None:
        np.matmul(U_k, inv_A, out=buf_Uk_invA)
        err_var = np.sum(buf_Uk_invA * U_k, axis=1)
    else:
        err_var = np.sum((U_k @ inv_A) * U_k, axis=1)

    err_var = np.maximum(err_var, 0.0)

    rel_err = np.ones(M, dtype=np.float32)
    rel_err[valid_b] = np.clip(err_var[valid_b] / b_diag[valid_b], 0.0, 1.0)
    err_var[~valid_b] = np.nan

    return err_var.astype(np.float32), rel_err.astype(np.float32)


# =============================================================================
#                   CROSS-VALIDATION PATCH SELECTION
# =============================================================================

def select_cv_blocks_multi(
    X: NDArray[np.float32],
    n_cv_total: int,
    n_patches: int = 3,
    mask_2d: NDArray[np.bool_] | None = None,
    patch_size: int = 11,
    seed: int | None = None
) -> list[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Generate disjoint validation patches that mimic spatial cloud gaps.

    Parameters
    ----------
    X : NDArray[np.float32]
        Data matrix of shape (M, N).
    n_cv_total : int
        Target total number of cross-validation points.
    n_patches : int
        Number of independent patch sets to extract.
    mask_2d : NDArray[np.bool_] or None
        2D boolean sea/water domain mask.
    patch_size : int
        Side length (in pixels) of the 2D square patches.
    seed : int or None
        Random seed for patch centroid selection.

    Returns
    -------
    list[tuple[NDArray[np.intp], NDArray[np.intp]]]
        List of ``(rows, cols)`` tuples. Each pair contains spatial and temporal
        indices of valid observations withheld for cross-validation. Returned
        patches are disjoint.

    Notes
    -----
    With ``mask_2d`` set, ``rows`` index the flattened water pixels represented
    by ``X`` and ``cols`` index time. Without a grid mask, spatial and temporal
    coordinates are sampled independently.
    """
    M, N = X.shape
    rng = np.random.default_rng(seed)
    n_per_patch = max(1, n_cv_total // n_patches)

    water_y, water_x = (np.where(mask_2d) if mask_2d is not None else (None, None))
    grid_lookup = None
    if mask_2d is not None:
        grid_lookup = np.full(mask_2d.shape, -1, dtype=np.int64)
        grid_lookup[water_y, water_x] = np.arange(M)

    half_w = patch_size // 2
    taken_mask = np.zeros((M, N), dtype=bool)
    patches = []

    for _ in range(n_patches):
        rows_list, cols_list = [], []
        count = 0
        attempts = 0

        while count < n_per_patch and attempts < n_per_patch * 20:
            attempts += 1
            if mask_2d is not None:
                c_idx = rng.integers(0, len(water_y))
                cy, cx = water_y[c_idx], water_x[c_idx]
                t = rng.integers(0, N)

                y_min = max(0, cy - half_w)
                y_max = min(mask_2d.shape[0], cy + half_w + 1)
                x_min = max(0, cx - half_w)
                x_max = min(mask_2d.shape[1], cx + half_w + 1)

                sub = grid_lookup[y_min:y_max, x_min:x_max]
                valid = sub[(sub != -1)]
                if len(valid) == 0:
                    continue

                obs_pts = valid[(~np.isnan(X[valid, t])) & (~taken_mask[valid, t])]
            else:
                batch = (n_per_patch - count) * 2
                r_pts = rng.integers(0, M, size=batch)
                c_pts = rng.integers(0, N, size=batch)
                valid_idx = (~np.isnan(X[r_pts, c_pts])) & (~taken_mask[r_pts, c_pts])
                obs_pts = r_pts[valid_idx]
                t = c_pts[valid_idx]

            if len(obs_pts) == 0:
                continue

            take = min(len(obs_pts), n_per_patch - count)
            sel_r = obs_pts[:take]
            sel_c = np.full(take, t, dtype=np.intp) if mask_2d is not None else t[:take]

            taken_mask[sel_r, sel_c] = True
            rows_list.append(sel_r)
            cols_list.append(sel_c)
            count += take

        if len(rows_list) > 0:
            patches.append((np.concatenate(rows_list).astype(np.intp), np.concatenate(cols_list).astype(np.intp)))

    return patches


# =============================================================================
#                                CORE DINEOF PIPELINE
# =============================================================================

def dineof(
    data: NDArray[np.float32],
    time_coords: NDArray | None = None,
    mask: NDArray[np.bool_] | None = None,
    apply_log: bool = False,
    use_climatology: bool = False,
    keep_observations: bool = True,
    threshold: float | None = None,
    nev: int | None = None,
    toliter: float = 1e-3,
    nitemax: int = 300,
    alpha: float = 0.005,
    numit: int = 3,
    cv_ratio: float = 0.02,
    cv_patch_size: int = 11,
    cv_max_points: int | None = 500_000,
    cv_patience: int = 3,
    n_cv_runs: int = 3,
    min_improvement: float = 0.0,
    subsample_ratio: float = 1.0,
    subsample_min: int = 5_000_000,
    seed: int | None = 42,
    verbose: bool = True,
    use_disk: bool = False,
    solver: str = 'dense',
    debug: bool = False
) -> tuple[NDArray[np.float32], dict]:
    """Reconstruct missing values with iterative DINEOF and cross-validation.

    Input values are centered internally, optionally transformed to logarithmic
    space, and reconstructed by progressively increasing the EOF rank. Original
    observations remain unchanged when ``keep_observations=True``. The optimal
    rank is selected from the pooled RMSE on withheld spatial patches.

    Parameters
    ----------
    data : NDArray[np.float32]
        2D matrix with shape ``(M, N)`` or 3D cube with shape ``(N, Y, X)``.
        Missing values must be represented by ``NaN``.
    time_coords : NDArray or None
        One-dimensional dates or timestamps of length ``N``. Numeric values are
        interpreted as relative time coordinates; datetime values are converted
        to decimal days.
    mask : NDArray[np.bool_] or None
        Boolean spatial domain mask with shape ``(Y, X)`` for 3D input. ``True``
        identifies pixels included in the reconstruction. If omitted, pixels
        containing at least one non-NaN value are used.
    apply_log : bool
        Apply ``ln(X)`` before reconstruction and ``exp`` after reconstruction.
        Non-positive values are treated as missing.
    use_climatology : bool
        Remove each spatial point's temporal mean before reconstruction and add
        it back to the result. If false, remove and restore one global mean.
    keep_observations : bool
        If True, preserves original physical observations intact in the output.
    threshold : float or None
        Physical lower bound. Values less than or equal to it are treated as
        missing before the optional logarithmic transform.
    nev : int or None
        Maximum number of EOF modes to explore (must satisfy nev < N). If None, estimated automatically.
    toliter : float
        Relative convergence tolerance, measured as ``RMS(delta) / stdvini``.
    nitemax : int
        Maximum iterations allowed per mode.
    alpha : float
        Laplacian temporal diffusion filter parameter.
    numit : int
        Number of temporal diffusion passes applied to the covariance matrix.
    cv_ratio : float
        Approximate fraction of valid observations withheld for cross-validation.
    cv_patch_size : int
        Side length (in pixels) of 2D validation cloud patches.
    cv_max_points : int or None
        Upper bound on withheld points, limiting validation memory and runtime.
    cv_patience : int
        Number of consecutive modes with no new CV minimum before stopping.
    n_cv_runs : int
        Number of independent spatial patch sets used for validation.
    min_improvement : float
        Minimum relative CV improvement required to test another mode. ``0``
        disables this plateau criterion.
    subsample_ratio : float
        Fraction of spatial rows used for covariance estimation; ``1.0`` is exact.
    subsample_min : int
        Minimum spatial size before covariance subsampling is activated.
    seed : int or None
        Random seed for reproducible cross-validation patch extraction.
    verbose : bool
        Whether to display execution logs and iteration diagnostics.
    use_disk : bool
        Store the working matrix in a temporary ``numpy.memmap`` file to reduce
        RAM usage. The returned output may also be a memmap.
    solver : str, default='dense'
        Choice of solver ('auto', 'dense', 'lanczos', 'randomized').
    debug : bool, default=False
        Collect fine-grained timings in ``meta['timings']``.

    Returns
    -------
    X_out : NDArray[np.float32]
        Reconstructed field in the input shape and physical units. Observed
        values are preserved when ``keep_observations`` is true.
    meta : dict
        Metadata containing ``n_opt``, ``valc``, ``valcopt``, ``var_ratio``,
        ``u``, ``s``, ``v``, ``obs_mask``, ``cv_rows``, ``cv_cols``,
        ``cv_truth``, ``cv_pred``, ``final_iters`` and ``disk_file``. When
        ``debug=True``, detailed timings are available in ``meta['timings']``.

    Raises
    ------
    ValueError
        If fewer than 100 valid observations are available or if ``nev`` is not
        strictly smaller than the number of time steps.

    Notes
    -----
    The input array is copied before preprocessing, but the returned memmap can
    point to a temporary file when ``use_disk=True``. Temporary files are
    removed when they are no longer needed; a retained ``disk_file`` path in
    the metadata identifies the returned on-disk result.

    Example
    -------
    A minimal call is ``reconstructed, meta = dineof(data, nev=5)`` where
    ``data`` is a 2D or 3D floating-point array containing ``NaN`` gaps.
    """
    t_start_total = time.perf_counter()
    t_profiling: dict = {
        'prep': 0.0,
        'cv_setup': 0.0,
        'init_pass': 0.0,
        'search_total': 0.0,
        'svd_search': 0.0,
        'update_search': 0.0,
        'cv_eval': 0.0,
        'final_pass': 0.0,
        'ortho_u': 0.0,
        'energy': 0.0,
        'restore': 0.0,
        'modes_detail': []
    }

    t0_step = time.perf_counter()
    is_3d = (data.ndim == 3)
    if is_3d:
        n_t, n_y, n_x = data.shape
        if mask is None:
            mask = ~np.isnan(data).all(axis=0)
        X_work = np.empty((int(np.sum(mask)), n_t), dtype=np.float32)
        for t in range(n_t):
            X_work[:, t] = data[t][mask]
    else:
        X_work = data.copy()

    # Remove rows that cannot contribute to the covariance or reconstruction.
    if verbose:
        print("   [1/5] Ingesting domain data and filtering inactive spatial points...")
    nan_rows = np.isnan(X_work).all(axis=1)
    X_active = X_work[~nan_rows] if np.any(nan_rows) else X_work

    M, N = X_active.shape

    # Reuse one spatial sample for every spectral step to keep the estimate
    # deterministic and avoid drawing a new sample at each iteration.
    if (subsample_ratio < 1.0) and (M > subsample_min):
        target = min(max(int(M * subsample_ratio), subsample_min), M)
        rng = np.random.default_rng(seed)
        idx_sub_global = np.sort(rng.choice(M, size=target, replace=False))
        subsample_scale = float(M / target)
    else:
        idx_sub_global = None
        subsample_scale = 1.0

    if nev is None:
        if verbose:
            print("   [2/5] Estimating optimal rank ceiling from initial covariance spectrum...")
        nev = _estimate_nev(
            Xa=X_active,
            N=N,
            target_var=0.95,
            max_nev=30,
            min_nev=6,
            subsample_ratio=subsample_ratio,
            subsample_min=subsample_min,
            seed=seed
        )
        if verbose:
            print(f"Estimated optimal number of EOF modes: {nev} (temporal dimension: {N}).")

    if nev >= N:
        raise ValueError(f"nev ({nev}) must be strictly less than temporal dimension ({N}).")

    active_solver = _resolve_solver(solver, N, nev)
    if verbose:
        print(f"Spectral solver selected: {active_solver} (configured: {solver})")

    if threshold is not None:
        X_active[X_active <= threshold] = np.nan

    obs_mask_original = (~np.isnan(X_active)).copy()
    n_miss = int(np.isnan(X_active).sum())
    n_valid = M * N - n_miss

    if n_valid < 100:
        raise ValueError("Insufficient valid observations to compute DINEOF.")

    # Transform to the numerical working space, then center the field.
    if verbose:
        print("   [3/5] Applying data transformations and spatial climatology centering...")
    if apply_log:
        X_active[X_active <= 0.0] = np.nan
        np.log(X_active, out=X_active)

    if use_climatology:
        clim = np.nanmean(X_active, axis=1, keepdims=True)
        global_mean = float(np.nanmean(X_active))
        clim = np.where(np.isnan(clim), global_mean, clim)
        X_active -= clim
    else:
        global_mean = float(np.nanmean(X_active))
        X_active -= global_mean
        clim = global_mean

    # Normalize all supported time representations to decimal days from t0.
    if time_coords is None:
        arr_time = np.arange(N, dtype=np.float32)
    else:
        if isinstance(time_coords, list):
            time_coords = np.array(time_coords)
        if np.issubdtype(time_coords.dtype, np.integer) or np.issubdtype(time_coords.dtype, np.floating):
            ts = (time_coords - time_coords[0]).astype(np.float32)
        else:
            ts = (pd.to_datetime(time_coords).values.astype('datetime64[ns]') - np.datetime64('1970-01-01')) / np.timedelta64(1, 's')
            ts = ((ts - ts[0]) / 86400.0).astype(np.float32)
        arr_time = ts

    t_profiling['prep'] = time.perf_counter() - t0_step

    # Hide a reproducible subset of observed pixels for out-of-sample scoring.
    t0_step = time.perf_counter()
    n_cv = int(cv_ratio * n_valid)
    if cv_max_points is not None:
        n_cv = min(n_cv, cv_max_points)
    n_cv = max(min(n_cv, int(0.05 * n_valid)), min(30, n_valid))

    active_mask_2d = None
    if mask is not None:
        active_mask_2d = mask.copy()
        if np.any(nan_rows):
            water_coords = np.where(mask)
            active_mask_2d[water_coords[0][nan_rows], water_coords[1][nan_rows]] = False

    patches = select_cv_blocks_multi(
        X_active, n_cv, n_patches=n_cv_runs, mask_2d=active_mask_2d,
        patch_size=cv_patch_size, seed=seed
    )

    mask_miss = np.isnan(X_active)
    Xo_cv_list = []
    for f_rows, f_cols in patches:
        mask_miss[f_rows, f_cols] = True
        Xo_cv_list.append(X_active[f_rows, f_cols].copy())

    n_total_miss = int(mask_miss.sum())
    np.nan_to_num(X_active, copy=False)

    n_jobs = int(os.environ.get('OMP_NUM_THREADS') or min(cpu_count(), 64))
    chunk_size = int(np.ceil(M / max(8, n_jobs // 2)))

    # Compute the initial variance in float64 to avoid accumulation overflow.
    varini = float(sum(np.sum(X_active[i:min(i + chunk_size, M)] ** 2, dtype=np.float64) for i in range(0, M, chunk_size)) / n_valid)
    stdvini = float(np.sqrt(varini))

    u = np.empty((M, nev), dtype=np.float32)
    s = np.empty((nev,), dtype=np.float32)
    v = np.empty((N, nev), dtype=np.float32)

    buf_proj = np.empty((chunk_size, nev), dtype=np.float32)
    buf_rec = np.empty((chunk_size, N), dtype=np.float32)

    scratch_dir = os.environ.get('JOBLIB_TEMP_FOLDER', '/tmp')

    if use_disk:
        xa_path = os.path.join(scratch_dir, f"Xa_{os.getpid()}_{int(time.time()*1000)}.dat")
        Xa = np.memmap(xa_path, dtype=np.float32, mode='w+', shape=(M, N))
        chunk_init = int(np.ceil(M / max(32, n_jobs * 2)))
        for i in range(0, M, chunk_init):
            sl = slice(i, min(i + chunk_init, M))
            Xa[sl] = X_active[sl]
        Xa.flush()
        del X_active
        if verbose:
            print(f"Out-of-Core activated: matrix mapped on disk ({xa_path}).")
    else:
        Xa = X_active.copy()

    for f_rows, f_cols in patches:
        Xa[f_rows, f_cols] = 0.0
    if use_disk and hasattr(Xa, 'flush'):
        Xa.flush()

    best_path = os.path.join(scratch_dir, f"Xmiss_{os.getpid()}_{int(time.time()*1000)}.dat")
    Xmiss_best = np.memmap(best_path, dtype=np.float32, mode='w+', shape=(n_total_miss,))

    total_cv_points = sum(len(f_r) for f_r, _ in patches)
    t_profiling['cv_setup'] = time.perf_counter() - t0_step

    if verbose:
        print(f"DINEOF initialized: {M} spatial points x {N} dates.")
        print(f"Cross-validation: {len(patches)} patch sets ({total_cv_points} masked evaluation points).")

    # Start with one mode so later rank increments have a stable initialization.
    if verbose:
        print("   [5/5] Performing progressive initialization (Mode 1 warm-up pass)...")
    t0_step = time.perf_counter()
    _ssvd_spectral(Xa, 1, arr_time, alpha, numit, u, s, v, compute_u=False,
               idx_sub=idx_sub_global, scale=subsample_scale, solver=active_solver, seed=seed)
    _parallel_update_orchestrator(Xa, v, 1, mask_miss, buf_proj, buf_rec)
    if verbose:
        print(f"\nDINEOF engine ready: {M} spatial points x {N} dates.")
        print(f"Cross-validation: {len(patches)} patch sets ({total_cv_points} masked evaluation points).")
        print("Starting iterative mode exploration:\n")

    t_profiling['init_pass'] = time.perf_counter() - t0_step

    valc = []
    valcopt = 1e10
    popt = 0
    nitedone = np.zeros(nev, dtype=np.uint16)
    cv_pred_out_of_sample = None
    t0_search = time.perf_counter()    
    
    for p in range(nev):
        t0_mode = time.perf_counter()
        t_mode_svd = 0.0
        t_mode_update = 0.0

        for nite in range(nitemax):
            if debug:
                t0_op = time.perf_counter()
            _ssvd_spectral(Xa, p + 1, arr_time, alpha, numit, u, s, v, compute_u=False,
               idx_sub=idx_sub_global, scale=subsample_scale, solver=active_solver, seed=seed)
            if debug:
                t_mode_svd += time.perf_counter() - t0_op

            if debug:
                t0_op = time.perf_counter()
            diff_sq = _parallel_update_orchestrator(Xa, v, p + 1, mask_miss, buf_proj, buf_rec)
            if debug:
                t_mode_update += time.perf_counter() - t0_op

            nitedone[p] = nite + 1
            if float(np.sqrt(diff_sq / n_total_miss) / stdvini) < toliter:
                break

            # Score all withheld patches together so each withheld pixel has equal
            # weight regardless of the patch in which it was sampled.
        if debug:
            t0_op = time.perf_counter()
        sq_err_total = sum(
            float(((Xa[f_r, f_c] - f_true) ** 2).sum())
            for (f_r, f_c), f_true in zip(patches, Xo_cv_list)
        )
        pooled_cv_rmse = float(np.sqrt(sq_err_total / max(total_cv_points, 1)))
        valc.append(pooled_cv_rmse)

        if pooled_cv_rmse < valcopt:
            valcopt = pooled_cv_rmse
            popt = p
            np.copyto(Xmiss_best, Xa[mask_miss])
            cv_pred_out_of_sample = np.concatenate([Xa[f_r, f_c].copy() for f_r, f_c in patches])

        t_mode_cv = (time.perf_counter() - t0_op) if debug else 0.0
        t_mode_elapsed = time.perf_counter() - t0_mode

        if debug:
            t_profiling['svd_search'] += t_mode_svd
            t_profiling['update_search'] += t_mode_update
            t_profiling['cv_eval'] += t_mode_cv
            t_profiling['modes_detail'].append({
                'mode': p + 1,
                'iters': int(nitedone[p]),
                'time_total': float(t_mode_elapsed),
                'time_svd': float(t_mode_svd),
                'time_update': float(t_mode_update),
                'time_cv': float(t_mode_cv)
            })

        if verbose:
            if debug:
                pct_svd = (t_mode_svd / max(t_mode_elapsed, 1e-6)) * 100.0
                pct_upd = (t_mode_update / max(t_mode_elapsed, 1e-6)) * 100.0
                print(
                    f"  Mode {p+1:2d}/{nev:2d} | CV RMSE: {pooled_cv_rmse:.6f} | Iters: {nitedone[p]:3d} | "
                    f"Time: {t_mode_elapsed:6.2f}s [SVD: {t_mode_svd:5.2f}s ({pct_svd:4.1f}%) | "
                    f"Update: {t_mode_update:5.2f}s ({pct_upd:4.1f}%)]"
                )
            else:
                print(f"  Mode {p+1:2d}/{nev:2d} | Pooled CV RMSE: {pooled_cv_rmse:.6f} | Iterations: {nitedone[p]:3d}")

        if p > 0:
            if (p - popt) >= cv_patience:
                if verbose:
                    print(f"  Early stopping: CV minimum reached at mode k = {popt + 1}")
                break
            improvement = (valc[p - 1] - valc[p]) / valc[p - 1]
            if min_improvement > 0 and improvement < min_improvement:
                if verbose:
                    print(f"  Early stopping: CV plateau reached ({improvement:.2%} < {min_improvement:.2%})")
                break

    t_profiling['search_total'] = time.perf_counter() - t0_search

    # Restore validation observations, then converge once at the selected rank.
    t0_step = time.perf_counter()
    n_opt = popt + 1
    Xa[mask_miss] = Xmiss_best
    for (f_r, f_c), f_true in zip(patches, Xo_cv_list):
        Xa[f_r, f_c] = f_true
        mask_miss[f_r, f_c] = False

    if verbose:
        print(f"Final convergence pass with CV points restored ({n_opt} modes)...")

    final_iters = 0
    for nite in range(nitemax):
        _ssvd_spectral(Xa, n_opt, arr_time, alpha, numit, u, s, v, compute_u=False,
                    idx_sub=idx_sub_global, scale=subsample_scale, solver=active_solver, seed=seed)
        diff_sq = _parallel_update_orchestrator(Xa, v, n_opt, mask_miss, buf_proj, buf_rec)
        final_iters = nite + 1
        if float(np.sqrt(diff_sq / n_miss) / stdvini) < toliter:
            break
    t_profiling['final_pass'] = time.perf_counter() - t0_step

    # Compute the final normalized spatial EOFs and singular values.
    t0_step = time.perf_counter()
    u, s, v = _ssvd_spectral(Xa, n_opt, arr_time, alpha, numit, u, s, v, compute_u=True,
                         idx_sub=idx_sub_global, scale=subsample_scale, solver=active_solver, seed=seed)

    if not keep_observations:
        u_scaled = u[:, :n_opt] * s[:n_opt]
        v_T = np.ascontiguousarray(v[:, :n_opt].T)
        for i in range(0, M, chunk_size):
            sl = slice(i, min(i + chunk_size, M))
            np.matmul(u_scaled[sl], v_T, out=Xa[sl])
    t_profiling['ortho_u'] = time.perf_counter() - t0_step

    # Measure retained spectral energy in float64 for numerical robustness.
    t0_step = time.perf_counter()
    total_field_energy = float(sum(np.sum(Xa[i:min(i + chunk_size, M)] ** 2, dtype=np.float64) for i in range(0, M, chunk_size)))
    recon_energy = float(np.sum(s[:n_opt] ** 2))
    var_ratio = float(recon_energy / total_field_energy) if total_field_energy > 0.0 else 0.0
    varend = float(recon_energy / float(M * N))
    t_profiling['energy'] = time.perf_counter() - t0_step

    if hasattr(Xmiss_best, '_mmap'):
        Xmiss_best._mmap.close()
    del Xmiss_best
    if os.path.exists(best_path):
        try:
            os.remove(best_path)
        except OSError:
            pass

    # Undo centering and optional logarithmic scaling before rebuilding the
    # original 2D or 3D domain.
    t0_step = time.perf_counter()
    Xa += clim
    if apply_log:
        np.exp(Xa, out=Xa)

    cv_r_sub = np.concatenate([f[0] for f in patches])
    cv_c_all = np.concatenate([f[1] for f in patches])
    cv_truth_all = np.concatenate(Xo_cv_list)

    c_cv = clim[cv_r_sub, 0] if use_climatology else clim
    cv_truth_phys = cv_truth_all + c_cv
    cv_pred_phys = cv_pred_out_of_sample + c_cv

    if apply_log:
        cv_truth_phys = np.exp(cv_truth_phys)
        cv_pred_phys = np.exp(cv_pred_phys)

    # Reconstruct domain-wide matrices matching original inputs
    out_disk_path = None

    if np.any(nan_rows):
        total_rows = len(nan_rows)
        valid_indices = np.where(~nan_rows)[0]
        cv_r_all = valid_indices[cv_r_sub]

        if use_disk:
            out_disk_path = os.path.join(scratch_dir, f"Xrec_{os.getpid()}_{int(time.time()*1000)}.dat")
            X_rec_full = np.memmap(out_disk_path, dtype=np.float32, mode='w+', shape=(total_rows, N))
            X_rec_full[:] = np.nan
            chunk_copy = int(np.ceil(M / max(32, n_jobs * 2)))
            for i in range(0, M, chunk_copy):
                sl_src = slice(i, min(i + chunk_copy, M))
                X_rec_full[valid_indices[sl_src], :] = Xa[sl_src, :]
            X_rec_full.flush()

            if hasattr(Xa, '_mmap'):
                Xa._mmap.close()
            del Xa
            if os.path.exists(xa_path):
                try:
                    os.remove(xa_path)
                except OSError:
                    pass
        else:
            X_rec_full = np.full((total_rows, N), np.nan, dtype=np.float32)
            X_rec_full[~nan_rows] = Xa

        U_full = np.zeros((total_rows, n_opt), dtype=np.float32)
        U_full[~nan_rows] = u[:, :n_opt]

        obs_mask_full = np.zeros((total_rows, N), dtype=bool)
        obs_mask_full[~nan_rows] = obs_mask_original

        if use_climatology and isinstance(clim, np.ndarray):
            clim_full = np.full((total_rows, 1), np.nan, dtype=np.float32)
            clim_full[~nan_rows] = clim
        else:
            clim_full = clim
    else:
        if use_disk:
            X_rec_full = Xa
            out_disk_path = xa_path
        else:
            X_rec_full = Xa

        U_full = u[:, :n_opt]
        obs_mask_full = obs_mask_original
        cv_r_all = cv_r_sub
        clim_full = clim

    if is_3d:
        if use_disk:
            out_3d_path = os.path.join(scratch_dir, f"Xout3d_{os.getpid()}_{int(time.time()*1000)}.dat")
            X_out = np.memmap(out_3d_path, dtype=np.float32, mode='w+', shape=(n_t, n_y, n_x))
            X_out[:] = np.nan
            for t in range(n_t):
                X_out[t, mask] = X_rec_full[:, t]
            X_out.flush()
            out_disk_path = out_3d_path
        else:
            X_out = np.full((n_t, n_y, n_x), np.nan, dtype=np.float32)
            for t in range(n_t):
                X_out[t, mask] = X_rec_full[:, t]
    else:
        X_out = X_rec_full

    t_profiling['restore'] = time.perf_counter() - t0_step
    t_total_elapsed = time.perf_counter() - t_start_total
    t_profiling['total'] = t_total_elapsed

    if debug and verbose:
        print("\n" + "=" * 66)
        print("               DINEOF EXECUTION PROFILE (TIMINGS)")
        print("=" * 66)
        print(f"  {'Operation / Stage':<35} {'Duration (s)':>12} {'Ratio':>10}")
        print("-" * 66)
        print(f"  {'Data Preprocessing & Centering':<35} {t_profiling['prep']:12.2f}s {t_profiling['prep']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Spatial CV Patch Generation':<35} {t_profiling['cv_setup']:12.2f}s {t_profiling['cv_setup']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Mode 1 Progressive Initialization':<35} {t_profiling['init_pass']:12.2f}s {t_profiling['init_pass']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Iterative Mode Exploration (Search)':<35} {t_profiling['search_total']:12.2f}s {t_profiling['search_total']/t_total_elapsed*100:9.1f}%")
        print(f"  {'  - Eigensolvers (_ssvd_spectral)':<35} {t_profiling['svd_search']:12.2f}s {t_profiling['svd_search']/t_total_elapsed*100:9.1f}%")
        print(f"  {'  - Chunked Updates (Orchestrator)':<35} {t_profiling['update_search']:12.2f}s {t_profiling['update_search']/t_total_elapsed*100:9.1f}%")
        print(f"  {'  - Cross-Validation Error Eval':<35} {t_profiling['cv_eval']:12.2f}s {t_profiling['cv_eval']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Final Convergence Pass (Restored CV)':<35} {t_profiling['final_pass']:12.2f}s {t_profiling['final_pass']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Final Spatial Modes U & Ortho':<35} {t_profiling['ortho_u']:12.2f}s {t_profiling['ortho_u']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Variance Energy Diagnostics':<35} {t_profiling['energy']:12.2f}s {t_profiling['energy']/t_total_elapsed*100:9.1f}%")
        print(f"  {'Domain Rescaling & Restoration':<35} {t_profiling['restore']:12.2f}s {t_profiling['restore']/t_total_elapsed*100:9.1f}%")
        print("-" * 66)
        print(f"  {'Total DINEOF Runtime':<35} {t_total_elapsed:12.2f}s {'100.0%':>10}")
        print("=" * 66 + "\n")

    meta = {
        'n_opt': n_opt,
        'valc': np.array(valc, dtype=np.float32),
        'valcopt': float(valcopt),
        'sigma_o': float(valcopt),
        'varini': varini,
        'varend': varend,
        'var_ratio': var_ratio,
        'u': U_full,
        's': s[:n_opt].copy(),
        'v': v[:, :n_opt].copy(),
        'obs_mask': obs_mask_full,
        'cv_rows': cv_r_all,
        'cv_cols': cv_c_all,
        'cv_truth': cv_truth_phys,
        'cv_pred': cv_pred_phys,
        'final_iters': final_iters,
        'apply_log': apply_log,
        'use_climatology': use_climatology,
        'keep_observations': keep_observations,
        'clim': clim_full,
        'disk_file': out_disk_path,
        'nev_used': nev,
        'timings': t_profiling if debug else None
    }

    if verbose:
        print(f"DINEOF completed: {n_opt} modes retained | Explained variance: {meta['var_ratio']:.2%}")

    return X_out, meta