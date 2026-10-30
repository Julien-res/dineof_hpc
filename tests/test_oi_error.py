"""Unit tests of the OI analysis error: effective observation variance, automatic correlation length, absolute variance."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from dineof_hpc.core.main import compute_dineof_oi_error_step, dineof
from dineof_hpc.core.io_utils import (
    effective_obs_sigma, estimate_error_corr_length_km, estimate_pixel_area_km2, resolve_corr_length_km)


def _synthetic(ny=32, nx=32, nt=40, noise_px=3.0, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:ny, :nx]
    mask = np.hypot(yy - ny / 2, xx - nx / 2) < ny / 2 - 1
    modes = [gaussian_filter(rng.normal(size=(ny, nx)), 6) for _ in range(3)]
    amps = rng.normal(size=(3, nt))
    field = sum(m[..., None] * a[None, None, :] for m, a in zip(modes, amps)) * 10.0
    noise = np.stack([gaussian_filter(rng.normal(size=(ny, nx)), noise_px) for _ in range(nt)], -1) * 5.0
    X = (field + noise)[mask]
    X[rng.random(X.shape) < 0.35] = np.nan
    t = np.datetime64("2020-01-01") + np.arange(nt).astype("timedelta64[D]")
    return X.astype(np.float32), mask, t


@pytest.fixture(scope="module")
def run():
    X, mask, t = _synthetic()
    _, meta = dineof(data=X, time_coords=t, mask=mask, apply_log=False, use_climatology=False, nev=8,
                     cv_ratio=0.05, seed=1, verbose=False)
    return X, mask, meta


def test_pixel_area_degrees_and_metres():
    assert estimate_pixel_area_km2(np.arange(0, 5) * 0.1, np.arange(0, 5) * 0.1) == pytest.approx(0.1 * 111.32 * 0.1 * 111.32, rel=0.01)
    assert estimate_pixel_area_km2(np.arange(5) * 1000.0 + 5e6, np.arange(5) * 1000.0 + 4e5) == pytest.approx(1.0)


def test_effective_sigma():
    assert effective_obs_sigma(0.3, None, 1.0) == 0.3
    assert effective_obs_sigma(0.3, 0.0, 1.0) == 0.3
    assert effective_obs_sigma(0.3, 5.0, None) == 0.3                       # unknown pixel size: no correction
    assert effective_obs_sigma(0.3, 0.5, 1.0) == 0.3                        # L below the pixel size: no correction
    assert effective_obs_sigma(0.3, 10.0, 1.0) == pytest.approx(3.0)         # sigma * L / sqrt(dx dy)


def test_oi_error_is_absolute_and_bounded_by_background(run):
    X, mask, meta = run
    U, s, N = meta['u'], meta['s'], X.shape[1]
    b = np.sum((U * (s ** 2 / N)) * U, axis=1)
    obs_t = meta['obs_mask'][:, 0]
    p_small, rel_small = compute_dineof_oi_error_step(U, s, N, 0.1, obs_t)
    p_big, rel_big = compute_dineof_oi_error_step(U, s, N, 10.0, obs_t)
    ok = b > 1e-9
    assert np.all(p_small[ok] <= b[ok] * (1 + 1e-4)) and np.all(p_big[ok] <= b[ok] * (1 + 1e-4))
    assert np.nanmean(p_big) > np.nanmean(p_small)                          # more observation noise, more analysis error
    p_none, rel_none = compute_dineof_oi_error_step(U, s, N, 0.1, np.zeros_like(obs_t))
    np.testing.assert_allclose(p_none[ok], b[ok], rtol=1e-4)                # no observation: error equals background
    assert np.all(rel_none == 1.0)
    np.testing.assert_allclose(rel_small[ok], np.clip(p_small[ok] / b[ok], 0, 1), rtol=1e-4)


def test_auto_length_matches_noise_scale(run):
    X, mask, meta = run
    area = 1.0                                                              # 1 km pixels
    est = estimate_error_corr_length_km(meta, X, mask, area, apply_log=False)
    assert est['ok'] and 1.0 <= est['L_km'] <= 200.0
    X2, mask2, t2 = _synthetic(noise_px=0.7, seed=0)                       # nearly white noise: shorter length
    _, meta2 = dineof(data=X2, time_coords=t2, mask=mask2, apply_log=False, use_climatology=False, nev=8,
                      cv_ratio=0.05, seed=1, verbose=False)
    est2 = estimate_error_corr_length_km(meta2, X2, mask2, area, apply_log=False)
    assert est2['ok'] and est2['L_km'] < est['L_km']


def test_resolve_fixed_none_auto_and_fallback(run):
    X, mask, meta = run
    ny, nx = mask.shape
    coords = {'lat': np.arange(ny) * 0.01, 'lon': np.arange(nx) * 0.01}
    assert resolve_corr_length_km(None, meta, X, mask, coords, False) == (None, None)
    assert resolve_corr_length_km(0, meta, X, mask, coords, False) == (None, None)
    assert resolve_corr_length_km(12.5, meta, X, mask, coords, False) == (12.5, None)
    L, info = resolve_corr_length_km('auto', meta, X, mask, coords, False)
    assert info['ok'] and L == info['L_km']
    L_fb, info_fb = resolve_corr_length_km('auto', meta, X, mask, {'lat': np.array([0.0]), 'lon': np.array([0.0])}, False)
    assert L_fb == 50.0 and not info_fb['ok']                               # unknown pixel size: 50 km fallback
    with pytest.raises(ValueError):
        resolve_corr_length_km('median', meta, X, mask, coords, False)


def test_alpha_zero_and_shrinkage_keep_error_finite():
    X, mask, t = _synthetic()
    for alpha in (0.0, 0.01):
        _, meta = dineof(data=X, time_coords=t, mask=mask, apply_log=False, use_climatology=False, nev=8,
                         cv_ratio=0.05, alpha=alpha, seed=1, verbose=False)
        p, rel = compute_dineof_oi_error_step(meta['u'], meta['s'], X.shape[1], meta['sigma_o'], meta['obs_mask'][:, 0])
        assert np.all(np.isfinite(rel)) and np.all((rel >= 0) & (rel <= 1))
        assert np.nanmin(p) >= 0.0
