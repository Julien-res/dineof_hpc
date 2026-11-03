"""Tests of the space-time Barnes refinement (arbitrate_spacetime_refinement / apply_spacetime_refinement)."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from dineof_hpc.core.io_utils import (
    apply_spacetime_refinement,
    arbitrate_spacetime_refinement,
)


def _field(ny=24, nx=24, nt=30, seed=0, p_obs=0.5, st=1.5):
    rng = np.random.default_rng(seed)
    mask = np.ones((ny, nx), dtype=bool)
    mask[:3, :] = False
    t = np.arange(nt)[:, None, None]
    y, x = np.mgrid[:ny, :nx]
    truth = np.sin(t / 5.0) + 0.5 * np.cos((x + y)[None] / 6.0)
    # smooth local anomaly that the "reconstruction" misses
    anom = gaussian_filter(rng.standard_normal((nt, ny, nx)), (st, 3, 3)) * 2.0
    full = truth + anom
    rec = truth
    X_true = full[:, mask].T.astype(np.float32)
    X_rec = rec[:, mask].T.astype(np.float32)
    obs = rng.uniform(size=X_true.shape) < p_obs
    X_raw = np.where(obs, X_true, np.nan).astype(np.float32)
    return mask, X_true, X_rec, X_raw


def _meta(X_true, X_rec, X_raw, n=200, seed=1):
    rng = np.random.default_rng(seed)
    r, c = np.where(np.isfinite(X_raw))
    sel = rng.choice(len(r), n, replace=False)
    r, c = r[sel], c[sel]
    return {"cv_rows": r, "cv_cols": c, "cv_truth": X_true[r, c].astype(np.float64),
            "cv_pred": X_rec[r, c].astype(np.float64)}


def test_refinement_lowers_error_and_keeps_observations():
    mask, X_true, X_rec, X_raw = _field()
    meta = _meta(X_true, X_rec, X_raw)
    ok, s, st, pred = arbitrate_spacetime_refinement(X_rec, X_raw, mask, meta)
    assert ok and s is not None and st is not None
    rm = lambda a: float(np.sqrt(np.mean((a - meta["cv_truth"]) ** 2)))  # noqa: E731
    assert rm(pred) < rm(meta["cv_pred"])
    out = apply_spacetime_refinement(X_rec, X_raw, mask, s, st)
    obs = np.isfinite(X_raw)
    np.testing.assert_array_equal(out[obs], X_rec[obs])
    gap = ~obs
    assert np.sqrt(np.mean((out[gap] - X_true[gap]) ** 2)) < np.sqrt(np.mean((X_rec[gap] - X_true[gap]) ** 2))


def test_time_smoothing_helps_when_anomaly_persists():
    mask, X_true, X_rec, X_raw = _field(p_obs=0.12, st=30.0)
    meta = _meta(X_true, X_rec, X_raw, n=100)
    _, _, st, _ = arbitrate_spacetime_refinement(X_rec, X_raw, mask, meta, candidate_sigmas=[1.5], candidate_sigmas_t=[0.0, 2.0, 5.0])
    assert st > 0


def test_rejected_when_residual_is_noise():
    rng = np.random.default_rng(3)
    mask = np.ones((16, 16), dtype=bool)
    X_true = rng.standard_normal((256, 20)).astype(np.float32)
    X_rec = np.zeros_like(X_true)
    X_raw = np.where(rng.uniform(size=X_true.shape) > 0.5, X_true, np.nan).astype(np.float32)
    meta = _meta(X_true, X_rec, X_raw)
    ok, s, st, pred = arbitrate_spacetime_refinement(X_rec, X_raw, mask, meta)
    assert (not ok) and s is None and st is None
    np.testing.assert_array_equal(pred, meta["cv_pred"])


def test_spatial_only_matches_independent_gaussian():
    """sigma_t = 0: each date is smoothed independently with the normalised 2D kernel (cutoff 0.05)."""
    mask, X_true, X_rec, X_raw = _field(nt=6)
    out = apply_spacetime_refinement(X_rec, X_raw, mask, 3.5, 0.0)
    obs = np.isfinite(X_raw)
    res = np.where(obs, X_raw - X_rec, 0.0)
    ny, nx = mask.shape
    expect = X_rec.copy()
    for t in range(X_rec.shape[1]):
        R = np.zeros((ny, nx), np.float32); W = np.zeros((ny, nx), np.float32)
        R[mask] = res[:, t]; W[mask] = obs[:, t]
        num = gaussian_filter(R, 3.5, mode="constant"); den = gaussian_filter(W, 3.5, mode="constant")
        c = np.where(den > 0.05, num / np.where(den > 0.05, den, 1), 0.0)[mask]
        expect[~obs[:, t], t] += c[~obs[:, t]]
    np.testing.assert_allclose(out, expect, atol=1e-5)


def test_log_scale_keeps_positivity():
    mask, X_true, X_rec, X_raw = _field()
    X_true, X_rec, X_raw = (np.exp(a * 0.5) for a in (X_true, X_rec, X_raw))
    meta = _meta(X_true, X_rec, X_raw)
    ok, s, st, pred = arbitrate_spacetime_refinement(X_rec, X_raw, mask, meta, is_log_scale=True)
    assert ok
    out = apply_spacetime_refinement(X_rec, X_raw, mask, s, st, is_log_scale=True)
    assert np.all(out[np.isfinite(out)] > 0)


def test_pipeline_rejects_unknown_mode(tmp_path):
    from dineof_hpc.core.io_utils import run_reconstruction_pipeline
    with pytest.raises(ValueError):
        run_reconstruction_pipeline("x.nc", "m.nc", "v", str(tmp_path), local_refinement_mode="bad")
