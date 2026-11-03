"""Cross-validation points in the final pass: ``exclude_cv_in_final`` (default True, as DINEOF GHER) or False (restored as data)."""
import numpy as np
import pytest

from dineof_hpc.core.main import dineof
from test_oi_error import _synthetic


def _run(X, mask, t, exclude, nev=8):
    return dineof(data=X, time_coords=t, mask=mask, apply_log=False, use_climatology=False, nev=nev, cv_ratio=0.05, seed=3,
                  verbose=False, exclude_cv_in_final=exclude)


@pytest.fixture(scope="module")
def case():
    X, mask, t = _synthetic(seed=2)
    return X, mask, t


def test_default_is_exclusion(case):
    X, mask, t = case
    _, meta = dineof(data=X, time_coords=t, mask=mask, apply_log=False, use_climatology=False, nev=8, cv_ratio=0.05, seed=3, verbose=False)
    assert meta["exclude_cv_in_final"] is True


@pytest.mark.parametrize("exclude", [True, False])
def test_observations_kept_everywhere_including_cv_points(case, exclude):
    X, mask, t = case
    rec, meta = _run(X, mask, t, exclude)
    obs = np.isfinite(X)
    np.testing.assert_allclose(rec[obs], X[obs], rtol=0, atol=1e-6)
    r, c = meta["cv_rows"], meta["cv_cols"]
    assert obs[r, c].all()                                   # the CV points are genuine observations
    np.testing.assert_allclose(rec[r, c], X[r, c], rtol=0, atol=1e-6)


def test_gap_filling_does_not_depend_on_cv_values_when_excluded(case):
    """With exclusion the held-out values play no role in the fit (rank forced to 1 so that the CV score cannot change it; mean and variance of the data are unchanged by the shuffle), so shuffling them cannot change the filled gaps; with False it does."""
    X, mask, t = case
    _, meta = _run(X, mask, t, True, nev=1)
    r, c = meta["cv_rows"], meta["cv_cols"]
    gaps = ~np.isfinite(X)
    Xp = X.copy()
    Xp[r, c] = Xp[r, c][np.random.default_rng(0).permutation(len(r))]   # same values, shuffled: mean and variance of the data unchanged
    rec_a, _ = _run(X, mask, t, True, nev=1)
    rec_b, _ = _run(Xp, mask, t, True, nev=1)
    np.testing.assert_allclose(rec_a[gaps], rec_b[gaps], rtol=1e-4, atol=1e-4)
    rec_c, _ = _run(X, mask, t, False, nev=1)
    rec_d, _ = _run(Xp, mask, t, False, nev=1)
    assert np.max(np.abs(rec_c[gaps] - rec_d[gaps])) > 1e-2


def test_both_modes_select_similar_models_and_stay_close(case):
    X, mask, t = case
    ra, ma = _run(X, mask, t, True)
    rb, mb = _run(X, mask, t, False)
    assert ma["n_opt"] == mb["n_opt"]
    gaps = ~np.isfinite(X)
    assert np.sqrt(np.mean((ra[gaps] - rb[gaps]) ** 2)) < 0.5 * np.nanstd(X)
