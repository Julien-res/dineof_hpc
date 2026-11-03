"""Cross-validation hold-out: isolated points drawn like DINEOF GHER (the 2D patch hold-out was removed)."""
import warnings

import numpy as np
import pytest

from dineof_hpc.core.main import dineof, select_cv_points
from test_oi_error import _synthetic


def test_points_are_observed_disjoint_and_reproducible():
    X, mask, t = _synthetic(seed=1)
    groups = select_cv_points(X, 300, mask_2d=mask, seed=5)
    again = select_cv_points(X, 300, mask_2d=mask, seed=5)
    rows = np.concatenate([g[0] for g in groups]); cols = np.concatenate([g[1] for g in groups])
    assert len(rows) == 300
    assert np.isfinite(X[rows, cols]).all()                        # only observed values are withheld
    assert len(set(zip(rows.tolist(), cols.tolist()))) == len(rows)  # no point is drawn twice
    for (r1, c1), (r2, c2) in zip(groups, again):
        np.testing.assert_array_equal(r1, r2); np.testing.assert_array_equal(c1, c2)


def test_points_without_mask():
    X = np.random.default_rng(0).normal(size=(200, 30)).astype(np.float32)
    X[X > 1.0] = np.nan
    rows, cols = [np.concatenate(z) for z in zip(*select_cv_points(X, 90, seed=2))]
    assert np.isfinite(X[rows, cols]).all() and len(rows) == 90


def test_points_are_isolated_not_blocks():
    X, mask, t = _synthetic(seed=3)
    rows, cols = [np.concatenate(z) for z in zip(*select_cv_points(X, 200, mask_2d=mask, seed=1))]
    same_date = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        same_date.setdefault(c, []).append(r)
    assert max(len(v) for v in same_date.values()) < 20           # spread over the dates, not 121-pixel blocks on one date


def test_patch_parameters_are_deprecated_and_ignored():
    X, mask, t = _synthetic(seed=2)
    kw = dict(data=X, time_coords=t, mask=mask, apply_log=False, use_climatology=False, nev=6, cv_ratio=0.05, seed=3, verbose=False)
    _, m0 = dineof(**kw)
    with pytest.warns(DeprecationWarning):
        _, m1 = dineof(cv_patch_size=11, n_cv_runs=3, **kw)
    np.testing.assert_array_equal(m0["cv_rows"], m1["cv_rows"])    # same points whatever the old patch settings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dineof(**kw)                                              # no warning with the defaults
