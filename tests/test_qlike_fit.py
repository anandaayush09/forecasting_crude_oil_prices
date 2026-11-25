"""QLIKE and weighted-least-squares log-volatility fits, OVX features and the one-sided DM helper."""

import numpy as np
import pandas as pd
import pytest

from oilfc.eval.losses import qlike
from oilfc.eval.stat_tests import dm_better_p
from oilfc.models.baselines.qlike_fit import (
    _design,
    forecast_log_qlike,
    qlike_coefficients,
    wls_coefficients,
)
from oilfc.models.baselines.vol import ovx_features


def _sample(n=2000, seed=0, hetero=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    scale = np.exp(0.5 * x[:, 0]) * 0.3 if hetero else 0.3
    y = 0.5 + x @ np.array([0.8, -0.3]) + rng.normal(size=n) * scale
    return x, y


def test_the_qlike_fit_has_a_lower_training_qlike_than_ols_with_the_variance_correction():
    x, y = _sample(hetero=True)
    d = _design(x)
    ols, *_ = np.linalg.lstsq(d, y, rcond=None)
    ols[0] += float(np.var(y - d @ ols))
    fitted = qlike_coefficients(d, y)
    loss = lambda b: qlike(np.exp(y), np.exp(d @ b)).mean()  # noqa: E731
    assert loss(fitted) <= loss(ols) + 1e-12


def test_with_equal_error_variance_qlike_and_ols_agree_on_the_slopes():
    x, y = _sample(n=6000, seed=1)
    d = _design(x)
    ols, *_ = np.linalg.lstsq(d, y, rcond=None)
    assert np.allclose(qlike_coefficients(d, y)[1:], ols[1:], atol=0.05)


def test_the_wls_fit_matches_a_hand_computed_weighted_solution():
    x, y = _sample(hetero=True)
    d = _design(x)
    ols, *_ = np.linalg.lstsq(d, y, rcond=None)
    w = np.exp(-2 * (d @ ols))
    w = w / w.mean()
    expected = np.linalg.solve(d.T @ (d * w[:, None]), d.T @ (w * y))
    beta, s2 = wls_coefficients(d, y)
    assert np.allclose(beta, expected, atol=1e-8) and s2 > 0


@pytest.mark.parametrize("method", ["qlike", "wls"])
def test_forecasts_use_training_rows_only_and_are_nan_outside_test_blocks(method):
    x, y = _sample(n=600)
    idx = pd.bdate_range("2015-01-01", periods=600)
    X = pd.DataFrame(x, index=idx, columns=["a", "b"])
    ys = pd.Series(y, index=idx)
    splits = [(np.arange(0, 300), np.arange(305, 360)), (np.arange(0, 360), np.arange(365, 420))]
    full = forecast_log_qlike(X, ys, splits, ["a", "b"], method)
    assert full.iloc[:305].isna().all() and full.iloc[305:360].notna().all()
    ys2 = ys.copy()
    ys2.iloc[300:] = 99.0  # targets after the first split's training end must not matter for it
    changed = forecast_log_qlike(X, ys2, splits, ["a", "b"], method)
    pd.testing.assert_series_equal(full.iloc[305:360], changed.iloc[305:360])


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        forecast_log_qlike(pd.DataFrame({"a": [1.0]}), pd.Series([1.0]), [], ["a"], "huber")


def test_ovx_features_have_the_expected_columns_and_a_signed_gap():
    idx = pd.bdate_range("2020-01-01", periods=40)
    log_ovx = pd.Series(np.linspace(-1.0, -0.5, 40), index=idx)
    log_rv = pd.Series(-0.8, index=idx)
    f = ovx_features(log_ovx, log_rv)
    assert list(f.columns) == ["log_ovx_5", "log_ovx_22", "gap_pos", "gap_neg"]
    assert (f["gap_pos"] >= 0).all() and (f["gap_neg"] <= 0).all()
    assert f["gap_pos"].iloc[-1] == pytest.approx(0.3) and f["gap_neg"].iloc[0] == pytest.approx(
        -0.2
    )


def test_the_one_sided_p_is_small_for_a_better_model_and_near_one_for_a_worse_one():
    rng = np.random.default_rng(3)
    bench = np.abs(rng.normal(1.0, 0.2, 800))
    assert dm_better_p(bench, bench * 0.8 + rng.normal(0, 0.01, 800), 5) < 0.01
    assert dm_better_p(bench, bench * 1.3 + rng.normal(0, 0.01, 800), 5) > 0.99
    assert np.isnan(dm_better_p(bench, bench.copy(), 5))
