import numpy as np
import pandas as pd
import pytest

from oilfc.eval.calibration import mincer_zarnowitz
from oilfc.eval.multiplicity import holm_adjust


def test_holm_matches_a_hand_worked_example():
    # sorted 0.01, 0.03, 0.04 -> x3, x2, x1 = 0.03, 0.06, 0.04 -> running max 0.03, 0.06, 0.06
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_holm_is_capped_at_one_and_never_below_the_raw_p():
    p = np.array([0.4, 0.8, 0.9, 0.001])
    adj = holm_adjust(p)
    assert (adj <= 1.0).all()
    assert (adj >= p - 1e-12).all()


def test_holm_keeps_nan_in_place_and_ignores_it_in_the_count():
    adj = holm_adjust([0.01, np.nan, 0.03])
    assert np.isnan(adj[1])
    assert np.allclose([adj[0], adj[2]], [0.02, 0.03])  # two tests: x2, x1


def test_holm_on_an_empty_list_is_empty():
    assert len(holm_adjust([])) == 0


def _series(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    f = pd.Series(np.exp(rng.normal(-1.5, 0.4, n)), index=idx)
    return rng, f


def test_a_calibrated_forecast_has_slope_one_and_is_not_rejected():
    rng, f = _series()
    y = f * np.exp(rng.normal(0, 0.2, len(f)))
    out = mincer_zarnowitz(y, f, h=5)
    assert abs(out["slope"] - 1) < 0.1
    assert out["wald_p"] > 0.01
    assert abs(out["mean_log_ratio"]) < 0.03


def test_a_forecast_twice_too_high_is_flagged_as_biased():
    rng, f = _series()
    y = f * np.exp(rng.normal(0, 0.2, len(f)))
    out = mincer_zarnowitz(y, 2 * f, h=5)
    assert out["wald_p"] < 1e-6
    assert out["mean_log_ratio"] == pytest.approx(-np.log(2), abs=0.03)


def test_overlap_widens_the_standard_error():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2012-01-02", periods=1500)
    level = pd.Series(rng.normal(0, 1, 1500), index=idx).rolling(30, min_periods=1).mean() * 3
    f = np.exp(-1.5 + level)  # persistent forecast, like real volatility
    err = pd.Series(rng.normal(0, 1, 1500), index=idx).rolling(22, min_periods=1).mean() * 2
    y = f * np.exp(err)  # errors overlap, as h-day targets do
    short = mincer_zarnowitz(y, f, h=1)
    long = mincer_zarnowitz(y, f, h=22)
    assert long["slope_se"] > 1.5 * short["slope_se"]


def test_too_few_rows_raise():
    rng, f = _series(n=20)
    with pytest.raises(ValueError):
        mincer_zarnowitz(f, f, h=22)
