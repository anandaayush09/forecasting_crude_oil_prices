import numpy as np
import pandas as pd
import pytest

from oilfc.eval.vol_target import managed_returns, target_metrics, vol_target_weights


def _series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n)
    return pd.Series(rng.normal(0, 0.02, n), index=idx)


def test_weights_scale_inversely_with_the_forecast_and_are_capped():
    f = pd.Series([0.10, 0.25, 0.50, 1.00])
    w = vol_target_weights(f, target=0.25, cap=2.0)
    assert list(w) == [2.0, 1.0, 0.5, 0.25]  # 0.25 / 0.10 = 2.5 is capped at 2


def test_the_position_for_day_t_uses_only_the_forecast_made_before_it():
    r = _series()
    f = pd.Series(0.3, index=r.index)
    f.iloc[100] = 0.15  # a forecast made at the close of day 100
    m = managed_returns(r, f, target=0.3)
    assert m.iloc[100] == pytest.approx(
        r.iloc[100]
    )  # day 100 still used the day-99 forecast (weight 1)
    assert m.iloc[101] == pytest.approx(2.0 * r.iloc[101])  # the new forecast acts from day 101
    assert np.isnan(m.iloc[0])  # no forecast before the first day


def test_changing_a_later_return_never_changes_an_earlier_managed_return():
    r = _series()
    f = pd.Series(np.linspace(0.2, 0.6, len(r)), index=r.index)
    m1 = managed_returns(r, f, target=0.3)
    r2 = r.copy()
    r2.iloc[300:] = 0.5
    m2 = managed_returns(r2, f, target=0.3)
    pd.testing.assert_series_equal(m1.iloc[:300], m2.iloc[:300])


def test_a_perfect_forecast_keeps_realised_risk_near_the_target():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-05", periods=3000)
    vol = pd.Series(np.where(np.arange(3000) // 250 % 2 == 0, 0.20, 0.60), index=idx)  # annualised
    r = pd.Series(rng.normal(0, 1, 3000) * vol / np.sqrt(252), index=idx)
    perfect = target_metrics(
        managed_returns(r, vol, target=0.25), vol_target_weights(vol, 0.25), 0.25
    )
    flat = target_metrics(r, pd.Series(1.0, index=idx), 0.25)
    assert perfect["vol_of_realised_vol"] < 0.5 * flat["vol_of_realised_vol"]
    assert 0.20 < perfect["ann_vol"] < 0.30


def test_metrics_on_a_constant_series_are_known():
    idx = pd.bdate_range("2020-01-01", periods=300)
    m = pd.Series(0.01, index=idx)
    w = pd.Series(1.0, index=idx)
    out = target_metrics(m, w, target=0.1585)  # 0.01 * sqrt(252) = 0.1587 annualised
    assert out["ann_vol"] == pytest.approx(0.0, abs=1e-12)  # a constant return has no dispersion
    assert out["mean_weight"] == 1.0 and out["turnover"] == 0.0
    assert out["max_drawdown"] == 0.0  # only gains
    assert out["worst_22d"] == pytest.approx(22 * 0.01)
