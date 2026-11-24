"""V6: volatility benchmark ladder. See docs/tasks/V6_vol_baselines.md.


Interface
  ewma_vol(returns, lam=0.94) -> Series      annualised EWMA volatility, uses returns up to and including t
  har_design(returns, ovx=None) -> DataFrame columns log_rv_5, log_rv_22, log_rv_63 (past realised vol, annualised)
                                              and, if ovx is given, log_ovx = log(ovx / 100); all use information up to t
  forecast_log_linear(X, y_log, splits, cols) -> Series
                                              for each (train, test) split fit OLS (with intercept) of y_log on X[cols]
                                              using the train rows that have no missing values, predict the test rows;
                                              returns the predictions as a Series indexed like X, NaN outside the test blocks
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANN = np.sqrt(252)


def _returns(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.02, n), index=pd.bdate_range("2015-01-01", periods=n))


def test_ewma_of_constant_returns_converges_to_the_constant_vol():
    from oilfc.models.baselines.vol import ewma_vol

    r = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=300))
    assert ewma_vol(r).iloc[-1] == pytest.approx(0.01 * ANN, rel=1e-3)


def test_ewma_follows_the_recursion_and_is_causal():
    from oilfc.models.baselines.vol import ewma_vol

    r = _returns()
    out = ewma_vol(r, lam=0.94)
    t = 200
    var_prev = (out.iloc[t - 1] / ANN) ** 2
    expected = np.sqrt(252 * (0.94 * var_prev + 0.06 * r.iloc[t] ** 2))
    assert out.iloc[t] == pytest.approx(expected)
    future = r.copy()
    future.iloc[t + 1 :] = 9.0
    assert ewma_vol(future, lam=0.94).iloc[t] == pytest.approx(out.iloc[t])


def test_har_design_columns_and_values():
    from oilfc.eval.targets import past_realized_vol
    from oilfc.models.baselines.vol import har_design

    r = _returns()
    ovx = pd.Series(40.0, index=r.index)
    d = har_design(r, ovx)
    assert list(d.columns) == ["log_rv_5", "log_rv_22", "log_rv_63", "log_ovx"]
    assert d["log_rv_22"].iloc[100] == pytest.approx(np.log(past_realized_vol(r, 22).iloc[100]))
    assert d["log_ovx"].iloc[100] == pytest.approx(np.log(0.40))
    assert list(har_design(r).columns) == ["log_rv_5", "log_rv_22", "log_rv_63"]


def test_har_design_is_causal():
    from oilfc.models.baselines.vol import har_design

    r = _returns()
    base = har_design(r)
    t = 150
    future = r.copy()
    future.iloc[t + 1 :] = 9.0
    assert np.allclose(har_design(future).iloc[t], base.iloc[t])


def _toy_frame():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=300)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)}, index=idx)
    y = 1.0 + 2.0 * X["a"] - 0.5 * X["b"]
    splits = [(np.arange(0, 150), np.arange(160, 200)), (np.arange(0, 200), np.arange(210, 250))]
    return X, y, splits


def test_forecast_log_linear_recovers_an_exact_linear_relation():
    from oilfc.models.baselines.vol import forecast_log_linear

    X, y, splits = _toy_frame()
    out = forecast_log_linear(X, y, splits, ["a", "b"])
    assert out.index.equals(X.index)
    test_pos = np.concatenate([t for _, t in splits])
    assert np.allclose(out.iloc[test_pos], y.iloc[test_pos])
    other = np.setdiff1d(np.arange(300), test_pos)
    assert out.iloc[other].isna().all()


def test_forecast_log_linear_never_looks_at_test_targets_and_skips_missing_rows():
    from oilfc.models.baselines.vol import forecast_log_linear

    X, y, splits = _toy_frame()
    base = forecast_log_linear(X, y, splits, ["a", "b"])
    y2 = y.copy()
    y2.iloc[160:200] = 99.0  # targets inside the first test block
    y2.iloc[210:250] = -99.0
    assert np.allclose(
        forecast_log_linear(X, y2, splits, ["a", "b"]).iloc[160:200], base.iloc[160:200]
    )
    X2 = X.copy()
    X2.iloc[[5, 6, 7], 0] = np.nan  # missing training rows are dropped, not filled
    assert forecast_log_linear(X2, y, splits, ["a", "b"]).iloc[160:200].notna().all()


PANEL = Path(__file__).resolve().parents[1] / "data" / "processed" / "panel_daily.parquet"


@pytest.mark.skipif(
    not PANEL.exists(), reason="run the data build first (data/processed/panel_daily.parquet)"
)
def test_reproduces_the_preliminary_brent_result():
    """The read-only check of 2026-11-23 (Brent, 22-day vol, test from 2012, monthly refit):
    OVX-only vs naive about 33% lower MSE(log vol), vs HAR about 24%, Diebold-Mariano p < 0.01."""
    from oilfc.eval.holdout import development_only
    from oilfc.eval.losses import mse_logvol
    from oilfc.eval.rolling import expanding_splits
    from oilfc.eval.stat_tests import dm_hln
    from oilfc.eval.targets import forward_realized_vol
    from oilfc.models.baselines.vol import forecast_log_linear, har_design

    panel = pd.read_parquet(PANEL)
    panel.index = pd.to_datetime(panel.index)
    r = development_only(
        np.log(panel["brent_spot"]).diff().dropna(), "2025-07-01"
    )  # never the holdout
    X = har_design(r, panel["ovx"].reindex(r.index))
    y = np.log(forward_realized_vol(r, 22))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), 22, 1)
    pred = {
        "naive": X["log_rv_22"],
        "har": forecast_log_linear(X, y, splits, ["log_rv_5", "log_rv_22", "log_rv_63"]),
        "ovx": forecast_log_linear(X, y, splits, ["log_ovx"]),
    }
    ok = y.notna() & pred["har"].notna() & pred["ovx"].notna() & pred["naive"].notna()
    loss = {k: np.asarray(mse_logvol(np.exp(y[ok]), np.exp(v[ok]))) for k, v in pred.items()}
    gain_naive = 100 * (1 - loss["ovx"].mean() / loss["naive"].mean())
    gain_har = 100 * (1 - loss["ovx"].mean() / loss["har"].mean())
    assert gain_naive == pytest.approx(33.4, abs=3.0)
    assert gain_har == pytest.approx(24.3, abs=3.0)
    assert dm_hln(loss["naive"], loss["ovx"], 22)[1] < 0.01


def test_variance_correction_adds_the_training_residual_variance_only():
    from oilfc.models.baselines.vol import forecast_log_linear

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-01", periods=600)
    X = pd.DataFrame({"a": rng.normal(size=600)}, index=idx)
    y = 1.0 + 2.0 * X["a"] + pd.Series(rng.normal(0, 0.5, 600), index=idx)
    splits = [(np.arange(0, 400), np.arange(410, 500))]
    plain = forecast_log_linear(X, y, splits, ["a"])
    corrected = forecast_log_linear(X, y, splits, ["a"], variance_correction=True)
    diff = (corrected - plain).dropna()
    assert np.allclose(diff, diff.iloc[0])  # one constant for the whole block
    assert diff.iloc[0] == pytest.approx(0.25, rel=0.25)  # about sigma**2 = 0.5**2
    y_future = y.copy()
    y_future.iloc[410:500] = 50.0  # test targets must not change the correction
    again = forecast_log_linear(X, y_future, splits, ["a"], variance_correction=True)
    assert np.allclose(again.dropna(), corrected.dropna())


def test_leverage_design_is_the_rolling_mean_of_the_negative_part_and_causal():
    from oilfc.models.baselines.vol import leverage_design

    r = pd.Series(
        [0.01, -0.02, 0.03, -0.04, 0.0, -0.01], index=pd.bdate_range("2020-01-01", periods=6)
    )
    d = leverage_design(r, windows=(1, 3))
    assert list(d.columns) == ["neg_ret_1", "neg_ret_3"]
    assert d["neg_ret_1"].tolist() == pytest.approx([0.0, -0.02, 0.0, -0.04, 0.0, -0.01])
    assert d["neg_ret_3"].iloc[3] == pytest.approx((-0.02 + 0.0 - 0.04) / 3)
    assert (d.dropna() <= 0).all().all()
    assert (
        leverage_design(r.abs()).dropna().eq(0).all().all()
    )  # no negative returns, no leverage term
    future = r.copy()
    future.iloc[4:] = -0.5
    assert np.allclose(leverage_design(future, (1, 3)).iloc[3], d.iloc[3])
