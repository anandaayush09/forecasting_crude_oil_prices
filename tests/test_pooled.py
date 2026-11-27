"""Stage B: the pooled residual forecaster, its inputs and the companion dates (ADR 0004 amendment)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from oilfc.data.companions import at_availability
from oilfc.eval.rolling import expanding_splits
from oilfc.features.pooled import pooled_channels, pooled_table
from oilfc.models.ml.pooled import MIN_COMPANION_ROWS, pooled_residual_forecast
from oilfc.models.ml.residual import residual_forecast

H = 5
ANCHOR = ["a1", "a2"]
FEATURES = ["f1", "f2", "f3"]


def _table(seed, n=1100, start="2015-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    X = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=ANCHOR + FEATURES)
    X["y"] = 0.5 * X["a1"] - 0.3 * X["a2"] + 0.4 * X["f1"] + rng.normal(0, 0.5, n)
    X["end"] = pd.Series(idx, index=idx).shift(-H)
    X.loc[X["end"].isna(), "y"] = np.nan  # the last H origins have no realised target
    return X


def _splits(index):
    return expanding_splits(index, pd.Timestamp("2018-06-01"), H, 1)


def _ridge():
    return Ridge(alpha=10.0)


def test_with_one_series_it_equals_the_single_series_residual_forecast():
    t = _table(0)
    splits = _splits(t.index)
    single = residual_forecast(
        t[ANCHOR + FEATURES], t["y"], splits, ANCHOR, FEATURES, _ridge, residual_refit_every=3
    )
    pooled = pooled_residual_forecast(
        {"x": t}, {"x": ANCHOR}, FEATURES, "x", splits, _ridge, residual_refit_every=3
    )
    assert pooled["residual"].notna().sum() > 100
    pd.testing.assert_series_equal(pooled["residual"], single["residual"], check_names=False)
    pd.testing.assert_series_equal(pooled["alpha"], single["alpha"], check_names=False)


def test_a_companion_row_whose_target_is_not_over_before_the_block_is_never_used():
    target, comp = _table(1), _table(2)
    first = _splits(target.index)[:1]
    first_test = target.index[first[0][1][0]]
    tables = {"t": target, "c": comp}
    anchors = {"t": ANCHOR, "c": ANCHOR}
    clean = pooled_residual_forecast(tables, anchors, FEATURES, "t", first, _ridge)

    ruined = comp.copy()
    late = ruined["end"] >= first_test  # these rows reach into the test block
    ruined.loc[late, ANCHOR + FEATURES + ["y"]] = 1e6
    dirty = pooled_residual_forecast(
        {"t": target, "c": ruined}, anchors, FEATURES, "t", first, _ridge
    )
    pd.testing.assert_frame_equal(clean, dirty)

    early = comp.copy()
    early.loc[~late, "y"] *= -1.0  # the same kind of edit on usable rows must change the result
    moved = pooled_residual_forecast(
        {"t": target, "c": early}, anchors, FEATURES, "t", first, _ridge
    )
    assert not np.allclose(clean["residual"].dropna(), moved["residual"].dropna())


def test_a_companion_with_too_few_rows_does_not_change_the_forecast():
    target = _table(3)
    tiny = _table(4, n=MIN_COMPANION_ROWS // 2 + 10)
    splits = _splits(target.index)
    anchors = {"t": ANCHOR, "c": ANCHOR}
    alone = pooled_residual_forecast({"t": target}, {"t": ANCHOR}, FEATURES, "t", splits, _ridge)
    with_tiny = pooled_residual_forecast(
        {"t": target, "c": tiny}, anchors, FEATURES, "t", splits, _ridge
    )
    pd.testing.assert_frame_equal(alone, with_tiny)


def test_rows_with_an_undefined_log_volatility_are_left_out_not_fitted():
    target, comp = _table(6), _table(7)
    comp.iloc[100:130, comp.columns.get_loc("f1")] = -np.inf  # a flat-price window
    comp.iloc[200:210, comp.columns.get_loc("y")] = -np.inf
    splits = _splits(target.index)
    out = pooled_residual_forecast(
        {"t": target, "c": comp}, {"t": ANCHOR, "c": ANCHOR}, FEATURES, "t", splits, _ridge
    )
    assert np.isfinite(out["residual"].dropna()).all() and out["residual"].notna().sum() > 100


def test_refit_every_must_be_positive():
    t = _table(5)
    with pytest.raises(ValueError):
        pooled_residual_forecast(
            {"t": t}, {"t": ANCHOR}, FEATURES, "t", _splits(t.index), _ridge, residual_refit_every=0
        )


def test_a_companion_value_is_dated_at_the_day_it_becomes_public():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"]))
    out = at_availability(s, {"kind": "bdays", "n": 5})
    assert list(out.index) == list(pd.to_datetime(["2020-01-10", "2020-01-13", "2020-01-14"]))
    assert list(out) == [1.0, 2.0, 3.0]
    same = at_availability(s, {"kind": "bdays", "n": 0})
    assert list(same.index) == list(s.index)


def test_a_series_without_implied_volatility_is_flagged_and_zero_filled():
    idx = pd.bdate_range("2020-01-01", periods=120)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 120), index=idx)
    vix = pd.Series(20.0, index=idx)
    names = ["brent", "ulsd"]
    out = pooled_channels(r, None, vix, "ulsd", names)
    assert (out["iv_missing"] == 1.0).all() and (out["log_iv"] == 0.0).all()
    assert (out["is_ulsd"] == 1.0).all() and (out["is_brent"] == 0.0).all()
    assert out["log_ovx"].isna().all()  # the anchor must never see the zero fill

    with_iv = pooled_channels(r, pd.Series(30.0, index=idx), vix, "brent", names)
    assert with_iv["iv_missing"].iloc[0] == 1.0 and (with_iv["iv_missing"].iloc[1:] == 0.0).all()
    assert with_iv["log_iv"].iloc[1] == pytest.approx(np.log(0.30))  # delayed one day


def test_the_target_ends_h_returns_ahead():
    idx = pd.bdate_range("2020-01-01", periods=60)
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 60), index=idx)
    table = pooled_table(pd.DataFrame(index=idx), r, 5)
    assert table["end"].iloc[0] == idx[5]
    assert table["end"].iloc[-5:].isna().all() and table["y"].iloc[-5:].isna().all()
