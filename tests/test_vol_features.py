import numpy as np
import pandas as pd

from oilfc.features.vol import CHANNELS, daily_channels, lag_window


def _panel(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    panel = pd.DataFrame(
        {"ovx": 30 + rng.normal(0, 2, n), "vix": 15 + rng.normal(0, 1, n)}, index=idx
    )
    panel.loc[idx[10], "vix"] = np.nan  # a holiday gap
    return panel, pd.Series(rng.normal(0, 0.02, n), index=idx)


def test_daily_channels_contains_every_channel_and_the_har_columns():
    panel, r = _panel()
    d = daily_channels(panel, r)
    assert set(CHANNELS) <= set(d.columns)
    assert {"log_rv_5", "log_rv_22", "log_rv_63", "log_ovx"} <= set(d.columns)
    assert np.isclose(d["log_vix"].iloc[10], d["log_vix"].iloc[9])  # forward-filled


def test_daily_channels_are_causal():
    panel, r = _panel()
    base = daily_channels(panel, r)
    t = 150
    future = r.copy()
    future.iloc[t + 1 :] = 0.3
    panel2 = panel.copy()
    panel2.iloc[t + 1 :] = 999.0
    again = daily_channels(panel2, future)
    assert np.allclose(again.iloc[t], base.iloc[t], equal_nan=True)


def test_lag_window_layout_is_lag_major_with_today_first():
    frame = pd.DataFrame(
        {"a": np.arange(10.0), "b": np.arange(10.0) * 10},
        index=pd.bdate_range("2020-01-01", periods=10),
    )
    w = lag_window(frame, ["a", "b"], 3)
    assert list(w.columns) == ["a__0", "b__0", "a__1", "b__1", "a__2", "b__2"]
    assert w.iloc[5].tolist() == [5.0, 50.0, 4.0, 40.0, 3.0, 30.0]
    assert w.iloc[:2].isna().any(axis=1).all()  # not enough history yet
    assert w.iloc[2].notna().all()
    assert w.iloc[5].to_numpy().reshape(3, 2)[:, 0].tolist() == [5.0, 4.0, 3.0]  # (lag, channel)


def test_lag_window_uses_only_past_and_present_rows():
    frame = pd.DataFrame({"a": np.arange(20.0)}, index=pd.bdate_range("2020-01-01", periods=20))
    base = lag_window(frame, ["a"], 4)
    changed = frame.copy()
    changed.iloc[11:] = -1.0
    assert np.allclose(lag_window(changed, ["a"], 4).iloc[10], base.iloc[10])
