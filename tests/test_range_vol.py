"""Range-based volatility inputs."""

import numpy as np
import pandas as pd
import pytest

from oilfc.features.range_vol import daily_variance, load_ohlc, range_vol_design


def _ohlc(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_the_parkinson_variance_matches_the_formula_and_a_flat_day_has_zero():
    o = pd.DataFrame(
        {"open": [10.0, 10.0], "high": [11.0, 10.0], "low": [9.0, 10.0], "close": [10.5, 10.0]}
    )
    v = daily_variance(o, "parkinson")
    assert v.iloc[0] == pytest.approx(np.log(11 / 9) ** 2 / (4 * np.log(2)))
    assert v.iloc[1] == 0.0


def test_garman_klass_adds_the_open_to_close_term():
    o = pd.DataFrame({"open": [10.0], "high": [11.0], "low": [9.0], "close": [10.8]})
    hl, co = np.log(11 / 9), np.log(10.8 / 10)
    assert daily_variance(o, "garman_klass").iloc[0] == pytest.approx(
        0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    )


def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown kind"):
        daily_variance(_ohlc(), "yang_zhang")


def test_the_design_at_t_uses_ranges_up_to_the_previous_day_only():
    o = _ohlc()
    base = range_vol_design(o, o.index)
    changed = o.copy()
    changed.iloc[100:, :] *= 3.0  # rows from position 100 on are corrupted
    new = range_vol_design(changed, o.index)
    pd.testing.assert_frame_equal(base.iloc[:101], new.iloc[:101])  # position 100 sees through 99
    assert not base.iloc[101:].equals(new.iloc[101:])


def test_a_window_needs_enough_days_and_missing_days_are_averaged_over():
    o = _ohlc()
    sparse = o.drop(o.index[60:62])  # two missing days: a 22-day window is still 90% present
    d = range_vol_design(sparse, o.index)
    assert d["log_pk_22"].iloc[70:].notna().all()
    gone = o.drop(o.index[50:70])  # twenty missing days: windows around them fall under 80%
    assert range_vol_design(gone, o.index)["log_pk_22"].iloc[70:80].isna().any()


def test_load_ohlc_drops_bad_rows(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text(
        "date,open,high,low,close"
        + chr(10)
        + "2020-01-01,10,11,9,10"
        + chr(10)
        + "2020-01-02,10,9,11,10"
        + chr(10)
        + "2020-01-03,-5,1,-6,0"
        + chr(10)
        + "2020-01-04,,,,"
    )
    assert len(load_ohlc(f)) == 1
