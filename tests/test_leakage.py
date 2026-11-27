"""Leakage guard suite (see BUGS B-001: whole-series features produced a fake 26% gain).

Three guards:
1. Every feature builder is invariant to the future: replacing all rows after t with garbage leaves
   every feature at or before t unchanged. Add a new feature builder to FEATURES below.
2. Learned scalers are fitted on the training window only and applied unchanged afterwards.
3. No whole-series decomposition (VMD, EMD, CEEMDAN, wavelets) exists in the library code.
Alignment and release-date tests are in tests/test_alignment.py, split rules in tests/test_rolling.py,
the live-forecast no-look-ahead test in tests/test_live.py.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oilfc.eval.targets import past_realized_vol
from oilfc.features.exog import exogenous_features, fx_features
from oilfc.features.intraday_vol import intraday_vol_design
from oilfc.features.pooled import POOL_CHANNELS, pooled_channels
from oilfc.features.range_vol import range_vol_design
from oilfc.features.ttm_input import ttm_input
from oilfc.features.vol import CHANNELS, daily_channels, dl_channels, lag_window
from oilfc.models.baselines.vol import ewma_vol, har_design, leverage_design, ovx_features

N, T_CUT = 500, 350


def _data(seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=N)
    r = pd.Series(rng.standard_t(5, N) * 0.015, index=idx)
    panel = pd.DataFrame(
        {
            "ovx": 35 + rng.normal(0, 3, N),
            "vix": 18 + rng.normal(0, 2, N),
            "wti_fut1": 60 + rng.normal(0, 1, N),
            "wti_fut4": 61 + rng.normal(0, 1, N),
            "wti_fut1_age": 0.0,
            "wti_fut4_age": 0.0,
            "crude_stocks": 400_000 + rng.normal(0, 1000, N),
            "gpr_daily": 100 + rng.uniform(0, 50, N),
            "epu_daily": 100 + rng.uniform(0, 50, N),
            "usd_broad": 110 * np.exp(np.cumsum(rng.normal(0, 0.002, N))),
        },
        index=idx,
    )
    return panel, r


def _corrupt_future(panel, r, cut):
    """Copies in which every row after `cut` holds extreme values."""
    p, rr = panel.copy(), r.copy()
    future = p.index > cut
    p.loc[future, :] = 1e6
    rr.loc[future] = 1.0
    return p, rr


FEATURES = {
    "past_realized_vol_5": lambda p, r: past_realized_vol(r, 5).to_frame(),
    "past_realized_vol_22": lambda p, r: past_realized_vol(r, 22).to_frame(),
    "ewma_vol": lambda p, r: ewma_vol(r).to_frame(),
    "har_design": lambda p, r: har_design(r),
    "har_design_with_ovx": lambda p, r: har_design(r, p["ovx"]),
    "har_design_with_lagged_ovx": lambda p, r: har_design(r, p["ovx"].shift(1)),
    "leverage_design": lambda p, r: leverage_design(r),
    "daily_channels": lambda p, r: daily_channels(p, r),
    "lag_window": lambda p, r: lag_window(daily_channels(p, r), CHANNELS, 16),
    "exogenous_features": lambda p, r: exogenous_features(p, p.index),
    "fx_features": lambda p, r: fx_features(p, p.index),
    "ovx_features": lambda p, r: ovx_features(
        har_design(r, p["ovx"].shift(1))["log_ovx"], har_design(r)["log_rv_22"]
    ),
    "range_vol_design": lambda p, r: range_vol_design(
        pd.DataFrame(
            {
                "open": p["wti_fut1"],
                "high": p["wti_fut1"] * 1.01,
                "low": p["wti_fut1"] * 0.99,
                "close": p["wti_fut1"],
            }
        ),
        p.index,
    ),
    "intraday_vol_design": lambda p, r: intraday_vol_design(
        pd.DataFrame({"rv5": (p["wti_fut1"] / 60) ** 2 * 3e-4, "rv5_ss": p["wti_fut1"] * 5e-6}),
        p.index,
    ),
    "ttm_input_5": lambda p, r: ttm_input(r, 5).to_frame(),
    "ttm_input_66": lambda p, r: ttm_input(r, 66).to_frame(),
    "daily_channels_lag1": lambda p, r: daily_channels(p, r, ovx_lag=1),
    "dl_channels": lambda p, r: dl_channels(p, r),
    "pooled_channels_with_iv": lambda p, r: pooled_channels(r, p["ovx"], p["vix"], "a", ["a", "b"]),
    "pooled_channels_without_iv": lambda p, r: pooled_channels(r, None, p["vix"], "b", ["a", "b"]),
    "pooled_window": lambda p, r: lag_window(
        pooled_channels(r, p["ovx"], p["vix"], "a", ["a", "b"]),
        POOL_CHANNELS + ["is_a", "is_b"],
        16,
    ),
}


@pytest.mark.parametrize("name", list(FEATURES))
def test_a_feature_at_t_does_not_depend_on_anything_after_t(name):
    panel, r = _data()
    cut = panel.index[T_CUT]
    clean = FEATURES[name](panel, r).loc[:cut]
    bad_panel, bad_r = _corrupt_future(panel, r, cut)
    dirty = FEATURES[name](bad_panel, bad_r).loc[:cut]
    pd.testing.assert_frame_equal(clean, dirty, check_exact=False, rtol=1e-12, atol=1e-12)


def test_the_guard_itself_catches_a_feature_that_peeks_ahead():
    panel, r = _data()
    cut = panel.index[T_CUT]
    peeking = lambda p, rr: rr.shift(-3).to_frame()  # noqa: E731  (a look-ahead feature on purpose)
    clean = peeking(panel, r).loc[:cut]
    bad_panel, bad_r = _corrupt_future(panel, r, cut)
    dirty = peeking(bad_panel, bad_r).loc[:cut]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(clean, dirty, check_exact=False, rtol=1e-12, atol=1e-12)


def test_the_sequence_scaler_is_fitted_on_the_training_window_only():
    pytest.importorskip("torch")
    from oilfc.models.dl.residual_seq import SeqResidualRegressor

    rng = np.random.default_rng(0)
    seq, ch = 8, 2
    x_train = rng.normal(5.0, 2.0, (200, seq * ch)).astype(np.float32)
    y_train = rng.normal(0, 1.0, 200)
    model = SeqResidualRegressor(
        arch="mlp", seq_len=seq, n_channels=ch, mlp_hidden=8, epochs=2, seeds=(0,), device="cpu"
    ).fit(x_train, y_train)
    raw = x_train.reshape(200, seq, ch)
    assert np.allclose(model.mean_, raw.mean(axis=(0, 1), keepdims=True), atol=1e-5)
    assert model.y_scale_ == pytest.approx(float(y_train.std()), rel=1e-4)
    mean_before, std_before = model.mean_.copy(), model.std_.copy()

    x_test = np.vstack([rng.normal(500.0, 100.0, (30, seq * ch)), x_train[:1]]).astype(np.float32)
    batch = model.predict(x_test)
    assert np.allclose(model.mean_, mean_before) and np.allclose(model.std_, std_before)
    alone = model.predict(x_test[-1:])  # same row, but without the extreme test rows beside it
    assert alone[0] == pytest.approx(batch[-1], rel=1e-4)


def test_the_library_contains_no_whole_series_decomposition():
    src = Path(__file__).resolve().parents[1] / "src" / "oilfc"
    banned = re.compile(
        r"vmdpy|PyEMD|\bCEEMDAN\b|\bEEMD\b|\bEMD\b|\bVMD\b|pywt|wavedec", re.IGNORECASE
    )
    hits = []
    for f in src.rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if banned.search(code):
                hits.append(f"{f.relative_to(src.parent)}:{i}")
    assert not hits, f"whole-series decomposition found (leaks the future, see BUGS B-001): {hits}"
