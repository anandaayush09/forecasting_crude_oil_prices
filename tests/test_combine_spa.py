"""Combination rules, the SPA wrapper and the lagged channels of the deep-learning addendum (ADR 0004)."""

import numpy as np
import pandas as pd
import pytest

from oilfc.eval.losses import qlike
from oilfc.eval.spa import spa_pvalue
from oilfc.features.vol import daily_channels
from oilfc.models.ml.combine import equal_weight, gated_combination


def _blocks(n_blocks=8, size=60, start=0):
    return [np.arange(start + i * size, start + (i + 1) * size) for i in range(n_blocks)]


def _synthetic(n=480, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    realised = np.exp(rng.normal(-1.5, 0.2, n))
    bench = realised * np.exp(rng.normal(0, 0.25, n))
    # the learned forecast is close to the truth where z > 0 and noisy elsewhere
    model = realised * np.exp(np.where(z > 0, rng.normal(0, 0.03, n), rng.normal(0.5, 0.3, n)))
    return realised, bench, model, z


def test_equal_weight_is_the_log_average():
    assert equal_weight(np.log([0.2]), np.log([0.4]))[0] == pytest.approx(np.log(np.sqrt(0.08)))


def test_the_gate_uses_weight_one_half_until_enough_realised_origins_exist():
    realised, bench, model, z = _synthetic()
    out = gated_combination(realised, bench, model, z, _blocks(), h=5, min_obs=250)
    assert (out["weight"].iloc[:300] == 0.5).all()  # fewer than 250 realised earlier origins
    assert out["weight"].iloc[400:].nunique() > 1  # later blocks use a fitted gate


def test_the_fitted_gate_leans_on_the_model_where_the_model_is_good():
    realised, bench, model, z = _synthetic()
    out = gated_combination(realised, bench, model, z, _blocks(), h=5, min_obs=250)
    late = out.iloc[360:]
    assert late["weight"][z[360:] > 0].mean() > late["weight"][z[360:] <= 0].mean() + 0.2
    better = qlike(realised[360:], late["pred"]).mean() < qlike(realised[360:], bench[360:]).mean()
    assert better


def test_a_gate_forecast_never_depends_on_later_data():
    realised, bench, model, z = _synthetic()
    blocks = _blocks()
    full = gated_combination(realised, bench, model, z, blocks, h=5)
    r2, b2, m2, z2 = (v.copy() for v in (realised, bench, model, z))
    cut = blocks[5][0]
    r2[cut:], b2[cut:], m2[cut:], z2[cut:] = 9.0, 9.0, 9.0, 9.0  # corrupt blocks 5 onward
    corrupted = gated_combination(r2, b2, m2, z2, blocks, h=5)
    pd.testing.assert_frame_equal(full.iloc[:cut], corrupted.iloc[:cut])


def test_spa_detects_a_better_model_and_not_a_copy_of_the_benchmark():
    rng = np.random.default_rng(1)
    bench = np.abs(rng.normal(1.0, 0.3, 600))
    better = pd.DataFrame({"a": bench * 0.7, "b": bench * 1.1})
    same = pd.DataFrame(
        {"a": bench + rng.normal(0, 0.05, 600), "b": bench + rng.normal(0, 0.05, 600)}
    )
    assert spa_pvalue(bench, better, block_size=5, reps=300) < 0.05
    assert spa_pvalue(bench, same, block_size=5, reps=300) > 0.05


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    panel = pd.DataFrame(
        {"ovx": 35 + rng.normal(0, 3, n), "vix": 18 + rng.normal(0, 2, n)}, index=idx
    )
    return panel, pd.Series(rng.standard_t(5, n) * 0.015, index=idx)


def test_lagged_channels_use_yesterdays_ovx_and_vix():
    panel, r = _data()
    now, lagged = daily_channels(panel, r), daily_channels(panel, r, ovx_lag=1)
    for col in ("log_ovx", "log_vix"):
        pd.testing.assert_series_equal(
            lagged[col].iloc[1:], now[col].shift(1).iloc[1:], check_names=False
        )
    pd.testing.assert_series_equal(lagged["ret_1"], now["ret_1"])  # returns are not delayed


def test_spa_ignores_models_identical_to_the_benchmark():
    rng = np.random.default_rng(2)
    bench = np.abs(rng.normal(1.0, 0.3, 400))
    clones = pd.DataFrame({"a": bench.copy(), "b": bench.copy()})
    assert spa_pvalue(bench, clones, block_size=5, reps=200) == 1.0
    mixed = pd.DataFrame({"a": bench.copy(), "b": bench * 0.7})
    assert spa_pvalue(bench, mixed, block_size=5, reps=200) < 0.05
