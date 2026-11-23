import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from oilfc.eval.rolling import expanding_splits
from oilfc.models.ml.residual import residual_forecast


def _lgbm():
    return LGBMRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, verbose=-1)


def _data(signal=True, n=1800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    x, z = rng.normal(size=n), rng.normal(size=n)
    extra = 0.8 * np.sin(2 * z) if signal else 0.0
    y = 1.0 + 0.5 * x + extra + rng.normal(0, 0.1, n)
    return pd.DataFrame({"x": x, "z": z}, index=idx), pd.Series(y, index=idx)


def _splits(X):
    return expanding_splits(X.index, X.index[1200], 1, 3)[:4]


def _mse(a, b):
    return float(np.mean((a - b) ** 2))


def test_a_learnable_residual_is_picked_up_and_beats_the_anchor():
    X, y = _data(signal=True)
    splits = _splits(X)
    out = residual_forecast(X, y, splits, ["x"], ["x", "z"], _lgbm).dropna()
    assert (out["alpha"] >= 0.7).all()
    assert _mse(y.loc[out.index], out["pred"]) < 0.4 * _mse(y.loc[out.index], out["anchor"])


def test_pure_noise_features_are_shrunk_toward_the_anchor():
    X, y = _data(signal=False)
    out = residual_forecast(X, y, _splits(X), ["x"], ["x", "z"], _lgbm).dropna()
    assert out["alpha"].mean() < 0.35
    assert _mse(y.loc[out.index], out["pred"]) < 1.03 * _mse(y.loc[out.index], out["anchor"])


def test_alpha_zero_means_the_forecast_is_exactly_the_anchor():
    X, y = _data(signal=False)
    out = residual_forecast(X, y, _splits(X), ["x"], ["x", "z"], _lgbm).dropna()
    zero = out["alpha"] == 0
    assert np.allclose(out.loc[zero, "pred"], out.loc[zero, "anchor"])


def test_predictions_exist_only_in_test_blocks_and_never_use_test_targets():
    X, y = _data(signal=True)
    splits = _splits(X)
    base = residual_forecast(X, y, splits, ["x"], ["x", "z"], lambda: Ridge(alpha=1.0))
    test_pos = np.concatenate([t for _, t in splits])
    assert base.iloc[test_pos].notna().all().all()
    assert base.drop(base.index[test_pos]).isna().all().all()
    y2 = y.copy()
    y2.iloc[test_pos] = 99.0  # targets inside the test blocks
    again = residual_forecast(X, y2, splits[:1], ["x"], ["x", "z"], lambda: Ridge(alpha=1.0))
    first = splits[0][1]
    assert np.allclose(again["pred"].iloc[first], base["pred"].iloc[first])


def test_missing_training_rows_are_dropped_not_filled():
    X, y = _data(signal=True)
    X.iloc[[10, 11, 500], 0] = np.nan
    out = residual_forecast(X, y, _splits(X), ["x"], ["x", "z"], lambda: Ridge(alpha=1.0))
    assert out["pred"].dropna().shape[0] > 0


def test_variance_correction_shifts_anchor_and_forecast_by_the_same_training_constant():
    X, y = _data(signal=True)
    splits = _splits(X)
    plain = residual_forecast(X, y, splits, ["x"], ["x", "z"], _lgbm)
    corr = residual_forecast(X, y, splits, ["x"], ["x", "z"], _lgbm, variance_correction=True)
    shift = (corr["pred"] - plain["pred"]).dropna()
    assert (shift > 0).all()
    assert np.allclose(
        corr["anchor"] - plain["anchor"], corr["pred"] - plain["pred"], equal_nan=True
    )
    first = splits[0][1]
    assert np.allclose(
        shift.loc[X.index[first]], shift.loc[X.index[first[0]]]
    )  # one constant per block


def test_a_split_with_too_little_training_data_is_skipped_not_crashed():
    X, y = _data(signal=True)
    splits = expanding_splits(X.index, X.index[100], 1, 3)[:1]  # only 99 rows to train on
    out = residual_forecast(X, y, splits, ["x"], ["x", "z"], lambda: Ridge(alpha=1.0))
    assert out["pred"].isna().all()
