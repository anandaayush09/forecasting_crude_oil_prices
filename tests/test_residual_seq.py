import numpy as np
import pandas as pd
import pytest

from oilfc.eval.rolling import expanding_splits
from oilfc.features.vol import lag_window
from oilfc.models.dl.residual_seq import SeqResidualRegressor
from oilfc.models.ml.residual import residual_forecast

L, C = 16, 3


def _windows(n=1500, seed=0, signal=True):
    """Daily channels where the target depends on a pattern inside the window (channel 0 trend)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n + L)
    frame = pd.DataFrame(rng.normal(size=(n + L, C)), index=idx, columns=["a", "b", "c"])
    w = lag_window(frame, ["a", "b", "c"], L).iloc[L:]
    arr = w.to_numpy().reshape(len(w), L, C)
    y = 0.8 * (arr[:, :4, 0].mean(axis=1) - arr[:, -4:, 0].mean(axis=1)) if signal else 0.0
    y = y + rng.normal(0, 0.2, len(w))
    return w, pd.Series(y, index=w.index)


def _model(arch, **kw):
    return SeqResidualRegressor(
        arch=arch, seq_len=L, n_channels=C, patch_len=4, d_model=16, n_heads=2, n_layers=1,
        epochs=80, seeds=(0,), **kw,
    )  # fmt: skip


@pytest.mark.parametrize("arch", ["transformer", "mlp"])
def test_learns_a_pattern_inside_the_window(arch):
    X, y = _windows()
    tr, te = slice(0, 1100), slice(1100, 1500)
    pred = (
        _model(arch)
        .fit(X.iloc[tr].to_numpy(), y.iloc[tr].to_numpy())
        .predict(X.iloc[te].to_numpy())
    )
    r2 = 1 - np.mean((y.iloc[te] - pred) ** 2) / np.var(y.iloc[te])
    assert r2 > 0.5


def test_same_seed_gives_identical_predictions_and_shapes_are_right():
    X, y = _windows(n=600)
    a = _model("transformer").fit(X.to_numpy(), y.to_numpy()).predict(X.to_numpy()[:50])
    b = _model("transformer").fit(X.to_numpy(), y.to_numpy()).predict(X.to_numpy()[:50])
    assert a.shape == (50,)
    assert np.allclose(a, b)


def test_parameter_counts_and_unknown_arch():
    assert _model("transformer").n_parameters() > 0
    assert _model("mlp").n_parameters() > 0
    with pytest.raises(ValueError):
        SeqResidualRegressor(arch="nope", seq_len=L, n_channels=C).n_parameters()


def test_plugs_into_residual_forecast_and_shrinks_noise_toward_the_anchor():
    X, y = _windows(n=1500, signal=False)
    frame = X.copy()
    frame["x"] = np.random.default_rng(1).normal(size=len(frame))
    y = y + 0.5 * frame["x"]
    splits = expanding_splits(frame.index, frame.index[1000], 1, 4)[:2]
    out = residual_forecast(
        frame, y, splits, ["x"], list(X.columns), lambda: _model("transformer")
    ).dropna()
    assert out["alpha"].mean() < 0.5  # no real signal in the window: the correction is shrunk
    assert np.mean((y.loc[out.index] - out["pred"]) ** 2) < 1.05 * np.mean(
        (y.loc[out.index] - out["anchor"]) ** 2
    )
