"""Shrunk residual forecasting on top of a linear anchor (the AR-MoE idea, without the transformer).

The anchor is an OLS forecast of log volatility (for example HAR + OVX). A learned model predicts
what the anchor misses, and a weight alpha in [0, 1], chosen on a time-ordered validation block,
shrinks that correction toward zero. With alpha = 0 the forecast is exactly the anchor, so the
procedure cannot do much worse than its benchmark, and any gain is a clean increment.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

ALPHAS = np.linspace(0.0, 1.0, 11)
MIN_TRAIN_ROWS = (
    250  # a split with fewer complete training rows is skipped (its forecasts stay NaN)
)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def _predict_ols(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def residual_forecast(
    X: pd.DataFrame,
    y_log: pd.Series,
    splits,
    anchor_cols: list[str],
    feature_cols: list[str],
    model_factory: Callable,
    val_fraction: float = 0.2,
    variance_correction: bool = False,
) -> pd.DataFrame:
    """Anchor + alpha * learned residual, for every test origin.

    For each (train, test) split:
      1. Keep the complete training rows; split them in time: fit part, then validation block.
      2. Fit the anchor (OLS on anchor_cols) and the residual model (model_factory(), features
         feature_cols, target = anchor residual) on the fit part only.
      3. Choose alpha on the validation block by squared error of anchor + alpha * residual.
      4. Refit the anchor and the residual model on all training rows; predict the test rows.
    variance_correction: add s2 to both the anchor and the final forecast, where s2 is the variance of
    the validation-block error of anchor + alpha * residual (training data only). This makes the
    log-volatility forecast suitable for QLIKE (see vol.forecast_log_linear).
    Splits with fewer than MIN_TRAIN_ROWS complete training rows are skipped.
    Returns a DataFrame indexed like X with columns anchor, residual, alpha, pred (log volatility),
    NaN outside the test blocks.
    """
    out = pd.DataFrame(np.nan, index=X.index, columns=["anchor", "residual", "alpha", "pred"])
    need = sorted(set(anchor_cols) | set(feature_cols))
    for train, test in splits:
        data = X.iloc[train][need].join(y_log.iloc[train].rename("y")).dropna()
        if len(data) < MIN_TRAIN_ROWS:
            continue
        n_val = max(int(len(data) * val_fraction), 1)
        fit, val = data.iloc[:-n_val], data.iloc[-n_val:]

        beta = _ols(fit[anchor_cols].to_numpy(), fit["y"].to_numpy())
        res_fit = fit["y"].to_numpy() - _predict_ols(beta, fit[anchor_cols].to_numpy())
        model = model_factory().fit(fit[feature_cols].to_numpy(), res_fit)
        anchor_val = _predict_ols(beta, val[anchor_cols].to_numpy())
        corr_val = model.predict(val[feature_cols].to_numpy())
        errors = [np.mean((val["y"].to_numpy() - anchor_val - a * corr_val) ** 2) for a in ALPHAS]
        alpha = float(ALPHAS[int(np.argmin(errors))])
        s2 = 0.0
        if variance_correction:
            s2 = float(np.var(val["y"].to_numpy() - anchor_val - alpha * corr_val))

        beta = _ols(data[anchor_cols].to_numpy(), data["y"].to_numpy())
        res_all = data["y"].to_numpy() - _predict_ols(beta, data[anchor_cols].to_numpy())
        model = model_factory().fit(data[feature_cols].to_numpy(), res_all)

        te = X.iloc[test][need]
        ok = te.notna().all(axis=1).to_numpy()
        anchor = np.full(len(te), np.nan)
        corr = np.full(len(te), np.nan)
        anchor[ok] = _predict_ols(beta, te.loc[ok, anchor_cols].to_numpy()) + s2
        corr[ok] = model.predict(te.loc[ok, feature_cols].to_numpy())
        rows = out.index[test]
        out.loc[rows, "anchor"] = anchor
        out.loc[rows, "residual"] = corr
        out.loc[rows, "alpha"] = alpha
        out.loc[rows, "pred"] = anchor + alpha * corr
    return out
