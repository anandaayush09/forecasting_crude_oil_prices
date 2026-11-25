"""Volatility benchmark ladder: EWMA, HAR design matrix, and an expanding log-linear forecaster."""

import numpy as np
import pandas as pd

from oilfc.eval.targets import TRADING_DAYS, past_realized_vol


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Annualised RiskMetrics-style EWMA volatility; the value at t uses returns up to and including t.

    Recursion: var_t = lam * var_{t-1} + (1 - lam) * r_t**2, started from the mean of the first 20
    squared returns.
    """
    r2 = (returns**2).to_numpy()
    var = np.empty(len(r2))
    var[0] = r2[:20].mean()
    for i in range(1, len(r2)):
        var[i] = lam * var[i - 1] + (1 - lam) * r2[i]
    return pd.Series(np.sqrt(TRADING_DAYS * var), index=returns.index)


def har_design(returns: pd.Series, ovx: pd.Series | None = None) -> pd.DataFrame:
    """Log past realised volatility over 5, 22 and 63 days (the HAR inputs), plus log(OVX/100) if given.

    Every column at t uses information up to and including t.
    """
    cols = {f"log_rv_{h}": np.log(past_realized_vol(returns, h)) for h in (5, 22, 63)}
    design = pd.DataFrame(cols)
    if ovx is not None:
        design["log_ovx"] = np.log(ovx / 100)
    return design


def ovx_features(log_ovx: pd.Series, log_rv_22: pd.Series) -> pd.DataFrame:
    """OVX dynamics: 5-day and 22-day means of log OVX, and the positive and negative parts of the gap
    log OVX minus log 22-day realised volatility (an implied-versus-realised, variance-risk-premium proxy).

    log_ovx must already be lagged to the forecast origin (see har_design). Every value at t uses
    information up to and including t. The gap is split by sign because the gap itself is a linear
    combination of two HAR inputs and would add nothing to a linear model.
    """
    gap = log_ovx - log_rv_22
    return pd.DataFrame({
        "log_ovx_5": log_ovx.rolling(5).mean(),
        "log_ovx_22": log_ovx.rolling(22).mean(),
        "gap_pos": gap.clip(lower=0.0),
        "gap_neg": gap.clip(upper=0.0),
    })  # fmt: skip


def leverage_design(returns: pd.Series, windows=(1, 5, 22)) -> pd.DataFrame:
    """Leverage terms for the leverage HAR (Corsi and Reno 2012): the mean of the negative part of the
    return, min(r, 0), over the last 1, 5 and 22 days. Values are zero or negative; uses information up
    to and including t.
    """
    neg = returns.clip(upper=0)
    return pd.DataFrame({f"neg_ret_{w}": neg.rolling(w).mean() for w in windows})


def forecast_log_linear(
    X: pd.DataFrame, y_log: pd.Series, splits, cols: list[str], variance_correction: bool = False
) -> pd.Series:
    """For each (train, test) split, fit OLS with intercept of y_log on X[cols] and predict the test rows.

    Training rows with any missing value (in X[cols] or y_log) are dropped, not filled. The result is
    indexed like X and is NaN outside the test blocks (and where a test row has a missing input).

    variance_correction: add s2, the residual variance of the training fit, to every prediction.
    A log-volatility regression predicts the *median* volatility; QLIKE compares variances, whose
    best forecast is E[vol**2] = exp(2 * mean + 2 * s2) under log-normal errors, so the volatility
    forecast should be exp(pred + s2). Uses training rows only.
    """
    out = pd.Series(np.nan, index=X.index)
    for train, test in splits:
        data = X.iloc[train][cols].join(y_log.iloc[train].rename("y")).dropna()
        design = np.column_stack([np.ones(len(data)), data[cols].to_numpy()])
        beta = np.linalg.lstsq(design, data["y"].to_numpy(), rcond=None)[0]
        s2 = float(np.var(data["y"].to_numpy() - design @ beta)) if variance_correction else 0.0

        test_x = X.iloc[test][cols]
        ok = test_x.notna().all(axis=1).to_numpy()
        pred = np.full(len(test_x), np.nan)
        pred[ok] = np.column_stack([np.ones(ok.sum()), test_x[ok].to_numpy()]) @ beta + s2
        out.iloc[test] = pred
    return out
