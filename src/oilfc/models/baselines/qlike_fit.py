"""Log-linear volatility forecasts estimated by QLIKE or by weighted least squares.

Same interface as `vol.forecast_log_linear` (expanding training rows per split, intercept, training
data only), but the coefficients are not plain OLS. The QLIKE fit minimises the loss the forecasts are
judged on, so the forecast needs no separate variance correction.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def qlike_coefficients(design: np.ndarray, y_log: np.ndarray) -> np.ndarray:
    """Coefficients minimising mean exp(2 (y - f)) - 2 (y - f) - 1 with f = design @ beta.

    y_log is the log of the realised volatility, so exp(2 (y - f)) is the squared ratio of realised to
    forecast volatility (the QLIKE variance ratio). Started from OLS plus the residual variance.
    """
    beta0, *_ = np.linalg.lstsq(design, y_log, rcond=None)
    beta0[0] += float(np.var(y_log - design @ beta0))

    def loss(beta):
        r = y_log - design @ beta
        z = np.exp(np.clip(2 * r, -50.0, 50.0))  # keeps line-search steps finite
        return float(np.mean(z - 2 * r - 1)), (2 * (1 - z)) @ design / len(y_log)

    res = minimize(loss, beta0, jac=True, method="L-BFGS-B")
    return res.x if np.all(np.isfinite(res.x)) else beta0


def wls_coefficients(design: np.ndarray, y_log: np.ndarray) -> tuple[np.ndarray, float]:
    """One OLS fit, then weighted least squares with weights 1 / fitted variance, exp(-2 fitted log vol).

    Returns the coefficients and the residual variance of the weighted fit (for the variance correction).
    """
    beta, *_ = np.linalg.lstsq(design, y_log, rcond=None)
    w = np.exp(-2 * (design @ beta))
    root = np.sqrt(w / w.mean())
    beta, *_ = np.linalg.lstsq(design * root[:, None], y_log * root, rcond=None)
    return beta, float(np.var(y_log - design @ beta))


def forecast_log_qlike(
    X: pd.DataFrame, y_log: pd.Series, splits, cols: list[str], method: str = "qlike"
) -> pd.Series:
    """Log-volatility forecasts for every test origin, estimated by `method` on each split's training rows.

    method: "qlike" (QLIKE minimisation, no extra correction) or "wls" (weighted least squares plus the
    QLIKE variance correction, the residual variance added to the forecast). Training rows with a
    missing value in X[cols] or y_log are dropped. The result is indexed like X and is NaN outside the
    test blocks and where a test row has a missing input.
    """
    if method not in ("qlike", "wls"):
        raise ValueError(f"unknown method {method!r}")
    out = pd.Series(np.nan, index=X.index)
    for train, test in splits:
        data = X.iloc[train][cols].join(y_log.iloc[train].rename("y")).dropna()
        design, y = _design(data[cols].to_numpy()), data["y"].to_numpy()
        if method == "qlike":
            beta, s2 = qlike_coefficients(design, y), 0.0
        else:
            beta, s2 = wls_coefficients(design, y)
        test_x = X.iloc[test][cols]
        ok = test_x.notna().all(axis=1).to_numpy()
        pred = np.full(len(test_x), np.nan)
        pred[ok] = _design(test_x[ok].to_numpy()) @ beta + s2
        out.iloc[test] = pred
    return out
