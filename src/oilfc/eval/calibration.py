"""Calibration of volatility forecasts: the Mincer-Zarnowitz regression in logs."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def mincer_zarnowitz(realized: pd.Series, forecast: pd.Series, h: int) -> dict:
    """Regress log realised volatility on log forecast volatility: log y = a + b log f + e.

    Inputs are annualised volatilities (same units), aligned on the forecast origin. A calibrated
    forecast has a = 0 and b = 1. wald_p tests both at once, with Newey-West (Bartlett) standard
    errors using h lags because h-day targets overlap. mean_log_ratio = mean(log(y / f)): negative
    means the forecast is too high on average (QLIKE tends to reward that). Pairs with a missing or
    non-positive value are dropped. Raises ValueError with fewer than 3 * h usable pairs.
    """
    d = pd.concat([realized.rename("y"), forecast.rename("f")], axis=1).dropna()
    d = d[(d["y"] > 0) & (d["f"] > 0)]
    if len(d) <= 3 * h:
        raise ValueError(f"need more than 3 * h = {3 * h} usable pairs, got {len(d)}")
    ly, lf = np.log(d["y"].to_numpy()), np.log(d["f"].to_numpy())
    res = sm.OLS(ly, sm.add_constant(lf)).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    diff = np.asarray(res.params) - np.array([0.0, 1.0])
    stat = float(diff @ np.linalg.inv(np.asarray(res.cov_params())) @ diff)
    return {
        "n": len(d),
        "intercept": float(res.params[0]),
        "slope": float(res.params[1]),
        "intercept_se": float(res.bse[0]),
        "slope_se": float(res.bse[1]),
        "wald_p": float(stats.chi2.sf(stat, 2)),
        "mean_log_ratio": float(np.mean(ly - lf)),
    }
