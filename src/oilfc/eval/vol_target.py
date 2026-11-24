"""Volatility targeting: what a volatility forecast is worth to someone who sizes a position with it.

A position in an asset is scaled each day to w = min(cap, target / forecast), so that its expected
risk equals `target` (annualised). A better volatility forecast keeps the risk actually delivered
closer to the target. This needs no return forecast, so it is a risk-management test only: it says
nothing about whether the asset's return is predictable.

Information timing: the forecast made at the close of day t sets the position held on day t + 1, so the
managed return of day t + 1 is w_t * r_{t+1}.
"""

import numpy as np
import pandas as pd

from oilfc.eval.targets import TRADING_DAYS


def vol_target_weights(forecast: pd.Series, target: float, cap: float = 2.0) -> pd.Series:
    """target / forecast, capped at `cap` (no leverage above the cap). Forecast and target are annualised."""
    return (target / forecast).clip(upper=cap)


def managed_returns(
    returns: pd.Series, forecast: pd.Series, target: float, cap: float = 2.0
) -> pd.Series:
    """Daily returns of the volatility-targeted position. NaN until a forecast exists.

    The weight is lagged one day: the forecast at the close of t is applied to the return of t + 1.
    """
    w = vol_target_weights(forecast.reindex(returns.index), target, cap)
    return w.shift(1) * returns


def target_metrics(managed: pd.Series, weights: pd.Series, target: float, window: int = 22) -> dict:
    """Risk and return summary of a managed return series (log returns, decimal).

    ann_vol             annualised volatility of the managed returns over the whole period
    vol_of_realised_vol std of log(rolling realised vol / target), rolling window of `window` days:
                        how much the risk actually delivered wanders around the target (lower is better)
    breach_rate         share of days with rolling realised vol above 1.5 x target
    sharpe              mean over standard deviation, annualised, of the managed log returns
    max_drawdown        largest peak-to-trough fall of cumulative log return (a positive number)
    worst_22d           the worst sum of `window` consecutive managed returns
    mean_weight, turnover  average position and average absolute daily change in position
    """
    m = managed.dropna()
    rolling = np.sqrt(TRADING_DAYS * (m**2).rolling(window).mean()).dropna()
    cum = m.cumsum()
    w = weights.dropna()
    return {
        "n_days": len(m),
        "ann_vol": float(m.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": float(m.mean() / m.std() * np.sqrt(TRADING_DAYS))
        if m.std() > 0
        else float("nan"),
        "vol_of_realised_vol": float(np.log(rolling / target).std()),
        "breach_rate": float((rolling > 1.5 * target).mean()),
        "max_drawdown": float((cum.cummax() - cum).max()),
        "worst_22d": float(m.rolling(window).sum().min()),
        "mean_weight": float(w.mean()),
        "turnover": float(w.diff().abs().mean()) if len(w) > 1 else 0.0,
    }
