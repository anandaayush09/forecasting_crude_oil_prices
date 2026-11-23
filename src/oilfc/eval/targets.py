"""Realised-volatility targets and inputs.

Both functions take daily log returns (one row per trading day) and return annualised volatility in
decimal form (0.35, not 35): sqrt(252 * mean(r**2)) over a window. No mean is subtracted, the usual
definition of realised variance.

- past_realized_vol: window r[t-h+1 .. t]. Today's return is known at t, so this is a valid *input*.
- forward_realized_vol: window r[t+1 .. t+h]. This is the *target*: what we try to predict at origin t.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def past_realized_vol(returns: pd.Series, h: int) -> pd.Series:
    """Volatility over the last h returns, ending today. The first h-1 values are NaN."""
    return np.sqrt(TRADING_DAYS * (returns**2).rolling(h).mean())


def forward_realized_vol(returns: pd.Series, h: int) -> pd.Series:
    """Volatility over the next h returns, starting tomorrow. The last h values are NaN."""
    # The past window ending at t+h is r[t+1 .. t+h]; shifting it back by h rows puts it at origin t.
    return past_realized_vol(returns, h).shift(-h)
