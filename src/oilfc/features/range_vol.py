"""Range-based past-volatility inputs from daily futures high, low, open and close.

Parkinson (1980) and Garman and Klass (1980) use the daily range, which carries part of the intraday
information that squared daily returns miss.
"""

import numpy as np
import pandas as pd

WINDOWS = (5, 22, 63)
MIN_SHARE = 0.8  # a window needs at least this share of its days present


def load_ohlc(path) -> pd.DataFrame:
    """Daily open, high, low, close from a CSV with a date index; rows with a missing field, a
    non-positive price or a high below the low are dropped."""
    o = pd.read_csv(path, index_col=0, parse_dates=True)[["open", "high", "low", "close"]]
    o = o.dropna()
    return o[(o > 0).all(axis=1) & (o["high"] >= o["low"])]


def daily_variance(ohlc: pd.DataFrame, kind: str) -> pd.Series:
    """Daily variance estimate (per day, not annualised): "parkinson" or "garman_klass"."""
    hl = np.log(ohlc["high"] / ohlc["low"])
    if kind == "parkinson":
        return hl**2 / (4 * np.log(2))
    if kind == "garman_klass":
        co = np.log(ohlc["close"] / ohlc["open"])
        return 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    raise ValueError(f"unknown kind {kind!r}")


def range_vol_design(
    ohlc: pd.DataFrame, index: pd.DatetimeIndex, kind: str = "parkinson"
) -> pd.DataFrame:
    """Log annualised range-based volatility over 5, 22 and 63 days, known at each origin in `index`.

    Columns log_pk_<w> (Parkinson) or log_gk_<w> (Garman-Klass). The value at origin t uses the ranges
    of the days up to and including t - 1 (the futures close is later than the spot price stamp, as
    with OVX). Windows are counted in rows of `index`; a window needs at least 80% of its days.
    """
    tag = "pk" if kind == "parkinson" else "gk"
    var = daily_variance(ohlc, kind).reindex(index)
    out = {}
    for w in WINDOWS:
        mean = var.rolling(w, min_periods=int(np.ceil(MIN_SHARE * w))).mean()
        out[f"log_{tag}_{w}"] = 0.5 * np.log(252 * mean.where(mean > 0)).shift(1)
    return pd.DataFrame(out)
