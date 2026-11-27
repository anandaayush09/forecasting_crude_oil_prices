"""Per-series inputs for the pooled net of Stage B (ADR 0004 and its amendment of 2026-11-27)."""

import numpy as np
import pandas as pd

from oilfc.eval.targets import forward_realized_vol
from oilfc.features.vol import daily_channels
from oilfc.models.baselines.vol import leverage_design

# Brent and WTI use OVX, EUR per USD uses EVZ, the other companions have no implied volatility index.
ANCHOR_WITH_IV = [
    "log_rv_5", "log_rv_22", "log_rv_63", "neg_ret_1", "neg_ret_5", "neg_ret_22", "log_ovx",
]  # fmt: skip
ANCHOR_WITHOUT_IV = [c for c in ANCHOR_WITH_IV if c != "log_ovx"]
# `log_iv` is the series' own implied volatility, 0 where it has none, and `iv_missing` is 1 there.
POOL_CHANNELS = ["ret_1", "abs_ret", "log_rv_5", "log_rv_22", "log_iv", "log_vix", "iv_missing"]


def pooled_channels(
    returns: pd.Series,
    iv: pd.Series | None,
    vix: pd.Series,
    series: str,
    series_names: list[str],
) -> pd.DataFrame:
    """Anchor design columns, the POOL_CHANNELS and a one-hot column per series, for one series.

    returns: daily log returns of the series, indexed by the date each return is public.
    iv: the series' own implied volatility (index level, not logged) on the same dates, NaN where the
    index does not exist; None for a series without one. vix: the VIX level on the same dates.
    Implied volatility and VIX are delayed one trading day (the strict timing of the headline), so
    every value at t uses information up to and including t. The one-hot columns are constants: through
    the first layer of the net they act as a learned embedding of the series.
    """
    panel = pd.DataFrame(
        {
            "ovx": np.nan if iv is None else iv.reindex(returns.index),
            "vix": vix.reindex(returns.index),
        }
    )
    out = daily_channels(panel, returns, ovx_lag=1).join(leverage_design(returns))
    out["iv_missing"] = out["log_ovx"].isna().astype(float)
    out["log_iv"] = out["log_ovx"].fillna(0.0)
    for name in series_names:
        out[f"is_{name}"] = float(name == series)
    return out


def pooled_table(frame: pd.DataFrame, returns: pd.Series, h: int) -> pd.DataFrame:
    """Adds the target `y` (log realised volatility over the next h returns) and `end`, the date of the
    last return that target uses. A training row is usable by a test origin only if its `end` lies before
    that origin."""
    out = frame.copy()
    out["y"] = np.log(forward_realized_vol(returns, h))
    out["end"] = pd.Series(returns.index, index=returns.index).shift(-h)
    return out
