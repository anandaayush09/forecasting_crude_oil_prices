"""Daily input channels for volatility models, and lagged windows of them for sequence models."""

import numpy as np
import pandas as pd

from oilfc.models.baselines.vol import har_design

# Per-day channels fed to the sequence models. Every value at t uses information up to and including t.
CHANNELS = ["ret_1", "abs_ret", "log_rv_5", "log_rv_22", "log_ovx", "log_vix"]


def daily_channels(panel: pd.DataFrame, returns: pd.Series, ovx_lag: int = 0) -> pd.DataFrame:
    """The per-day channels in CHANNELS plus the HAR + OVX design columns (all causal).

    panel needs the columns `ovx` and `vix` (as in data/processed/panel_daily.parquet); `vix` is
    forward-filled (it has different holidays), which only carries past values forward.
    ovx_lag: trading days by which OVX and VIX are delayed. Both are New York closing values, later
    than the Brent price stamp, so ovx_lag=1 is the strict timing of the headline model.
    """
    design = har_design(returns, panel["ovx"].reindex(returns.index).shift(ovx_lag))
    out = design.copy()
    out["ret_1"] = returns
    out["abs_ret"] = returns.abs()
    out["log_vix"] = np.log(panel["vix"].reindex(returns.index).ffill().shift(ovx_lag))
    return out


def lag_window(frame: pd.DataFrame, channels: list[str], seq_len: int) -> pd.DataFrame:
    """Flatten the last `seq_len` days of `channels` into one row per origin.

    Columns are ordered lag-major, then channel: `<channel>__<lag>` with lag 0 = today (known at t)
    and lag seq_len - 1 = the oldest day. Reshaped as (rows, seq_len, n_channels) the first axis of
    the window is "newest first". Rows with fewer than seq_len days of history are NaN.
    """
    cols = {}
    for lag in range(seq_len):
        shifted = frame[channels].shift(lag)
        for c in channels:
            cols[f"{c}__{lag}"] = shifted[c]
    return pd.DataFrame(cols, index=frame.index)
