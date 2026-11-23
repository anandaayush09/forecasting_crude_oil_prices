"""Exogenous daily features from the aligned panel (futures curve, inventories, uncertainty indices).

All inputs in the panel are already aligned to their release dates (`oilfc.data.align.align_asof`),
so every feature at t uses only values published by t.
"""

import numpy as np
import pandas as pd

GROUPS = {
    "futures": ["fut_slope"],
    "inventories": ["stocks_chg_4w", "stocks_dev_1y"],
    "gpr": ["log_gpr"],
    "epu": ["log_epu"],
    "vix": ["log_vix"],
}


def exogenous_features(
    panel: pd.DataFrame, index: pd.DatetimeIndex, max_futures_age_days: int = 10
) -> pd.DataFrame:
    """Features on `index` (the forecast origins).

    fut_slope      log(contract 4 / contract 1): positive in contango, negative in backwardation. Set to
                   NaN when the latest futures price is older than `max_futures_age_days` (EIA stopped
                   publishing after 2024-04-05, and the alignment would otherwise carry that value forward).
    stocks_chg_4w  4-week (20 trading day) change in log US crude stocks excluding the SPR.
    stocks_dev_1y  log stocks minus their trailing one-year (250 trading day) mean.
    log_gpr, log_epu  log of the 5-day mean of the daily geopolitical-risk and policy-uncertainty indices.
    log_vix        log VIX (forward-filled over VIX holidays).
    """
    p = panel.reindex(index)
    stale = (p["wti_fut1_age"] > max_futures_age_days) | (p["wti_fut4_age"] > max_futures_age_days)
    positive = (p["wti_fut1"] > 0) & (
        p["wti_fut4"] > 0
    )  # 2020-04-20: front contract settled below zero
    slope = np.log(
        (p["wti_fut4"] / p["wti_fut1"]).where(positive & ~stale)
    )  # log only valid ratios
    log_stocks = np.log(p["crude_stocks"])
    return pd.DataFrame(
        {
            "fut_slope": slope,
            "stocks_chg_4w": log_stocks - log_stocks.shift(20),
            "stocks_dev_1y": log_stocks - log_stocks.rolling(250).mean(),
            "log_gpr": np.log(p["gpr_daily"].rolling(5).mean()),
            "log_epu": np.log(p["epu_daily"].rolling(5).mean()),
            "log_vix": np.log(p["vix"].ffill()),
        },
        index=index,
    )
