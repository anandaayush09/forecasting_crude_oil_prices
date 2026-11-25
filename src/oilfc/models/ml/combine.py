"""Combining a learned forecast with the benchmark, in log volatility.

Both rules use only forecasts and realised targets that were already known at each test block.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from oilfc.eval.losses import qlike

MIN_GATE_OBS = 250


def equal_weight(log_bench, log_model):
    """Equal-weight average of two log-volatility forecasts."""
    return 0.5 * (np.asarray(log_bench) + np.asarray(log_model))


def _weight(z, a, b):
    return 1.0 / (1.0 + np.exp(-(a + b * z)))


def gated_combination(
    realised, bench, model, z, blocks, h: int, min_obs: int = MIN_GATE_OBS
) -> pd.DataFrame:
    """Logistic gate between the benchmark and a learned forecast, refit at every test block.

    realised, bench, model: annualised volatilities (decimals), one entry per row of the sample,
    NaN outside the test blocks. z: the gate variable on the same rows (known at the origin).
    blocks: integer positions of each test block, in time order (as from `expanding_splits`).
    h: the forecast horizon in rows. The log forecast is log(bench) + w * (log(model) - log(bench))
    with w = 1 / (1 + exp(-(a + b z))). For a block starting at position s, (a, b) minimise the mean
    QLIKE over the origins of earlier blocks whose target is realised before s (position + h < s).
    With fewer than `min_obs` such origins the weight is 0.5.
    Returns a DataFrame with columns pred (volatility) and weight, NaN outside the test blocks.
    """
    realised, bench, model, z = (np.asarray(v, dtype=float) for v in (realised, bench, model, z))
    out = pd.DataFrame(np.nan, index=range(len(bench)), columns=["pred", "weight"])
    log_gap = np.log(model) - np.log(bench)
    for i, block in enumerate(blocks):
        start = block[0]
        past = np.concatenate([b for b in blocks[:i]]) if i else np.array([], dtype=int)
        past = past[past + h < start]
        past = past[np.isfinite(realised[past] + bench[past] + model[past] + z[past])]
        a = b = 0.0
        fitted = len(past) >= min_obs
        if fitted:

            def loss(theta, past=past):
                w = _weight(z[past], *theta)
                forecast = np.exp(np.log(bench[past]) + w * log_gap[past])
                return float(np.mean(qlike(realised[past], forecast)))

            a, b = minimize(loss, x0=[0.0, 0.0], method="Nelder-Mead").x
        w = _weight(z[block], a, b) if fitted else np.full(len(block), 0.5)
        out.loc[block, "weight"] = w
        out.loc[block, "pred"] = np.exp(np.log(bench[block]) + w * log_gap[block])
    return out
