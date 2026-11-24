"""Score frozen live forecasts once enough later returns exist."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from oilfc.eval.losses import qlike
from oilfc.eval.targets import TRADING_DAYS
from oilfc.live.forecast import MANIFEST, verify_forecast


def realised_vol_after(returns: pd.Series, origin: pd.Timestamp, h: int) -> float:
    """sqrt(252 * mean(r^2)) over the h returns after `origin` (annualised, decimal); NaN if fewer exist."""
    pos = returns.index.searchsorted(pd.Timestamp(origin), side="right")
    window = returns.iloc[pos : pos + h]
    if len(window) < h:
        return float("nan")
    return float(np.sqrt(TRADING_DAYS * np.mean(window.to_numpy() ** 2)))


def score_forecasts(forecast_dir: Path, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (asset, origin, horizon, model) with the forecast, the realised volatility (NaN
    until h later returns exist) and the QLIKE loss. Raises if a file fails its hash check."""
    rows = []
    for path in sorted(Path(forecast_dir).glob("*.json")):
        if not verify_forecast(path):
            raise ValueError(f"{path.name} does not match its stored hash or the {MANIFEST}")
        c = json.loads(path.read_text(encoding="utf-8"))["content"]
        price = panel[f"{c['asset']}_spot"]
        price = price.where(price > 0).dropna()
        r = np.log(price).diff().dropna()
        for h, models in c["horizons"].items():
            real = realised_vol_after(r, pd.Timestamp(c["origin_date"]), int(h))
            for model, f in models.items():
                loss = np.nan
                if np.isfinite(real):
                    loss = float(np.asarray(qlike(pd.Series([real]), pd.Series([f])))[0])
                rows.append({"asset": c["asset"], "origin_date": c["origin_date"], "horizon": int(h),
                             "model": model, "forecast": f, "realised": real, "qlike": loss,
                             "run_utc": c["run_utc"]})  # fmt: skip
    return pd.DataFrame(rows)
