"""Realised-volatility inputs from the VOLARE futures archive (ADR 0009).

VOLARE (Cipollini et al. 2026, arXiv 2602.19732) publishes daily realised variance of futures built
from tick data. Only the 5-minute measures are used: `rv5` and its subsampled version `rv5_ss`. Both
are daily variances (squared log return per day), dated by trading day, the day closing at 17:00
Eastern, after the EIA spot price stamp of the same date.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd

from oilfc.features.range_vol import windowed_log_vol

MEASURES = {"irv": "rv5", "irvss": "rv5_ss"}  # column prefix -> VOLARE column


def load_volare(path, symbol: str = "CL") -> pd.DataFrame:
    """Daily `rv5` and `rv5_ss` (variance per day) of one futures symbol, indexed by trading date.

    `path` is the downloaded zip (its CSV is read in memory) or the CSV itself. Rows with a missing or
    non-positive measure are dropped. Nothing is cut at the holdout here: callers reindex to a
    development-only index.
    """
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            raw = io.BytesIO(z.read(name))
    else:
        raw = path
    d = pd.read_csv(raw, parse_dates=["date"], usecols=["date", "symbol", *MEASURES.values()])
    d = d[d["symbol"] == symbol].set_index("date").sort_index()
    d = d[~d.index.duplicated(keep="first")][list(MEASURES.values())].dropna()
    return d[(d > 0).all(axis=1)]


def intraday_vol_design(volare: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Log annualised realised volatility over 5, 22 and 63 days, known at each origin in `index`.

    Columns log_irv_<w> (from rv5) and log_irvss_<w> (from rv5_ss). The names differ from the
    return-based `log_rv_<w>` of the HAR design on purpose. The value at origin t uses the days up to
    and including t - 1. A window needs at least 80% of its days present.
    """
    parts = [windowed_log_vol(volare[col], index, tag) for tag, col in MEASURES.items()]
    return pd.concat(parts, axis=1)
