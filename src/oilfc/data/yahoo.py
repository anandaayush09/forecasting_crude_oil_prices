"""Daily futures open, high, low and close from Yahoo Finance's chart endpoint (unofficial, free).

Used only for the range-based volatility inputs (ADR 0006). The data are front-month futures, not the
EIA spot prices, and Yahoo does not document or guarantee this endpoint.
"""

import pandas as pd
import requests

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_chart(payload: dict) -> pd.DataFrame:
    """Daily open, high, low, close indexed by date from a chart response (timestamps are Unix seconds)."""
    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    index = pd.to_datetime(result["timestamp"], unit="s").normalize()
    frame = pd.DataFrame({k: quote[k] for k in ("open", "high", "low", "close")}, index=index)
    frame.index.name = "date"
    return frame


def fetch_ohlc(symbol: str, timeout: int = 60) -> pd.DataFrame:
    """Download the full daily history of `symbol` (for example CL=F or BZ=F)."""
    r = requests.get(
        URL.format(symbol=symbol),
        params={"period1": 0, "period2": 2_000_000_000, "interval": "1d", "events": "history"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    r.raise_for_status()
    return parse_chart(r.json())
