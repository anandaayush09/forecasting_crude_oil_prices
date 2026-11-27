"""EIA downloader (public historical XLS files, no API key): spot and futures prices, weekly stocks."""
from __future__ import annotations

import io

import pandas as pd
import requests

# freq: "d" daily, "w" weekly. NYMEX futures (RCLC1-4) stopped being published after 2024-04-05.
EIA_XLS = "https://www.eia.gov/dnav/{section}/hist_xls/{key}{freq}.xls"


def clean_eia_frame(raw: pd.DataFrame, key: str) -> pd.Series:
    """`raw` is the 'Data 1' sheet read with header=None: 2 header rows + a column-label row, then data."""
    body = raw.iloc[3:, :2].copy()
    body.columns = ["date", key]
    body["date"] = pd.to_datetime(body["date"], errors="coerce")
    body[key] = pd.to_numeric(body[key], errors="coerce")
    body = body.dropna(subset=["date"])
    s = body.set_index("date")[key].rename(key)
    s.index.name = "date"
    return s.sort_index()


def fetch_eia(key: str, freq: str = "d", timeout: int = 120, section: str = "pet") -> pd.Series:
    """One EIA historical file. `section` is the catalogue part of the URL: "pet" for petroleum, "ng" for natural gas."""
    r = requests.get(EIA_XLS.format(section=section, key=key, freq=freq), timeout=timeout)
    r.raise_for_status()
    raw = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", header=None)
    return clean_eia_frame(raw, key)


def fetch_eia_spot(key: str, timeout: int = 120) -> pd.Series:
    return fetch_eia(key, "d", timeout)
