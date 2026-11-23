"""Daily uncertainty indices from the authors' public files (no key needed).

- Geopolitical Risk, daily (Caldara and Iacoviello): `GPRD`, index 1985-2019 = 100.
- Economic Policy Uncertainty, US daily news-based index (Baker, Bloom and Davis).
Both are newspaper-based. The published files are the current versions, not real-time vintages.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

GPR_DAILY_XLS = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EPU_DAILY_CSV = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
HEADERS = {"User-Agent": "oil-forecasting-research/0.1"}


def parse_gpr_daily(df: pd.DataFrame) -> pd.Series:
    """`df` is the GPR daily sheet; DAY is an integer yyyymmdd, GPRD the daily index."""
    dates = pd.to_datetime(df["DAY"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    s = pd.Series(pd.to_numeric(df["GPRD"], errors="coerce").to_numpy(), index=dates, name="gpr_daily")
    s = s[s.index.notna()].dropna()
    s.index.name = "date"
    return s.sort_index()


def parse_epu_daily(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    dates = pd.to_datetime(df[["year", "month", "day"]], errors="coerce")
    s = pd.Series(pd.to_numeric(df["daily_policy_index"], errors="coerce").to_numpy(), index=dates, name="epu_daily")
    s = s[s.index.notna()].dropna()
    s.index.name = "date"
    return s.sort_index()


def fetch_gpr_daily(timeout: int = 120) -> pd.Series:
    r = requests.get(GPR_DAILY_XLS, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return parse_gpr_daily(pd.read_excel(io.BytesIO(r.content)))


def fetch_epu_daily(timeout: int = 120) -> pd.Series:
    r = requests.get(EPU_DAILY_CSV, timeout=timeout, headers=HEADERS)
    r.raise_for_status()
    return parse_epu_daily(r.text)
