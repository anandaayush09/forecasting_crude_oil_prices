"""Stage B companion series: daily prices that only supply training rows to the pooled net.

    python -m oilfc.data.companions [--offline]

Reads `stage_b` and `stage_b_iv` of configs/data.yaml, caches each raw series in data/raw (like
`build.py`) and writes data/processed/panel_companions.parquet: one column per companion price, dated
at its availability date, and one column per implied volatility index, dated at its own observation.
"""

from __future__ import annotations

import argparse

import pandas as pd
import yaml

from .align import availability
from .build import PROCESSED, RAW, ROOT, _get
from .eia import fetch_eia
from .fred import fetch_fred

PANEL = PROCESSED / "panel_companions.parquet"


def _fetch(spec: dict) -> pd.Series:
    if spec["source"] == "eia":
        return fetch_eia(spec["key"], "d", section=spec["section"])
    if spec["source"] == "fred":
        return fetch_fred(spec["id"])
    raise ValueError(f"unknown source: {spec['source']}")


def at_availability(s: pd.Series, lag: dict) -> pd.Series:
    """Re-date `s` from its observation dates to the dates the values become public (d + lag)."""
    s = s.dropna().sort_index()
    out = pd.Series(s.to_numpy(), index=availability(s.index, lag["kind"], lag["n"]), name=s.name)
    return out[~out.index.duplicated(keep="last")]


def build_companions(offline: bool = False) -> pd.DataFrame:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, spec in cfg["stage_b"].items():
        cols[name] = at_availability(_get(name, lambda s=spec: _fetch(s), offline), spec["lag"])
    for name, spec in cfg["stage_b_iv"].items():
        cols[name] = at_availability(
            _get(name, lambda s=spec: fetch_fred(s["id"]), offline), spec["lag"]
        )
    panel = pd.DataFrame(cols).sort_index()
    panel.to_parquet(PANEL)
    return panel


def load_companions(path=PANEL) -> pd.DataFrame:
    """The wide frame written by `build_companions`; the index is the availability date."""
    panel = pd.read_parquet(path)
    panel.index = pd.to_datetime(panel.index)
    return panel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    p = build_companions(ap.parse_args().offline)
    print(p.apply(lambda c: (c.first_valid_index(), c.last_valid_index(), int(c.notna().sum()))).T)
