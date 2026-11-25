"""The single evaluation of the headline model on the clean holdout (origins from 2025-07-01).

    .venv/Scripts/python.exe experiments/holdout_final.py                        # dry run: checks only
    .venv/Scripts/python.exe experiments/holdout_final.py --confirm-single-run   # the real, one-time run

Rules (configs/protocol.yaml, ADR 0003): one run, one report, whatever the result. The script refuses to
run unless
  - the git tag `protocol-frozen` exists, is an ancestor of HEAD and configs/protocol.yaml is unchanged
    since the tag, and no tracked file has uncommitted edits;
  - results/holdout/holdout_v1.json does not exist yet (it is written once and never overwritten).
The dry run performs these checks and prints how many origins each cell will have. It computes no
forecasts and no loss. A bug found after the real run is fixed in a new, dated file, never by rerunning
over the old one.

Models and comparators come from the protocol file: the headline against `claim_comparators`, for the
asset and horizon grid below, with Holm correction over all comparisons of the run.
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.holdout_final import (  # noqa: E402
    check_frozen,
    evaluate,
    holdout_forecasts,
    sha256_file,
    src_changes_since_freeze,
    write_once,
)
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402

PROTOCOL = ROOT / "configs" / "protocol.yaml"
PANEL = ROOT / "data" / "processed" / "panel_daily.parquet"
OUT = ROOT / "results" / "holdout" / "holdout_v1.json"
ASSETS = ("brent", "wti")


def returns(panel: pd.DataFrame, asset: str) -> pd.Series:
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    return np.log(price).diff().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-single-run", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    start = pd.Timestamp(cfg["clean_holdout_start"])
    vol = cfg["volatility"]
    headline, comparators, horizons = (
        vol["headline_model"],
        vol["claim_comparators"],
        vol["horizons"],
    )
    models = tuple(dict.fromkeys([headline, *comparators]))

    head = check_frozen(ROOT)
    print(f"protocol frozen, HEAD {head[:7]}")
    changed = src_changes_since_freeze(ROOT)
    print(f"source files changed since the freeze tag: {changed or 'none'}")
    if OUT.exists():
        sys.exit(f"{OUT} exists: the holdout was already evaluated. Never rerun over it.")

    panel = pd.read_parquet(PANEL)
    panel.index = pd.to_datetime(panel.index)
    print(f"holdout start {start.date()}, data end {panel.index[-1].date()}")
    for asset in ASSETS:
        r = returns(panel, asset)
        after = int((r.index >= start).sum())
        for h in horizons:
            print(f"  {asset} {h}-day: {max(after - h, 0)} origins with a realised target")
    if not a.confirm_single_run:
        print("dry run: nothing was evaluated. Add --confirm-single-run for the one-time run.")
        return

    cells = []
    for asset in ASSETS:
        r = returns(panel, asset)
        for h in horizons:
            df = holdout_forecasts(r, panel["ovx"], h, start, models)
            for row in evaluate(df, h, headline, comparators):
                cells.append({"asset": asset, "horizon": h, **row})
            print(f"{asset} {h}-day done ({len(df)} origins)", flush=True)
    p_holm = holm_adjust(np.array([c["dm_p"] for c in cells]))
    for c, p in zip(cells, p_holm, strict=True):
        c["dm_p_holm"] = float(p)
    record = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "panel_sha256": sha256_file(PANEL),
        "code_commit": head,
        "src_files_changed_since_freeze": changed,
        "run_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "holdout_start": str(start.date()),
        "data_end": str(panel.index[-1].date()),
        "headline": headline,
        "comparators": comparators,
        "cells": cells,
    }
    digest = write_once(OUT, record)
    print(f"written {OUT} sha256 {digest}")
    print(pd.DataFrame(cells).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
