"""Combining the headline with the range-based and OVX-dynamics models (ADR 0008).

    .venv/Scripts/python.exe experiments/vol_combine.py

Three equal-weight log-volatility combinations (headline + R3, headline + C3, headline + R3 + C3) are
tested one-sided against the headline in the six cells, Holm over 18 tests plus Hansen's SPA per cell.
The rule is that of ADR 0005. All forecasts are compared on the rows where R3 exists (the range data
start after OVX). Development data only. Writes results/tables/combine_candidates.csv and, if a
candidate improves the headline, combine_vs_benchmarks.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from vol_improve import HEAD, OVXF  # noqa: E402
from vol_range import BENCHMARKS, HEADLINE  # noqa: E402
from vol_range import cell as range_cell  # noqa: E402

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.spa import spa_pvalue  # noqa: E402
from oilfc.eval.stat_tests import dm_better_p  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.models.baselines.vol import (  # noqa: E402
    forecast_log_linear,
    har_design,
    leverage_design,
    ovx_features,
)

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
R3 = "R3_parkinson_replaces_har"
COMBINATIONS = {
    "M1_headline_r3": [HEADLINE, R3],
    "M2_headline_c3": [HEADLINE, "C3_ovx_dynamics"],
    "M3_headline_r3_c3": [HEADLINE, R3, "C3_ovx_dynamics"],
}
CANDIDATES = list(COMBINATIONS)


def combine_log(parts: pd.DataFrame) -> pd.Series:
    """Equal-weight average of volatility forecasts in log volatility (the geometric mean).

    One column per forecast, same origins. No weight is estimated, so no information beyond the
    forecasts themselves enters.
    """
    return np.exp(np.log(parts).mean(axis=1))


def c3_forecast(asset: str, h: int, panel: pd.DataFrame) -> pd.Series:
    """C3 of ADR 0005: headline inputs plus OVX dynamics, OLS with the QLIKE variance correction."""
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    d = har_design(r, panel["ovx"].reindex(r.index).shift(1))
    X = d.join(leverage_design(r)).join(ovx_features(d["log_ovx"], har_design(r)["log_rv_22"]))
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    return np.exp(forecast_log_linear(X, y, splits, HEAD + OVXF, variance_correction=True))


def cell(asset: str, h: int, panel: pd.DataFrame) -> pd.DataFrame:
    """Realised volatility, headline, benchmarks, R3, C3 and the three combinations for one cell."""
    df = range_cell(asset, h, panel)  # y, headline, benchmarks, R1 to R3 on rows where all exist
    df = df[["y", HEADLINE, *BENCHMARKS, R3]].join(
        c3_forecast(asset, h, panel).rename("C3_ovx_dynamics"), how="inner"
    )
    for name, parts in COMBINATIONS.items():
        df[name] = combine_log(df[parts])
    return df.dropna()


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows, spa, store = [], {}, {}
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            df = cell(asset, h, panel)
            q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
            store[(asset, h)] = (df, q)
            for c in CANDIDATES:
                p = dm_better_p(q[HEADLINE], q[c], h)
                rows.append({
                    "asset": asset, "horizon": h, "candidate": c, "n": len(df),
                    "qlike": q[c].mean(), "qlike_headline": q[HEADLINE].mean(),
                    "gain_pct": 100 * (1 - q[c].mean() / q[HEADLINE].mean()),
                    "p_better": p, "p_worse": 1 - p,
                })  # fmt: skip
            spa[(asset, h)] = spa_pvalue(
                q[HEADLINE], pd.DataFrame({c: q[c] for c in CANDIDATES}), h
            )
            print(f"{asset} {h}-day done ({len(df)} origins)", flush=True)
    t = pd.DataFrame(rows)
    t["p_better_holm"] = holm_adjust(t["p_better"].to_numpy())
    t["spa_p_cell"] = [spa[(a, h)] for a, h in zip(t["asset"], t["horizon"], strict=True)]
    t.to_csv(ROOT / "results" / "tables" / "combine_candidates.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(t.round(4).to_string(index=False))
    improving = []
    for c, g in t.groupby("candidate"):
        passes, worse = (g["p_better_holm"] < 0.05).any(), (g["p_worse"] < 0.05).any()
        print(
            f"{c}: passes Holm in a cell = {passes}, significantly worse in a cell = {worse}, mean gain {g['gain_pct'].mean():.2f}%"
        )
        if passes and not worse:
            improving.append((g["gain_pct"].mean(), c))
    if not improving:
        print("DECISION: no candidate improves the headline; the headline stays.")
        return
    chosen = max(improving)[1]
    print(f"DECISION: {chosen} improves the headline; comparing it with the benchmarks.")
    rows = []
    for (asset, h), (df, q) in store.items():
        for b in BENCHMARKS:
            rows.append({
                "asset": asset, "horizon": h, "candidate": chosen, "benchmark": b, "n": len(df),
                "qlike_candidate": q[chosen].mean(), "qlike_benchmark": q[b].mean(),
                "gain_pct": 100 * (1 - q[chosen].mean() / q[b].mean()),
                "p_better": dm_better_p(q[b], q[chosen], h),
            })  # fmt: skip
    b = pd.DataFrame(rows)
    b["p_better_holm"] = holm_adjust(b["p_better"].to_numpy())
    b.to_csv(ROOT / "results" / "tables" / "combine_vs_benchmarks.csv", index=False)
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
