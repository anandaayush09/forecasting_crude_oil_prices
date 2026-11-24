"""Window robustness: does the headline survive a rolling estimation window instead of an expanding one?

    .venv/Scripts/python.exe experiments/vol_window.py [--reuse]

Why: Haukvik et al. (2024) find that OVX adds nothing out of sample with a 504-day rolling window;
Delis et al. (2025) use 1000 days and Omer et al. (2026) 252 days. Our headline uses an expanding window
from 1987. This script reruns the strict-timing ladder (OVX of the previous trading day, monthly refits,
test 2012-01 to 2025-06, development data only) with the last `window` training rows and compares:
  1. the headline (leverage HAR + OVX lag 1) against naive, HAR, leverage HAR, GARCH-t and GJR-t, with
     Holm correction over the 30 tests of each window (5 comparators x 2 assets x 3 horizons);
  2. HAR + OVX against HAR (the Haukvik question) and the 90% Model Confidence Set over the four HAR
     variants, to show whether a set that keeps every model is a sign of no gain or of low power.
Writes results/tables/window_robustness.csv. --reuse loads cached forecasts from data/processed.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from vol_ovx_timing import forecasts  # noqa: E402

from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.mcs import model_confidence_set  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402

OUT = ROOT / "results" / "tables"
HEADLINE = "lhar_ovx_bc_lag1"
COMPARATORS = ["naive", "har_bc", "lhar_bc", "garch_t", "gjr_t"]
HAR_FAMILY = ["har_bc", "lhar_bc", "har_ovx_bc_lag1", "lhar_ovx_bc_lag1"]
WINDOWS = {"expanding": None, "rolling_1000": 1000, "rolling_504": 504, "rolling_252": 252}


def get_forecasts(asset, h, panel, name, window, reuse):
    cache = ROOT / "data" / "processed" / f"forecasts_window_{name}_{asset}_{h}d.parquet"
    if reuse and cache.exists():
        return pd.read_parquet(cache)
    df = forecasts(asset, h, panel, window)
    df.to_parquet(cache)
    return df


def compare(df: pd.DataFrame, h: int) -> tuple[list[dict], dict]:
    """Headline against each comparator, plus the HAR-versus-HAR+OVX test and the MCS of the HAR family."""
    models = sorted({HEADLINE, "har_ovx_bc_lag1", *COMPARATORS, *HAR_FAMILY})
    q = {m: qlike(df["y"], df[m]) for m in models}
    rows = []
    for b in COMPARATORS:
        rows.append({
            "comparison": f"headline_vs_{b}", "n": len(df),
            "qlike_headline": q[HEADLINE].mean(), "qlike_other": q[b].mean(),
            "gain_pct": 100 * (1 - q[HEADLINE].mean() / q[b].mean()),
            "dm_p": dm_hln(q[b], q[HEADLINE], h)[1],
        })  # fmt: skip
    har = {
        "comparison": "har_ovx_vs_har", "n": len(df),
        "qlike_headline": q["har_ovx_bc_lag1"].mean(), "qlike_other": q["har_bc"].mean(),
        "gain_pct": 100 * (1 - q["har_ovx_bc_lag1"].mean() / q["har_bc"].mean()),
        "dm_p": dm_hln(q["har_bc"], q["har_ovx_bc_lag1"], h)[1],
    }  # fmt: skip
    mcs = model_confidence_set(pd.DataFrame({m: q[m] for m in HAR_FAMILY}), block_size=h)
    har["mcs_har_in_set"] = bool(mcs.loc["har_bc", "in_set"])
    har["mcs_set_size"] = int(mcs["in_set"].sum())
    return rows, har


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows = []
    for name, window in WINDOWS.items():
        for asset in ("brent", "wti"):
            for h in (5, 22, 66):
                df = get_forecasts(asset, h, panel, name, window, a.reuse)
                head, har = compare(df, h)
                for r in head + [har]:
                    rows.append({"window": name, "asset": asset, "horizon": h, **r})
                print(f"{name} {asset} {h}-day done ({len(df)} origins)", flush=True)
    table = pd.DataFrame(rows)
    table["dm_p_holm"] = float("nan")
    for name in WINDOWS:
        sel = (table["window"] == name) & table["comparison"].str.startswith("headline_vs_")
        table.loc[sel, "dm_p_holm"] = holm_adjust(table.loc[sel, "dm_p"].to_numpy())
    table.to_csv(OUT / "window_robustness.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    show = table[
        table["comparison"].isin(["headline_vs_har_bc", "headline_vs_naive", "har_ovx_vs_har"])
    ]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
