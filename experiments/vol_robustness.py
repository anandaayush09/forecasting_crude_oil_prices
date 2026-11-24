"""Robustness of the strict-timing headline: sub-periods, multiple-testing correction, calibration.

    .venv/Scripts/python.exe experiments/vol_robustness.py [--reuse]

Headline model: leverage HAR + OVX(t-1) with the QLIKE variance correction ("lhar_ovx_bc_lag1").
Forecasts come from vol_ovx_timing.forecasts (development data only, expanding window, monthly refits).
Writes results/tables/:
  robustness_periods.csv      headline against each benchmark in 2012-2019, 2020-2025 and without
                              the origins of calendar 2020, plus the full sample
  robustness_holm.csv         full-sample Diebold-Mariano p-values (rectangular and Bartlett kernel) with Holm
                              correction across the family
                              (headline vs 5 benchmarks x 2 assets x 3 horizons)
  robustness_calibration.csv  Mincer-Zarnowitz regression in logs and the mean log ratio, by model
--reuse loads forecasts cached in data/processed/forecasts_{asset}_{h}d.parquet instead of refitting.
The 'ex 2020' sample removes origins dated 2020; targets of the origins in late 2019 still reach into
2020, and the removed block is closed up before testing (a gap of at most h origins in the HAC sum).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from vol_ovx_timing import forecasts  # noqa: E402

from oilfc.eval.calibration import mincer_zarnowitz  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402

OUT = ROOT / "results" / "tables"
HEADLINE = "lhar_ovx_bc_lag1"
BENCHMARKS = ("naive", "har_bc", "lhar_bc", "garch_t", "gjr_t")
CALIBRATED = (*BENCHMARKS, HEADLINE, "ovx_raw_lag1")
PERIODS = {
    "full": lambda i: np.ones(len(i), bool),
    "2012-2019": lambda i: i.year <= 2019,
    "2020-2025": lambda i: i.year >= 2020,
    "ex_2020": lambda i: i.year != 2020,
}


def get_forecasts(asset, h, panel, reuse):
    cache = ROOT / "data" / "processed" / f"forecasts_{asset}_{h}d.parquet"
    if reuse and cache.exists():
        return pd.read_parquet(cache)
    df = forecasts(asset, h, panel)
    df.to_parquet(cache)
    return df


def compare(df, h):
    """Headline versus each benchmark on one sample: QLIKE gain over the benchmark and DM p."""
    q = {m: qlike(df["y"], df[m]) for m in (HEADLINE, *BENCHMARKS)}
    rows = []
    for b in BENCHMARKS:
        rows.append({
            "benchmark": b, "n": len(df), "qlike_headline": q[HEADLINE].mean(),
            "qlike_benchmark": q[b].mean(),
            "gain_vs_benchmark_pct": 100 * (1 - q[HEADLINE].mean() / q[b].mean()),
            "dm_p_headline_better": dm_hln(q[b], q[HEADLINE], h)[1],
            "dm_p_bartlett": dm_hln(q[b], q[HEADLINE], h, kernel="bartlett")[1],
        })  # fmt: skip
    return rows


def main():
    reuse = "--reuse" in sys.argv
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    periods, calibration = [], []
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            df = get_forecasts(asset, h, panel, reuse)
            for name, pick in PERIODS.items():
                for row in compare(df[pick(df.index)], h):
                    periods.append({"asset": asset, "horizon": h, "period": name, **row})
            for m in CALIBRATED:
                calibration.append({"asset": asset, "horizon": h, "model": m,
                                    **mincer_zarnowitz(df["y"], df[m], h)})  # fmt: skip
            print(f"{asset} {h}-day done ({len(df)} origins)", flush=True)
    periods = pd.DataFrame(periods)
    full = periods[periods["period"] == "full"].copy()
    full["dm_p_holm"] = holm_adjust(full["dm_p_headline_better"].to_numpy())
    full["dm_p_holm_bartlett"] = holm_adjust(full["dm_p_bartlett"].to_numpy())
    periods.to_csv(OUT / "robustness_periods.csv", index=False)
    full.drop(columns="period").to_csv(OUT / "robustness_holm.csv", index=False)
    pd.DataFrame(calibration).to_csv(OUT / "robustness_calibration.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(full.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
