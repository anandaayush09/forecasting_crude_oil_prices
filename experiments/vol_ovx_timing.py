"""Timing check: does the OVX advantage survive when OVX is lagged one trading day?

    .venv/Scripts/python.exe experiments/vol_ovx_timing.py

OVX for day t is computed at the New York close of t. The Brent spot price (EIA RBRTE) is a London
assessment, and the WTI spot (RWTC) settles earlier in the New York day than the OVX print. A forecast
made "at the price close of t" may therefore not have had OVX(t) yet, and the part of the return from
t to t+1 that happens after OVX(t) is computed is not independent of it. The strict version uses
OVX(t-1) at origin t. If the advantage over the non-OVX benchmarks survives the lag, the timing
mismatch is not driving the result.
Writes results/tables/ovx_timing_{asset}_{h}d.csv (development data only, same splits as the ladder).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import mse_logvol, qlike  # noqa: E402
from oilfc.eval.mcs import model_confidence_set  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.models.baselines.garch import garch_forecast  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

OUT = ROOT / "results" / "tables"
HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]


def forecasts(asset: str, h: int, panel: pd.DataFrame) -> pd.DataFrame:
    """Realised volatility `y` and every model's forecast, one row per origin in the test period."""
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    ovx0 = panel["ovx"].reindex(r.index)
    ovx1 = ovx0.shift(1)  # strict: yesterday's OVX at today's origin
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    base = har_design(r).join(leverage_design(r))
    X = {
        "lag0": base.join(har_design(r, ovx0)["log_ovx"]),
        "lag1": base.join(har_design(r, ovx1)["log_ovx"]),
    }
    fc = {"naive": np.exp(base["log_rv_22"])}
    fc["har_bc"] = np.exp(forecast_log_linear(X["lag0"], y, splits, HAR, variance_correction=True))
    fc["lhar_bc"] = np.exp(
        forecast_log_linear(X["lag0"], y, splits, HAR + LEV, variance_correction=True)
    )
    for lag in ("lag0", "lag1"):
        for name, cols in (
            ("ovx_bc", ["log_ovx"]),
            ("har_ovx_bc", HAR + ["log_ovx"]),
            ("lhar_ovx_bc", HAR + LEV + ["log_ovx"]),
        ):
            fc[f"{name}_{lag}"] = np.exp(
                forecast_log_linear(X[lag], y, splits, cols, variance_correction=True)
            )
    for kind in ("garch", "gjr"):
        fc[f"{kind}_t"] = garch_forecast(r, splits, h, kind)["rv_h"]
    fc["ovx_raw_lag0"] = ovx0 / 100
    fc["ovx_raw_lag1"] = ovx1 / 100
    return pd.concat({"y": np.exp(y), **fc}, axis=1).dropna()


def run(asset: str, h: int, panel: pd.DataFrame) -> pd.DataFrame:
    df = forecasts(asset, h, panel)
    models = [c for c in df.columns if c != "y"]
    ql = pd.DataFrame({m: qlike(df["y"], df[m]) for m in models})
    ms = pd.DataFrame({m: mse_logvol(df["y"], df[m]) for m in models})
    rows = []
    for m in models:
        rows.append({
            "model": m, "n": len(df), "qlike": ql[m].mean(),
            "qlike_gain_vs_naive_pct": 100 * (1 - ql[m].mean() / ql["naive"].mean()),
            "dm_p_vs_naive": dm_hln(ql["naive"], ql[m], h)[1] if m != "naive" else np.nan,
            "mse_gain_vs_naive_pct": 100 * (1 - ms[m].mean() / ms["naive"].mean()),
            "dm_p_better_than_lhar_bc": dm_hln(ql["lhar_bc"], ql[m], h)[1] if m != "lhar_bc" else np.nan,
            "dm_p_better_than_har_bc": dm_hln(ql["har_bc"], ql[m], h)[1] if m != "har_bc" else np.nan,
            "dm_p_better_than_garch_t": dm_hln(ql["garch_t"], ql[m], h)[1] if m != "garch_t" else np.nan,
            "dm_p_better_than_gjr_t": dm_hln(ql["gjr_t"], ql[m], h)[1] if m != "gjr_t" else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("model").join(model_confidence_set(ql, block_size=h))


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            t = run(asset, h, panel)
            t.to_csv(OUT / f"ovx_timing_{asset}_{h}d.csv")
            keep = [
                m
                for m in t.index
                if m.startswith(("ovx_bc", "ovx_raw", "lhar_ovx_bc"))
                or m in ("naive", "har_bc", "lhar_bc")
            ]
            print(f"{asset} {h}-day")
            print(
                t.loc[
                    keep,
                    [
                        "qlike",
                        "qlike_gain_vs_naive_pct",
                        "dm_p_vs_naive",
                        "dm_p_better_than_lhar_bc",
                        "dm_p_better_than_har_bc",
                        "in_set",
                    ],
                ]
                .round(4)
                .to_string(),
                flush=True,
            )


if __name__ == "__main__":
    main()
