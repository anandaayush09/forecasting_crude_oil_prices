"""Does new information beat the strongest volatility benchmark (leverage HAR + OVX)?

    .venv/Scripts/python.exe experiments/vol_exog.py

Base: log volatility on HAR (5, 22, 63-day), leverage terms and log OVX, with the QLIKE variance correction.
Each candidate adds one group of exogenous features (futures slope, inventories, GPR, EPU, VIX) or all
of them. Same splits for every model: expanding window, monthly refits, gap = horizon, test from 2012,
development data only. Two samples, because EIA stopped publishing futures prices after 2024-04-05:
  A: every group except futures, test 2012-01 to 2025-06
  B: every group including futures, test 2012-01 to 2024-04
Writes results/tables/vol_exog_{asset}_{h}d_{sample}.csv.
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
from oilfc.features.exog import GROUPS, exogenous_features  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

OUT = ROOT / "results" / "tables"
HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
BASE = ["log_rv_5", "log_rv_22", "log_rv_63", "neg_ret_1", "neg_ret_5", "neg_ret_22", "log_ovx"]


def run(asset: str, h: int, sample: str, panel: pd.DataFrame) -> pd.DataFrame:
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    X = (
        har_design(r, panel["ovx"].reindex(r.index))
        .join(leverage_design(r))
        .join(exogenous_features(panel, r.index))
    )
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    groups = {k: v for k, v in GROUPS.items() if sample == "B" or k != "futures"}
    specs = {"base_lhar_ovx": BASE}
    specs.update({f"+{k}": BASE + v for k, v in groups.items()})
    specs["+all"] = BASE + [c for v in groups.values() for c in v]
    pred = {
        m: np.exp(forecast_log_linear(X, y, splits, cols, variance_correction=True))
        for m, cols in specs.items()
    }
    df = pd.concat({"y": np.exp(y), **pred}, axis=1).dropna()
    ql = pd.DataFrame({m: qlike(df["y"], df[m]) for m in specs})
    ms = pd.DataFrame({m: mse_logvol(df["y"], df[m]) for m in specs})
    rows = []
    for m in specs:
        rows.append({
            "model": m, "n": len(df), "first": df.index.min().date(), "last": df.index.max().date(),
            "qlike": ql[m].mean(), "qlike_gain_vs_base_pct": 100 * (1 - ql[m].mean() / ql["base_lhar_ovx"].mean()),
            "dm_p_qlike_vs_base": dm_hln(ql["base_lhar_ovx"], ql[m], h)[1] if m != "base_lhar_ovx" else np.nan,
            "mse_logvol": ms[m].mean(), "mse_gain_vs_base_pct": 100 * (1 - ms[m].mean() / ms["base_lhar_ovx"].mean()),
            "dm_p_mse_vs_base": dm_hln(ms["base_lhar_ovx"], ms[m], h)[1] if m != "base_lhar_ovx" else np.nan,
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("model").join(model_confidence_set(ql, block_size=h))


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            for sample in ("A", "B"):
                t = run(asset, h, sample, panel)
                t.to_csv(OUT / f"vol_exog_{asset}_{h}d_{sample}.csv")
                cols = [
                    "n",
                    "last",
                    "qlike_gain_vs_base_pct",
                    "dm_p_qlike_vs_base",
                    "mse_gain_vs_base_pct",
                    "dm_p_mse_vs_base",
                    "in_set",
                ]
                print(f"\n{asset} {h}-day, sample {sample}")
                print(t[cols].round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
