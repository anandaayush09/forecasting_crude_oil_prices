"""Compare the Kaggle residual models with benchmarks refit monthly (the like-for-like check).

    .venv/Scripts/python.exe experiments/residual_vs_monthly.py [results_folder]

The deep-model pipeline refits yearly (the anchor too), the benchmark ladder monthly. This script scores
the saved Kaggle predictions (results/kaggle_residual/vol_residual_seq_predictions_*.csv) and the monthly
HAR + OVX and leverage HAR + OVX forecasts on exactly the same origins. results_folder
defaults to kaggle_residual (yearly anchor); pass kaggle_residual_monthly for the monthly-anchor rerun.
Writes results/tables/residual_vs_monthly_benchmarks.csv (or ..._<folder>.csv for another folder).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
RUNS = (("brent", 5), ("brent", 22), ("wti", 22))


def main(folder="kaggle_residual"):
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows = []
    for asset, h in RUNS:
        f = ROOT / "results" / folder / f"vol_residual_seq_predictions_{asset}_{h}d.csv"
        pred = pd.read_csv(f, index_col=0, parse_dates=True)
        price = panel[f"{asset}_spot"]
        price = price.where(price > 0).dropna()
        r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
        assert_development_only(r.index, HOLDOUT_START)
        X = har_design(r, panel["ovx"].reindex(r.index)).join(leverage_design(r))
        y = np.log(forward_realized_vol(r, h))
        splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
        bench = {
            "har_ovx_bc_monthly": HAR + ["log_ovx"],
            "lhar_ovx_bc_monthly": HAR + LEV + ["log_ovx"],
            "ovx_bc_monthly": ["log_ovx"],
        }
        for name, cols in bench.items():
            b = np.exp(forecast_log_linear(X, y, splits, cols, variance_correction=True)).rename(
                "bench"
            )
            d = pd.concat([pred, b], axis=1, join="inner").dropna()
            q = {
                m: qlike(d["y"], d[m])
                for m in ("transformer_seq", "mlp_seq", "ridge_window", "anchor_har_ovx_bc")
            }
            qb = qlike(d["y"], d["bench"])
            for m, qm in q.items():
                rows.append({
                    "asset": asset, "horizon": h, "n": len(d), "benchmark": name, "model": m,
                    "qlike_model": qm.mean(), "qlike_benchmark": qb.mean(),
                    "gain_vs_benchmark_pct": 100 * (1 - qm.mean() / qb.mean()),
                    "dm_p": dm_hln(qb, qm, h)[1],
                })  # fmt: skip
    out = pd.DataFrame(rows)
    suffix = "" if folder == "kaggle_residual" else f"_{folder}"
    out.to_csv(
        ROOT / "results" / "tables" / f"residual_vs_monthly_benchmarks{suffix}.csv", index=False
    )
    pd.set_option("display.width", 220)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:2])
