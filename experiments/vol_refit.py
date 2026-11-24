"""Refit-frequency check: does refitting the log-linear models weekly instead of monthly change the headline?

    .venv/Scripts/python.exe experiments/vol_refit.py

Audrino and Chassot (2024) argue that HAR benchmarks are often fitted too rarely, which flatters machine
learning. The protocol refits monthly. This script refits the OLS models (HAR, leverage HAR, headline
leverage HAR + OVX(t-1), all with the QLIKE variance correction) every 7 calendar days instead, on the
same origins, and tests the headline against naive, HAR and leverage HAR in both settings (Holm
correction over the 18 comparisons of each setting). GARCH models are not refitted weekly (too slow,
and they are not the object of the argument). Development data only, expanding window, test 2012-01 to
2025-06. Writes results/tables/refit_robustness.csv.
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
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

OUT = ROOT / "results" / "tables"
HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
HEADLINE, COMPARATORS = "lhar_ovx_bc_lag1", ["naive", "har_bc", "lhar_bc"]
SETTINGS = {
    "monthly": {"refit_months": 1, "refit_days": None},
    "weekly": {"refit_months": 1, "refit_days": 7},
}


def forecasts(
    r: pd.Series, ovx_lag1: pd.Series, h: int, refit_months: int, refit_days
) -> pd.DataFrame:
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(
        r.index, pd.Timestamp("2012-01-01"), h, refit_months, refit_days=refit_days
    )
    base = har_design(r).join(leverage_design(r))
    X = base.join(har_design(r, ovx_lag1)["log_ovx"])
    fc = {"naive": np.exp(base["log_rv_22"])}
    for name, cols in (
        ("har_bc", HAR),
        ("lhar_bc", HAR + LEV),
        (HEADLINE, HAR + LEV + ["log_ovx"]),
    ):
        fc[name] = np.exp(forecast_log_linear(X, y, splits, cols, variance_correction=True))
    return pd.concat({"y": np.exp(y), **fc}, axis=1).dropna()


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows = []
    for setting, kw in SETTINGS.items():
        for asset in ("brent", "wti"):
            price = panel[f"{asset}_spot"]
            price = price.where(price > 0).dropna()
            r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
            assert_development_only(r.index, HOLDOUT_START)
            ovx1 = panel["ovx"].reindex(r.index).shift(1)
            for h in (5, 22, 66):
                df = forecasts(r, ovx1, h, **kw)
                q_head = qlike(df["y"], df[HEADLINE])
                for c in COMPARATORS:
                    q_c = qlike(df["y"], df[c])
                    rows.append({
                        "refit": setting, "asset": asset, "horizon": h, "comparator": c, "n": len(df),
                        "qlike_headline": q_head.mean(), "qlike_comparator": q_c.mean(),
                        "gain_pct": 100 * (1 - q_head.mean() / q_c.mean()),
                        "dm_p": dm_hln(q_c, q_head, h)[1],
                    })  # fmt: skip
                print(f"{setting} {asset} {h}-day done ({len(df)} origins)", flush=True)
    table = pd.DataFrame(rows)
    table["dm_p_holm"] = np.nan
    for setting in SETTINGS:
        sel = table["refit"] == setting
        table.loc[sel, "dm_p_holm"] = holm_adjust(table.loc[sel, "dm_p"].to_numpy())
    table.to_csv(OUT / "refit_robustness.csv", index=False)
    wide = table.pivot_table(
        index=["asset", "horizon", "comparator"], columns="refit", values=["gain_pct", "dm_p"]
    ).round(3)
    pd.set_option("display.width", 200)
    print(wide.to_string())
    print(table.groupby("refit").agg(p_lt_05=("dm_p", lambda v: int((v < 0.05).sum())),
                                     holm_lt_05=("dm_p_holm", lambda v: int((v < 0.05).sum()))))  # fmt: skip


if __name__ == "__main__":
    main()
