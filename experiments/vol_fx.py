"""Does the exchange-rate channel add to the headline model? (the oil literature uses FX and stock inputs)

    .venv/Scripts/python.exe experiments/vol_fx.py

Base: the strict-timing headline, leverage HAR + OVX of the previous trading day, with the QLIKE variance
correction. Candidates add the broad US dollar index (FRED DTWEXBGS, aligned with a 5 business day lag):
its 22-day change, the log of its 22-day volatility, or both. Same splits as every other experiment:
expanding window, monthly refits, test 2012-01 to 2025-06, development data only. The S&P 500 is not
tested: FRED holds it only from 2016-11, which cannot cover the test period (VIX was tested in
vol_exog.py). Holm correction over all 18 comparisons. Writes results/tables/vol_fx.csv.
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
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.exog import fx_features  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

HOLDOUT = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())["clean_holdout_start"]
BASE = ["log_rv_5", "log_rv_22", "log_rv_63", "neg_ret_1", "neg_ret_5", "neg_ret_22", "log_ovx"]
SPECS = {
    "+usd_chg": ["usd_chg_22"],
    "+usd_vol": ["log_usd_vol_22"],
    "+both": ["usd_chg_22", "log_usd_vol_22"],
}


def run(asset: str, h: int, panel: pd.DataFrame) -> list[dict]:
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT)
    assert_development_only(r.index, HOLDOUT)
    ovx_lag1 = panel["ovx"].reindex(r.index).shift(1)  # strict timing, as in the headline
    X = har_design(r, ovx_lag1).join(leverage_design(r)).join(fx_features(panel, r.index))
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    cols = {"base": BASE, **{k: BASE + v for k, v in SPECS.items()}}
    pred = {
        m: np.exp(forecast_log_linear(X, y, splits, c, variance_correction=True))
        for m, c in cols.items()
    }
    df = pd.concat({"y": np.exp(y), **pred}, axis=1).dropna()
    ql = {m: qlike(df["y"], df[m]) for m in cols}
    ms = {m: mse_logvol(df["y"], df[m]) for m in cols}
    rows = []
    for m in SPECS:
        rows.append({"asset": asset, "horizon": h, "spec": m, "n": len(df),
                     "qlike_gain_pct": 100 * (1 - ql[m].mean() / ql["base"].mean()),
                     "dm_p_qlike": dm_hln(ql["base"], ql[m], h)[1],
                     "mse_gain_pct": 100 * (1 - ms[m].mean() / ms["base"].mean()),
                     "dm_p_mse": dm_hln(ms["base"], ms[m], h)[1]})  # fmt: skip
    return rows


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows = [row for a in ("brent", "wti") for h in (5, 22, 66) for row in run(a, h, panel)]
    out = pd.DataFrame(rows)
    out["dm_p_qlike_holm"] = holm_adjust(out["dm_p_qlike"].to_numpy())
    out.to_csv(ROOT / "results" / "tables" / "vol_fx.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
