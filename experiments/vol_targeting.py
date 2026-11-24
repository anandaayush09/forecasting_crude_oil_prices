"""Economic value of the volatility forecasts: volatility targeting (exploratory, risk management only).

    .venv/Scripts/python.exe experiments/vol_targeting.py [--reuse]

A position in the spot return is scaled every day to w = min(2, 0.25 / forecast), with the forecast of
each model (5-day or 22-day horizon, strict OVX timing for the headline). A better forecast keeps the
risk actually delivered closer to the 25% target. The table reports realised volatility of the managed
returns, how much rolling 22-day volatility wanders around the target, how often it exceeds 1.5 x the
target, Sharpe, drawdown and turnover, against an unmanaged position (w = 1). No return forecast is
used, so this says nothing about return predictability, and the differences are descriptive only (no
significance test). The target (0.25) and cap (2) were fixed before running and were not tuned.
Forecasts come from vol_robustness.get_forecasts (development data only). Writes
results/tables/vol_targeting.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from vol_robustness import get_forecasts  # noqa: E402

from oilfc.eval.holdout import development_only  # noqa: E402
from oilfc.eval.vol_target import managed_returns, target_metrics, vol_target_weights  # noqa: E402

TARGET, CAP = 0.25, 2.0
MODELS = ("naive", "har_bc", "lhar_bc", "garch_t", "gjr_t", "lhar_ovx_bc_lag1")


def main():
    reuse = "--reuse" in sys.argv
    import yaml

    holdout = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
        "clean_holdout_start"
    ]
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows = []
    for asset in ("brent", "wti"):
        price = panel[f"{asset}_spot"]
        price = price.where(price > 0).dropna()
        r = development_only(np.log(price).diff().dropna(), holdout)
        for h in (5, 22):
            fc = get_forecasts(asset, h, panel, reuse)
            idx = r.index[(r.index >= fc.index[0]) & (r.index <= fc.index[-1])]
            ret = r.loc[idx]
            variants = {m: fc[m].reindex(idx) for m in MODELS}
            variants["unmanaged"] = pd.Series(TARGET, index=idx)  # w = target / target = 1
            for name, f in variants.items():
                managed = managed_returns(ret, f, TARGET, CAP)
                w = vol_target_weights(f, TARGET, CAP)
                rows.append({"asset": asset, "horizon": h, "model": name,
                             **target_metrics(managed, w, TARGET)})  # fmt: skip
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "tables" / "vol_targeting.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    cols = ["asset", "horizon", "model", "ann_vol", "vol_of_realised_vol", "breach_rate", "sharpe",
            "max_drawdown", "mean_weight", "turnover"]  # fmt: skip
    print(out[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
