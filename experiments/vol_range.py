"""Range-based volatility inputs (ADR 0006): do daily high-low ranges improve the headline?

    .venv/Scripts/python.exe experiments/vol_range.py

Needs data/raw/ohlc_CLF.csv (WTI) and data/raw/ohlc_BZF.csv (Brent): daily futures OHLC from Yahoo
Finance, fetched once (see ADR 0006). Three candidates add or substitute Parkinson or Garman-Klass
past-volatility terms (known through the previous day) in the headline regression. Same tests and rule
as ADR 0005, with a family of 18 tests. Development data only. Writes results/tables/range_candidates.csv
and, if a candidate improves the headline, range_vs_benchmarks.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from vol_ovx_timing import forecasts as headline_forecasts  # noqa: E402

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.spa import spa_pvalue  # noqa: E402
from oilfc.eval.stat_tests import dm_better_p  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.range_vol import load_ohlc, range_vol_design  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
PK = ["log_pk_5", "log_pk_22", "log_pk_63"]
GK = ["log_gk_5", "log_gk_22", "log_gk_63"]
HEAD = HAR + LEV + ["log_ovx"]
HEADLINE = "lhar_ovx_bc_lag1"
CANDIDATES = [
    "R1_headline_plus_parkinson",
    "R2_headline_plus_garman_klass",
    "R3_parkinson_replaces_har",
]
BENCHMARKS = ["naive", "har_bc", "lhar_bc", "garch_t", "gjr_t"]
OHLC_FILE = {"brent": "ohlc_BZF.csv", "wti": "ohlc_CLF.csv"}


def cell(asset: str, h: int, panel: pd.DataFrame) -> pd.DataFrame:
    cache = ROOT / "data" / "processed" / f"forecasts_{asset}_{h}d.parquet"
    base_fc = pd.read_parquet(cache) if cache.exists() else headline_forecasts(asset, h, panel)
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    ohlc = load_ohlc(ROOT / "data" / "raw" / OHLC_FILE[asset])
    d = har_design(r, panel["ovx"].reindex(r.index).shift(1))
    X = d.join(leverage_design(r))
    X = X.join(range_vol_design(ohlc, r.index, "parkinson")).join(
        range_vol_design(ohlc, r.index, "garman_klass")
    )
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    check = np.exp(forecast_log_linear(X, y, splits, HEAD, variance_correction=True))
    both = pd.concat([check, base_fc[HEADLINE]], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-9), "headline does not match cache"

    def fit(cols):
        return np.exp(forecast_log_linear(X, y, splits, cols, variance_correction=True))

    cand = {
        CANDIDATES[0]: fit(HEAD + PK),
        CANDIDATES[1]: fit(HEAD + GK),
        CANDIDATES[2]: fit(PK + LEV + ["log_ovx"]),
    }
    return pd.concat([base_fc[["y", HEADLINE, *BENCHMARKS]], pd.DataFrame(cand)], axis=1).dropna()


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
    t.to_csv(ROOT / "results" / "tables" / "range_candidates.csv", index=False)
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
    b.to_csv(ROOT / "results" / "tables" / "range_vs_benchmarks.csv", index=False)
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
