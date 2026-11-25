"""Can the headline be improved on development data? Candidates and rule of ADR 0005.

    .venv/Scripts/python.exe experiments/vol_improve.py [--smoke]

Five candidates (QLIKE estimation, weighted least squares, OVX-dynamics inputs, both, and a combination)
are tested one-sided against the headline (leverage HAR + OVX of the previous day) in the six cells,
Holm over 30 tests plus Hansen's SPA per cell. A candidate improves the headline if it passes Holm in at
least one cell and is significantly worse (unadjusted one-sided p < 0.05) in none. The improving
candidate with the largest mean gain across cells is then compared with the five benchmarks (Holm over
30). Development data only. Writes results/tables/improve_candidates.csv and improve_vs_benchmarks.csv.
"""

import argparse
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
from oilfc.models.baselines.qlike_fit import forecast_log_qlike  # noqa: E402
from oilfc.models.baselines.vol import (  # noqa: E402
    forecast_log_linear,
    har_design,
    leverage_design,
    ovx_features,
)

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
HEAD = HAR + LEV + ["log_ovx"]
OVXF = ["log_ovx_5", "log_ovx_22", "gap_pos", "gap_neg"]
HEADLINE = "lhar_ovx_bc_lag1"
CANDIDATES = ["C1_qlike", "C2_wls", "C3_ovx_dynamics", "C4_ovx_dynamics_qlike", "C5_combination"]
BENCHMARKS = ["naive", "har_bc", "lhar_bc", "garch_t", "gjr_t"]


def cell_forecasts(asset: str, h: int, panel: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    """Realised volatility, headline, benchmarks and the five candidates for one cell."""
    cache = ROOT / "data" / "processed" / f"forecasts_{asset}_{h}d.parquet"
    base_fc = pd.read_parquet(cache) if cache.exists() else headline_forecasts(asset, h, panel)
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    d = har_design(r, panel["ovx"].reindex(r.index).shift(1))
    X = d.join(leverage_design(r)).join(ovx_features(d["log_ovx"], har_design(r)["log_rv_22"]))
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    if smoke:
        splits = splits[:24]
    check = np.exp(forecast_log_linear(X, y, splits, HEAD, variance_correction=True))
    both = pd.concat([check, base_fc[HEADLINE]], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-9), "headline does not match cache"
    cand = {
        "C1_qlike": np.exp(forecast_log_qlike(X, y, splits, HEAD, "qlike")),
        "C2_wls": np.exp(forecast_log_qlike(X, y, splits, HEAD, "wls")),
        "C3_ovx_dynamics": np.exp(
            forecast_log_linear(X, y, splits, HEAD + OVXF, variance_correction=True)
        ),
        "C4_ovx_dynamics_qlike": np.exp(forecast_log_qlike(X, y, splits, HEAD + OVXF, "qlike")),
    }
    parts = np.log(base_fc[[HEADLINE, "gjr_t", "lhar_bc"]])
    cand["C5_combination"] = np.exp(parts.mean(axis=1))
    cols = ["y", HEADLINE, *BENCHMARKS]
    return pd.concat([base_fc[cols], pd.DataFrame(cand)], axis=1).dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="two cells, 24 monthly splits")
    a = ap.parse_args()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    cells = [("brent", 22), ("wti", 22)] if a.smoke else [
        (s, h) for s in ("brent", "wti") for h in (5, 22, 66)
    ]  # fmt: skip
    rows, spa, qls = [], {}, {}
    for asset, h in cells:
        df = cell_forecasts(asset, h, panel, a.smoke)
        q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
        qls[(asset, h)] = (df, q)
        for c in CANDIDATES:
            p_better = dm_better_p(q[HEADLINE], q[c], h)
            rows.append({
                "asset": asset, "horizon": h, "candidate": c, "n": len(df),
                "qlike": q[c].mean(), "qlike_headline": q[HEADLINE].mean(),
                "gain_pct": 100 * (1 - q[c].mean() / q[HEADLINE].mean()),
                "p_better": p_better, "p_worse": 1 - p_better,
            })  # fmt: skip
        spa[(asset, h)] = spa_pvalue(q[HEADLINE], pd.DataFrame({c: q[c] for c in CANDIDATES}), h)
        print(f"{asset} {h}-day done ({len(df)} origins)", flush=True)
    t = pd.DataFrame(rows)
    t["p_better_holm"] = holm_adjust(t["p_better"].to_numpy())
    t["spa_p_cell"] = [spa[(x, h)] for x, h in zip(t["asset"], t["horizon"], strict=True)]
    t.to_csv(
        ROOT
        / "results"
        / "tables"
        / ("improve_candidates_smoke.csv" if a.smoke else "improve_candidates.csv"),
        index=False,
    )
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(t.round(4).to_string(index=False))

    # selection rule of ADR 0005
    improving = []
    for c, g in t.groupby("candidate"):
        passes = (g["p_better_holm"] < 0.05).any()
        worse = (g["p_worse"] < 0.05).any()
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
    for (asset, h), (df, q) in qls.items():
        for b in BENCHMARKS:
            rows.append({
                "asset": asset, "horizon": h, "candidate": chosen, "benchmark": b, "n": len(df),
                "qlike_candidate": q[chosen].mean(), "qlike_benchmark": q[b].mean(),
                "gain_pct": 100 * (1 - q[chosen].mean() / q[b].mean()),
                "p_better": dm_better_p(q[b], q[chosen], h),
            })  # fmt: skip
    b = pd.DataFrame(rows)
    b["p_better_holm"] = holm_adjust(b["p_better"].to_numpy())
    b.to_csv(
        ROOT
        / "results"
        / "tables"
        / ("improve_vs_benchmarks_smoke.csv" if a.smoke else "improve_vs_benchmarks.csv"),
        index=False,
    )
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
