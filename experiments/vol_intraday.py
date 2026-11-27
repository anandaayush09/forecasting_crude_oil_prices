"""Intraday realised-volatility inputs for WTI (ADR 0009): do VOLARE 5-minute measures improve the headline?

    .venv/Scripts/python.exe experiments/vol_intraday.py

Needs data/raw/volare/realized_variance_futures_20261030.zip (VOLARE futures archive, never committed;
its hash is checked against the one fixed in ADR 0009) and data/raw/ohlc_CLF.csv (for the R3 control).
Three candidates add or substitute realised-volatility terms (known through the previous day) in the
headline regression. Same tests and rule as ADR 0006, with a family of 9 tests (WTI only: VOLARE has no
Brent). Development data only. Writes results/tables/intraday_candidates.csv, intraday_controls.csv
and, if a candidate improves the headline, intraday_vs_benchmarks.csv.
"""

import hashlib
import sys
import zipfile
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
from oilfc.features.intraday_vol import intraday_vol_design, load_volare  # noqa: E402
from oilfc.features.range_vol import load_ohlc, range_vol_design  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design  # noqa: E402

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
VOLARE_ZIP = ROOT / "data" / "raw" / "volare" / "realized_variance_futures_20261030.zip"
VOLARE_CSV_SHA256 = "458df82dbb209313e4bffb0b712e29738e3dc4d265700db6ed7340bb922fa9ae"  # ADR 0009
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
PK = ["log_pk_5", "log_pk_22", "log_pk_63"]
IRV = ["log_irv_5", "log_irv_22", "log_irv_63"]
IRVSS = ["log_irvss_5", "log_irvss_22", "log_irvss_63"]
HEAD = HAR + LEV + ["log_ovx"]
HEADLINE = "lhar_ovx_bc_lag1"
CANDIDATES = [
    "I1_headline_plus_rv5",
    "I2_rv5_replaces_har",
    "I3_rv5_subsampled_replaces_har",
]
H_SHORT = "H_short"
R3 = "R3_parkinson_replaces_har"
BENCHMARKS = ["naive", "har_bc", "lhar_bc", "garch_t", "gjr_t"]
HORIZONS = (5, 22, 66)


def check_volare_file() -> None:
    """Stop if the archive is not the file ADR 0009 fixed (VOLARE refreshes monthly)."""
    with zipfile.ZipFile(VOLARE_ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        digest = hashlib.sha256(z.read(name)).hexdigest()
    assert digest == VOLARE_CSV_SHA256, f"VOLARE file differs from ADR 0009: {digest}"


def cell(h: int, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Forecasts of one WTI cell on its common rows, plus R3 on its own rows (a descriptive control)."""
    cache = ROOT / "data" / "processed" / f"forecasts_wti_{h}d.parquet"
    base_fc = pd.read_parquet(cache) if cache.exists() else headline_forecasts("wti", h, panel)
    price = panel["wti_spot"]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), HOLDOUT_START)
    assert_development_only(r.index, HOLDOUT_START)
    volare = load_volare(VOLARE_ZIP, "CL")
    ohlc = load_ohlc(ROOT / "data" / "raw" / "ohlc_CLF.csv")
    d = har_design(r, panel["ovx"].reindex(r.index).shift(1))
    X = d.join(leverage_design(r)).join(intraday_vol_design(volare, r.index))
    X = X.join(range_vol_design(ohlc, r.index, "parkinson"))
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    check = np.exp(forecast_log_linear(X, y, splits, HEAD, variance_correction=True))
    both = pd.concat([check, base_fc[HEADLINE]], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], rtol=1e-9), "headline does not match cache"

    def fit(cols, frame=X):
        return np.exp(forecast_log_linear(frame, y, splits, cols, variance_correction=True))

    # the headline on the candidates' own training rows: only rows with the realised-volatility features
    has_rv = X[IRV].notna().all(axis=1)
    short = X.copy()
    short.loc[~has_rv, HEAD] = np.nan
    cand = {
        CANDIDATES[0]: fit(HEAD + IRV),
        CANDIDATES[1]: fit(IRV + LEV + ["log_ovx"]),
        CANDIDATES[2]: fit(IRVSS + LEV + ["log_ovx"]),
        H_SHORT: fit(HEAD, short),
    }
    main_df = pd.concat(
        [base_fc[["y", HEADLINE, *BENCHMARKS]], pd.DataFrame(cand)], axis=1
    ).dropna()
    return main_df, fit(PK + LEV + ["log_ovx"])


def gain_row(comparison, model, ref, h, q_model, q_ref):
    return {
        "comparison": comparison, "model": model, "reference": ref, "asset": "wti", "horizon": h,
        "n": len(q_model), "qlike_model": q_model.mean(), "qlike_reference": q_ref.mean(),
        "gain_pct": 100 * (1 - q_model.mean() / q_ref.mean()),
        "p_better": dm_better_p(q_ref, q_model, h),
    }  # fmt: skip


def main():
    check_volare_file()
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    rows, controls, spa, store = [], [], {}, {}
    for h in HORIZONS:
        df, r3 = cell(h, panel)
        q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
        store[h] = (df, q)
        for c in CANDIDATES:
            p = dm_better_p(q[HEADLINE], q[c], h)
            rows.append({
                "asset": "wti", "horizon": h, "candidate": c, "n": len(df),
                "qlike": q[c].mean(), "qlike_headline": q[HEADLINE].mean(),
                "gain_pct": 100 * (1 - q[c].mean() / q[HEADLINE].mean()),
                "p_better": p, "p_worse": 1 - p,
            })  # fmt: skip
            controls.append(gain_row("candidate_vs_H_short", c, H_SHORT, h, q[c], q[H_SHORT]))
        controls.append(
            gain_row("H_short_vs_headline", H_SHORT, HEADLINE, h, q[H_SHORT], q[HEADLINE])
        )
        both = pd.concat([df["y"], df[CANDIDATES[1]], r3.rename(R3)], axis=1).dropna()
        qi2, qr3 = qlike(both["y"], both[CANDIDATES[1]]), qlike(both["y"], both[R3])
        controls.append(gain_row("I2_vs_range_model_R3", CANDIDATES[1], R3, h, qi2, qr3))
        spa[h] = spa_pvalue(q[HEADLINE], pd.DataFrame({c: q[c] for c in CANDIDATES}), h)
        print(f"wti {h}-day done ({len(df)} origins)", flush=True)
    t = pd.DataFrame(rows)
    t["p_better_holm"] = holm_adjust(t["p_better"].to_numpy())
    t["spa_p_cell"] = [spa[h] for h in t["horizon"]]
    out = ROOT / "results" / "tables"
    t.to_csv(out / "intraday_candidates.csv", index=False)
    pd.DataFrame(controls).to_csv(out / "intraday_controls.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(t.round(4).to_string(index=False))
    print(pd.DataFrame(controls).round(4).to_string(index=False))
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
    for h, (df, q) in store.items():
        for b in BENCHMARKS:
            rows.append({
                "asset": "wti", "horizon": h, "candidate": chosen, "benchmark": b, "n": len(df),
                "qlike_candidate": q[chosen].mean(), "qlike_benchmark": q[b].mean(),
                "gain_pct": 100 * (1 - q[chosen].mean() / q[b].mean()),
                "p_better": dm_better_p(q[b], q[chosen], h),
            })  # fmt: skip
    b = pd.DataFrame(rows)
    b["p_better_holm"] = holm_adjust(b["p_better"].to_numpy())
    b.to_csv(out / "intraday_vs_benchmarks.csv", index=False)
    print(b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
