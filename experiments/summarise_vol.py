"""Write results/tables/SUMMARY.md from the result CSVs (so documents quote the tables, not memory).

.venv/Scripts/python.exe experiments/summarise_vol.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.multiplicity import holm_adjust  # noqa: E402

T = ROOT / "results" / "tables"
KEY = [
    "naive",
    "ewma",
    "har",
    "har_bc",
    "lhar_bc",
    "ovx_raw",
    "ovx_bc",
    "har_ovx_bc",
    "lhar_ovx_bc",
    "garch_t",
    "gjr_t",
]


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append(
            "| " + " | ".join([str(idx)] + [("" if pd.isna(v) else str(v)) for v in row]) + " |"
        )
    return "\n".join(lines)


def ladder(asset: str) -> str:
    out = []
    for h in (5, 22, 66):
        f = T / f"vol_ladder_{asset}_{h}d.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0)
        keep = [m for m in KEY if m in d.index]
        t = pd.DataFrame(
            {
                "QLIKE": d.loc[keep, "qlike"].round(4),
                "gain vs naive %": d.loc[keep, "qlike_gain_vs_naive_pct"].round(1),
                "p vs naive": d.loc[keep, "dm_p_vs_naive_qlike"].round(3),
                "p: ovx_bc better": d.loc[keep, "dm_p_ovx_bc_better_qlike"].round(3),
                "MSE(logvol) gain %": d.loc[keep, "mse_gain_vs_naive_pct"].round(1),
                "in 90% MCS": d.loc[keep, "in_set"].map({True: "yes", False: ""}),
            }
        )
        t.index.name = "model"
        out.append(
            f"**{asset.upper()}, {h}-day volatility (n = {int(d['n'].iloc[0])}), QLIKE**\n\n{md(t)}\n"
        )
    return "\n".join(out)


def exog_summary() -> str:
    """One row per asset, horizon and sample: do new data groups improve on leverage HAR + OVX?"""
    nl = chr(10)
    rows = []
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            for sample in ("A", "B"):
                f = T / f"vol_exog_{asset}_{h}d_{sample}.csv"
                if not f.exists():
                    continue
                d = pd.read_csv(f, index_col=0)
                e = d.drop(index="base_lhar_ovx")
                q, p = e["qlike_gain_vs_base_pct"], e["dm_p_qlike_vs_base"]
                better = e[(p < 0.05) & (q > 0)].index.tolist()
                worse = e[(p < 0.05) & (q < 0)].index.tolist()
                mse_ok = (e["dm_p_mse_vs_base"] < 0.05) & (e["mse_gain_vs_base_pct"] > 0)
                rows.append(
                    {
                        "asset": asset,
                        "horizon": h,
                        "sample": sample,
                        "n": int(d["n"].iloc[0]),
                        "best QLIKE gain %": f"{q.max():+.2f} ({q.idxmax()})",
                        "worst QLIKE gain %": f"{q.min():+.2f} ({q.idxmin()})",
                        "QLIKE significantly better": ", ".join(better) or "none",
                        "QLIKE significantly worse": ", ".join(worse) or "none",
                        "MSE(logvol) significantly better": ", ".join(e[mse_ok].index) or "none",
                    }
                )
    if not rows:
        return ""
    note = (
        "Sample A: no futures, test 2012-01 to 2025-06. Sample B: futures included, test 2012-01 to 2024-04 "
        "(EIA stopped publishing futures prices after 2024-04-05). Base: leverage HAR + OVX, QLIKE-corrected. "
        "Gains are against that base; significance is Diebold-Mariano p < 0.05."
    )
    table = md(pd.DataFrame(rows).set_index("asset"))
    return "**New data groups added to leverage HAR + OVX**" + nl + nl + note + nl + nl + table + nl


def timing_summary() -> str:
    """OVX at day t versus OVX at day t-1 (strict timing)."""
    nl = chr(10)
    blocks = []
    for asset in ("brent", "wti"):
        for h in (5, 22, 66):
            f = T / f"ovx_timing_{asset}_{h}d.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f, index_col=0)
            keep = [
                m
                for m in d.index
                if m.startswith(("ovx_bc", "ovx_raw", "lhar_ovx_bc"))
                or m in ("naive", "har_bc", "lhar_bc")
            ]
            t = pd.DataFrame(
                {
                    "QLIKE": d.loc[keep, "qlike"].round(4),
                    "gain vs naive %": d.loc[keep, "qlike_gain_vs_naive_pct"].round(1),
                    "p vs naive": d.loc[keep, "dm_p_vs_naive"].round(3),
                    "p: better than leverage HAR": d.loc[keep, "dm_p_better_than_lhar_bc"].round(3),
                    "p: better than HAR": d.loc[keep, "dm_p_better_than_har_bc"].round(3),
                    "p: better than GARCH-t": d.loc[keep, "dm_p_better_than_garch_t"].round(3),
                    "p: better than GJR-t": d.loc[keep, "dm_p_better_than_gjr_t"].round(3),
                    "in 90% MCS": d.loc[keep, "in_set"].map({True: "yes", False: ""}),
                }
            )
            t.index.name = "model"
            n = int(d["n"].iloc[0])
            blocks.append(
                f"**{asset.upper()}, {h}-day, OVX timing (n = {n})**" + nl + nl + md(t) + nl
            )
    return nl.join(blocks)


def residual_monthly_summary() -> str:
    """Kaggle residual models versus benchmarks refit monthly, on identical origins."""
    f = T / "residual_vs_monthly_benchmarks.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f).round(4)
    d = d[d["benchmark"] == "lhar_ovx_bc_monthly"].drop(columns=["benchmark"]).set_index("asset")
    title = "**Kaggle residual models vs monthly leverage HAR + OVX (same origins; gain > 0: model better)**"
    return title + nl + nl + md(d) + nl


def kaggle_monthly_summary() -> str:
    """Residual models from the monthly-anchor Kaggle rerun: gain vs their own anchor (Holm over all
    nine tests) and the same correction added to the monthly HAR + OVX benchmark."""
    base = ROOT / "results" / "kaggle_residual_monthly"
    nl = chr(10)
    cells = (("brent", 5), ("brent", 22), ("wti", 22))
    models = ("ridge_window", "mlp_seq", "transformer_seq")
    rows = []
    for asset, h in cells:
        f = base / f"vol_residual_seq_{asset}_{h}d.csv"
        if not f.exists():
            return ""
        d = pd.read_csv(f, index_col=0)
        for m in models:
            rows.append(
                {
                    "cell": f"{asset.upper()} {h}d",
                    "model": m,
                    "gain vs own anchor %": round(d.loc[m, "qlike_gain_vs_anchor_pct"], 2),
                    "p": d.loc[m, "dm_p_qlike_vs_anchor"],
                }
            )
    t = pd.DataFrame(rows)
    t["Holm p (9 tests)"] = holm_adjust(t["p"].to_numpy()).round(3)
    t["p"] = t["p"].round(3)
    f = T / "residual_vs_monthly_benchmarks_kaggle_residual_monthly.csv"
    if f.exists():
        g = pd.read_csv(f)
        g = g[g["model"].str.endswith("_on_benchmark_anchor")].copy()
        g["model"] = g["model"].str.replace("_on_benchmark_anchor", "")
        g["cell"] = g["asset"].str.upper() + " " + g["horizon"].astype(str) + "d"
        g = g.set_index(["cell", "model"])
        keys = list(zip(t["cell"], t["model"], strict=True))
        t["gain added to monthly HAR+OVX %"] = [
            round(g.loc[k, "gain_vs_benchmark_pct"], 2) for k in keys
        ]
        t["p (added)"] = [round(g.loc[k, "dm_p"], 3) for k in keys]
    t = t.set_index("cell")
    title = (
        "**Residual models, monthly-anchor Kaggle rerun (QLIKE gain; > 0: residual model better)**"
    )
    return title + nl + nl + md(t) + nl


def robustness_summary() -> str:
    """Sub-periods, Holm-corrected p-values and calibration of the strict-timing headline."""
    nl = chr(10)
    blocks = []
    f = T / "robustness_periods.csv"
    if f.exists():
        d = pd.read_csv(f)
        wide = d.pivot_table(
            index=["asset", "horizon", "period"], columns="benchmark", values="dm_p_headline_better"
        ).round(3)
        wide.columns = [f"p vs {c}" for c in wide.columns]
        gain = d[d["benchmark"] == "naive"].set_index(["asset", "horizon", "period"])
        wide.insert(0, "gain vs naive %", gain["gain_vs_benchmark_pct"].round(1))
        wide.insert(0, "n", gain["n"])
        wide = wide.reset_index()
        wide["asset"] = wide["asset"].str.upper()
        wide = wide.set_index("asset")
        blocks.append(
            "**Headline (leverage HAR + OVX(t-1)) by sample: gain over naive and Diebold-Mariano p"
            " that it beats each benchmark**" + nl + nl + md(wide) + nl
        )
    f = T / "robustness_holm.csv"
    if f.exists():
        d = pd.read_csv(f)
        d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
        d["p"] = (
            d["dm_p_headline_better"].round(3).astype(str)
            + " / "
            + d["dm_p_holm"].round(3).astype(str)
        )
        wide = d.pivot(index="cell", columns="benchmark", values="p")
        wide.index.name = "raw p / Holm p"
        blocks.append(
            "**Holm correction across the 30 headline comparisons (5 benchmarks x 6 cells)**"
            + nl
            + nl
            + md(wide)
            + nl
        )
        if "dm_p_bartlett" in d:
            d["pb"] = (
                d["dm_p_bartlett"].round(3).astype(str)
                + " / "
                + d["dm_p_holm_bartlett"].round(3).astype(str)
            )
            wideb = d.pivot(index="cell", columns="benchmark", values="pb")
            wideb.index.name = "Bartlett: raw p / Holm p"
            blocks.append(
                "**Same comparisons with a Bartlett-weighted variance (less noisy at 66 days)**"
                + nl
                + nl
                + md(wideb)
                + nl
            )
    f = T / "robustness_calibration.csv"
    if f.exists():
        d = pd.read_csv(f)
        d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
        star = d["wald_p"].lt(0.05).map({True: "*", False: ""})
        d["v"] = (
            d["slope"].round(2).astype(str)
            + " / "
            + d["mean_log_ratio"].round(2).astype(str)
            + star
        )
        wide = d.pivot(index="cell", columns="model", values="v")
        wide.index.name = "slope / mean log(y/f); * = rejects slope 1, intercept 0"
        blocks.append(
            "**Calibration (log-log Mincer-Zarnowitz, Newey-West)**" + nl + nl + md(wide) + nl
        )
    return nl.join(blocks)


def vol_targeting_summary() -> str:
    """Volatility-targeting evaluation (exploratory, risk management only)."""
    f = T / "vol_targeting.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    keep = ["model", "ann_vol", "vol_of_realised_vol", "breach_rate", "max_drawdown", "sharpe"]
    t = d.set_index("cell")[keep].round(3)
    title = (
        "**Volatility targeting, target 25% a year, cap 2 (exploratory; ann_vol should be near 0.25,"
        " lower vol_of_realised_vol and breach_rate are better; Sharpe is not informative)**"
    )
    return title + nl + nl + md(t) + nl


def fx_summary() -> str:
    """Exchange-rate channel added to the strict-timing headline."""
    f = T / "vol_fx.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    t = d.set_index("cell")[
        ["spec", "qlike_gain_pct", "dm_p_qlike", "dm_p_qlike_holm", "mse_gain_pct", "dm_p_mse"]
    ]
    t = t.round(3)
    title = "**Dollar-index channel added to the headline (gain > 0: better; Holm over 18 QLIKE tests)**"
    return title + nl + nl + md(t) + nl


def window_summary() -> str:
    """Rolling-window check: headline and HAR + OVX against HAR under four estimation windows."""
    f = T / "window_robustness.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    order = ["expanding", "rolling_1000", "rolling_504", "rolling_252"]

    def wide(comparison: str, extra: bool) -> pd.DataFrame:
        x = d[d["comparison"] == comparison]
        cols = {}
        for w in order:
            xw = x[x["window"] == w].set_index("cell")
            cols[f"{w} gain %"] = xw["gain_pct"].round(1)
            cols[f"{w} p"] = xw["dm_p"].round(3)
            if extra:
                cols[f"{w} HAR out of MCS"] = ~xw["mcs_har_in_set"].astype(bool)
        return pd.DataFrame(cols)

    counts = d[d["comparison"].str.startswith("headline_vs_")].groupby("window")
    tally = pd.DataFrame({
        "p < 0.05 (of 30)": counts["dm_p"].apply(lambda v: int((v < 0.05).sum())),
        "Holm p < 0.05 (of 30)": counts["dm_p_holm"].apply(lambda v: int((v < 0.05).sum())),
    }).loc[order]  # fmt: skip
    t1 = "**Headline against HAR by estimation window (gain > 0: headline better; p: Diebold-Mariano)**"
    t2 = "**HAR + OVX(t-1) against HAR by estimation window, and whether the 90% MCS of the four HAR variants drops HAR**"
    t3 = "**Headline against its five comparators: tests significant by window**"
    return (
        t1 + nl + nl + md(wide("headline_vs_har_bc", False)) + nl + nl
        + t2 + nl + nl + md(wide("har_ovx_vs_har", True)) + nl + nl
        + t3 + nl + nl + md(tally) + nl
    )  # fmt: skip


def refit_summary() -> str:
    """Weekly versus monthly refit of the log-linear models (headline against three comparators)."""
    f = T / "refit_robustness.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d vs " + d["comparator"]
    cols = {}
    for setting in ("monthly", "weekly"):
        x = d[d["refit"] == setting].set_index("cell")
        cols[f"{setting} gain %"] = x["gain_pct"].round(1)
        cols[f"{setting} p"] = x["dm_p"].round(3)
    title = "**Headline against naive, HAR and leverage HAR: monthly against weekly refit (Holm over 18 tests per setting)**"
    return title + nl + nl + md(pd.DataFrame(cols)) + nl


def dl_addendum_summary() -> str:
    """Pre-registered deep-learning addendum (ADR 0004): variants against the headline benchmark."""
    f = T / "dl_addendum.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    cols = {}
    for v in d["variant"].unique():
        x = d[d["variant"] == v].set_index("cell")
        cols[f"{v} gain %"] = x["gain_pct"].round(2)
        cols[f"{v} Holm p"] = x["dm_p_holm"].round(3)
    spa = d.groupby("cell")["spa_p_cell"].first().round(3).rename("SPA p (cell)")
    title = "**Deep-learning addendum: QLIKE gain over the headline benchmark (> 0: better), Holm p over 36 tests, SPA p per cell**"
    return title + nl + nl + md(pd.DataFrame(cols).join(spa)) + nl


def improve_summary() -> str:
    """Headline-improvement candidates of ADR 0005, against the headline."""
    f = T / "improve_candidates.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    cols = {}
    for c in d["candidate"].unique():
        x = d[d["candidate"] == c].set_index("cell")
        cols[f"{c} gain %"] = x["gain_pct"].round(2)
        cols[f"{c} Holm p"] = x["p_better_holm"].round(3)
    spa = d.groupby("cell")["spa_p_cell"].first().round(3).rename("SPA p (cell)")
    title = "**Headline-improvement candidates (ADR 0005): QLIKE gain over the headline (> 0: better), Holm p over 30 tests, SPA p per cell**"
    return title + nl + nl + md(pd.DataFrame(cols).join(spa)) + nl


def range_summary() -> str:
    """Range-based volatility candidates of ADR 0006, against the headline."""
    f = T / "range_candidates.csv"
    if not f.exists():
        return ""
    nl = chr(10)
    d = pd.read_csv(f)
    d["cell"] = d["asset"].str.upper() + " " + d["horizon"].astype(str) + "d"
    cols = {}
    for c in d["candidate"].unique():
        x = d[d["candidate"] == c].set_index("cell")
        cols[f"{c} gain %"] = x["gain_pct"].round(2)
        cols[f"{c} p"] = x["p_better"].round(3)
    spa = d.groupby("cell")["spa_p_cell"].first().round(3).rename("SPA p (cell)")
    title = "**Range-based volatility inputs (ADR 0006): QLIKE gain over the headline (> 0: better), one-sided p (no Holm column: every Holm p is 1.0), SPA p per cell**"
    return title + nl + nl + md(pd.DataFrame(cols).join(spa)) + nl


def main():
    parts = [
        "# Result summary (generated by experiments/summarise_vol.py; development data only)\n"
    ]
    for asset in ("brent", "wti"):
        parts.append(ladder(asset))
    for asset in ("brent", "wti"):
        label = "Brent" if asset == "brent" else "WTI"
        f = T / f"risk_backtests_{asset}.csv"
        if f.exists():
            d = pd.read_csv(f).round(4).set_index(["horizon_days", "tau", "model"])
            parts.append(
                f"**VaR / ES backtests ({label})**\n\n"
                + md(d.reset_index().set_index("horizon_days"))
                + "\n"
            )
        f = T / f"risk_fz0_dm_{asset}.csv"
        if f.exists():
            d = pd.read_csv(f).round(4)
            parts.append(
                f"**FZ0 ({label}), GJR-FHS vs each model (diff > 0: GJR-FHS better)**\n\n"
                + md(d.set_index("horizon_days"))
                + "\n"
            )
    f = T / "vol_residual_brent_22d.csv"
    if f.exists():
        d = pd.read_csv(f, index_col=0).drop(columns=["n"]).round(4)
        parts.append(
            "**Residual models on the HAR + OVX anchor (Brent, 22 days)**\n\n" + md(d) + "\n"
        )
    parts.append(exog_summary())
    parts.append(timing_summary())
    parts.append(residual_monthly_summary())
    parts.append(kaggle_monthly_summary())
    parts.append(robustness_summary())
    parts.append(vol_targeting_summary())
    parts.append(fx_summary())
    parts.append(window_summary())
    parts.append(refit_summary())
    parts.append(dl_addendum_summary())
    parts.append(improve_summary())
    parts.append(range_summary())
    (T / "SUMMARY.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", T / "SUMMARY.md")


if __name__ == "__main__":
    main()
