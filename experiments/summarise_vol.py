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


def main():
    parts = [
        "# Result summary (generated by experiments/summarise_vol.py; development data only)\n"
    ]
    for asset in ("brent", "wti"):
        parts.append(ladder(asset))
    f = T / "risk_backtests_brent.csv"
    if f.exists():
        d = pd.read_csv(f).round(4).set_index(["horizon_days", "tau", "model"])
        parts.append(
            "**VaR / ES backtests (Brent)**\n\n"
            + md(d.reset_index().set_index("horizon_days"))
            + "\n"
        )
    f = T / "risk_fz0_dm_brent.csv"
    if f.exists():
        d = pd.read_csv(f).round(4)
        parts.append(
            "**FZ0, GJR-FHS vs each model (diff > 0: GJR-FHS better)**\n\n"
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
    (T / "SUMMARY.md").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", T / "SUMMARY.md")


if __name__ == "__main__":
    main()
