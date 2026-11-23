"""Brent volatility ladder and VaR/ES backtests (roadmap weeks 8-10).

Run from the project root:  .venv/Scripts/python.exe experiments/vol_risk_brent.py
Writes CSV tables to results/tables/. Everything uses the protocol draft in configs/protocol.yaml:
expanding window, monthly refits, a gap of one horizon, volatility tests from 2012 (OVX), price-only
risk tests from 2010.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval import risk  # noqa: E402
from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import crps_normal, mse_logvol, qlike  # noqa: E402
from oilfc.eval.mcs import model_confidence_set  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import TRADING_DAYS, forward_realized_vol  # noqa: E402
from oilfc.models.baselines.garch import garch_fhs_var_es, garch_forecast  # noqa: E402
from oilfc.models.baselines.vol import (  # noqa: E402
    ewma_vol,
    forecast_log_linear,
    har_design,
    leverage_design,
)

OUT = ROOT / "results" / "tables"
HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
HAR_COLS = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV_COLS = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]


def load(asset="brent"):
    """Daily log returns and OVX. WTI: prices <= 0 (2020-04-20) are removed first, so the return that
    spans the gap covers several trading days; Brent is used as is."""
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = np.log(price).diff().dropna()
    r = development_only(r, HOLDOUT_START)  # the clean holdout is never touched in development
    assert_development_only(r.index, HOLDOUT_START)
    return r, panel["ovx"].reindex(r.index)


def volatility_ladder(r, ovx, h=22, test_start="2012-01-01"):
    X = har_design(r, ovx).join(leverage_design(r))
    y = forward_realized_vol(r, h)
    splits = expanding_splits(r.index, pd.Timestamp(test_start), h, 1)
    fc = {
        "naive": np.exp(X["log_rv_22"]),
        "ewma": ewma_vol(r),
        "ovx_raw": ovx
        / 100,  # OVX as published, no fitted slope or intercept (the literature's usage)
        "har": np.exp(forecast_log_linear(X, np.log(y), splits, HAR_COLS)),
        "ovx": np.exp(forecast_log_linear(X, np.log(y), splits, ["log_ovx"])),
        "har_ovx": np.exp(forecast_log_linear(X, np.log(y), splits, HAR_COLS + ["log_ovx"])),
    }
    # QLIKE-consistent versions of the log-linear models (exp(pred + residual variance), train only)
    for name, cols in (
        ("har", HAR_COLS),
        ("ovx", ["log_ovx"]),
        ("har_ovx", HAR_COLS + ["log_ovx"]),
        ("lhar", HAR_COLS + LEV_COLS),  # leverage HAR (Corsi and Reno 2012)
        ("lhar_ovx", HAR_COLS + LEV_COLS + ["log_ovx"]),
    ):
        fc[f"{name}_bc"] = np.exp(
            forecast_log_linear(X, np.log(y), splits, cols, variance_correction=True)
        )
    garch_out = {}
    for kind in ("garch", "gjr"):
        garch_out[kind] = garch_forecast(r, splits, h, kind)
        fc[f"{kind}_t"] = garch_out[kind]["rv_h"]
    df = pd.concat({"y": y, **fc}, axis=1).dropna()
    models = list(fc)
    ql = pd.DataFrame({m: qlike(df["y"], df[m]) for m in models})
    ms = pd.DataFrame({m: mse_logvol(df["y"], df[m]) for m in models})
    rows = []
    for m in models:
        rows.append(
            {
                "model": m,
                "n": len(df),
                "qlike": ql[m].mean(),
                "mse_logvol": ms[m].mean(),
                "qlike_gain_vs_naive_pct": 100 * (1 - ql[m].mean() / ql["naive"].mean()),
                "mse_gain_vs_naive_pct": 100 * (1 - ms[m].mean() / ms["naive"].mean()),
                "dm_p_vs_naive_qlike": dm_hln(ql["naive"], ql[m], h)[1] if m != "naive" else np.nan,
                "dm_p_vs_har_qlike": dm_hln(ql["har"], ql[m], h)[1]
                if m not in ("har",)
                else np.nan,
                "dm_p_har_ovx_vs_model_qlike": dm_hln(ql[m], ql["har_ovx"], h)[1]
                if m != "har_ovx"
                else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("model")
    table["dm_p_ovx_bc_better_qlike"] = [
        dm_hln(ql[m], ql["ovx_bc"], h)[1] if m != "ovx_bc" else np.nan for m in models
    ]
    table["dm_p_ovx_bc_better_mse"] = [
        dm_hln(ms[m], ms["ovx_bc"], h)[1] if m != "ovx_bc" else np.nan for m in models
    ]
    mcs = model_confidence_set(ql, size=0.10, block_size=h)

    # density of the h-day return: zero-mean normal with the model's volatility, plus GARCH's own mean
    ret_h = r[::-1].rolling(h).sum()[::-1].shift(-1).reindex(df.index)
    crps = {}
    for m in models:
        crps[m] = np.asarray(crps_normal(ret_h, 0.0, df[m] * np.sqrt(h / TRADING_DAYS)))
    for kind in ("garch", "gjr"):
        g = garch_out[kind].reindex(df.index)
        crps[f"{kind}_t_with_mean"] = np.asarray(crps_normal(ret_h, g["mu_h"], g["sigma_h"]))
    table = table.join(mcs)
    table["crps_h_return"] = pd.Series({m: c.mean() for m, c in crps.items()})
    table["crps_dm_p_better_than_naive"] = pd.Series(
        {m: dm_hln(crps["naive"], c, h)[1] for m, c in crps.items() if m != "naive"}
    )
    table["crps_dm_p_ovx_bc_better"] = pd.Series(
        {m: dm_hln(c, crps["ovx_bc"], h)[1] for m, c in crps.items() if m != "ovx_bc"}
    )
    return table


def risk_backtests(r, test_start="2010-01-01", taus=(0.01, 0.05), h=10):
    splits = expanding_splits(r.index, pd.Timestamp(test_start), h, 1)
    g = {k: garch_forecast(r, splits, h, k) for k in ("garch", "gjr")}
    fhs = {k: garch_fhs_var_es(r, splits, (1, h), k, taus) for k in ("garch", "gjr")}
    sig_ew = ewma_vol(r) / np.sqrt(TRADING_DAYS)
    y1 = r.shift(-1)
    y10 = r[::-1].rolling(h).sum()[::-1].shift(-1)
    test_pos = np.concatenate([t for _, t in splits])
    idx = r.index[test_pos]
    rows, fz0_series = [], {}
    for tau in taus:
        # one-day forecasts for every origin; ten-day forecasts on non-overlapping origins
        f1, f10 = {}, {}
        for k, out in g.items():
            f1[f"{k}_t"] = (risk.t_var_es(out["mu1"], out["sigma1"], out["nu"], tau), out["sigma1"])
            f10[f"{k}_t"] = (risk.normal_var_es(out["mu_h"], out["sigma_h"], tau), out["sigma_h"])
        for k in ("garch", "gjr"):  # filtered historical simulation: fat-tailed h-day distribution
            f1[f"{k}_fhs"] = ((fhs[k][f"var_h1_{tau}"], fhs[k][f"es_h1_{tau}"]), g[k]["sigma1"])
            f10[f"{k}_fhs"] = (
                (fhs[k][f"var_h{h}_{tau}"], fhs[k][f"es_h{h}_{tau}"]),
                g[k]["sigma_h"],
            )
        f1["ewma_normal"] = (risk.normal_var_es(0.0, sig_ew, tau), sig_ew)
        f10["ewma_normal"] = (
            risk.normal_var_es(0.0, sig_ew * np.sqrt(h), tau),
            sig_ew * np.sqrt(h),
        )
        # historical simulation: empirical quantile of the last 1000 returns (or 10-day sums)
        for horizon, target, f in ((1, y1, f1), (h, y10, f10)):
            base = r if horizon == 1 else r.rolling(horizon).sum()
            q = base.rolling(1000).quantile(tau)
            es = base.rolling(1000).apply(
                lambda a, tau=tau: a[a <= np.quantile(a, tau)].mean(), raw=True
            )
            f["hist_sim"] = ((q, es), base.rolling(1000).std())
            for name, ((v, e), s) in f.items():
                d = pd.DataFrame({"y": target, "v": v, "e": e, "s": s}, index=r.index).loc[idx]
                d = d.dropna()
                if horizon > 1:
                    d = d.iloc[::horizon]  # non-overlapping origins
                if len(d) < 100:
                    continue
                fz0_series[(horizon, tau, name)] = pd.Series(
                    risk.fz0(d["y"], d["v"], d["e"], tau), index=d.index
                )
                hits = risk.var_hits(d["y"], d["v"])
                rows.append(
                    {
                        "horizon_days": horizon,
                        "tau": tau,
                        "model": name,
                        "n": len(d),
                        "hit_rate": hits.mean(),
                        "kupiec_p": risk.kupiec_pof(hits, tau)[1],
                        "christoffersen_cc_p": risk.christoffersen_cc(hits, tau)[1],
                        "es_test_p": risk.es_exceedance_test(d["y"], d["v"], d["e"], d["s"])[1],
                        "fz0": risk.fz0(d["y"], d["v"], d["e"], tau).mean(),
                    }
                )
    # Diebold-Mariano on FZ0, GJR-FHS against every other model (non-overlapping or 1-day origins: h = 1)
    dm_rows = []
    for (horizon, tau), group in pd.Series(fz0_series).groupby(level=[0, 1]):
        losses = pd.concat({k[2]: v for k, v in group.items()}, axis=1).dropna()
        for m in losses.columns:
            if m != "gjr_fhs":
                stat, p = dm_hln(losses[m], losses["gjr_fhs"], 1)
                dm_rows.append(
                    {
                        "horizon_days": horizon,
                        "tau": tau,
                        "model": m,
                        # > 0 means GJR-FHS has the lower (better) loss
                        "fz0_diff_model_minus_gjr_fhs": losses[m].mean() - losses["gjr_fhs"].mean(),
                        "dm_p": p,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(dm_rows)


def main(horizons=(5, 22, 66)):
    """Modes (arguments): `wti` runs the volatility ladder on WTI only; `risk-only` runs the VaR/ES
    backtests only; `ladder-only` skips the risk backtests; no argument runs everything for Brent."""
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    asset = "wti" if "wti" in sys.argv else "brent"
    r, ovx = load(asset)
    for h in () if "risk-only" in sys.argv else horizons:
        vol = volatility_ladder(r, ovx, h=h)
        vol.to_csv(OUT / f"vol_ladder_{asset}_{h}d.csv")
        print(f"\n{asset} {h}-day volatility ladder (test 2012-01 onward)")
        print(vol.round(4).to_string(), flush=True)
    if asset == "wti" or "ladder-only" in sys.argv:
        return
    rk, dm = risk_backtests(r)
    rk.to_csv(OUT / "risk_backtests_brent.csv", index=False)
    dm.to_csv(OUT / "risk_fz0_dm_brent.csv", index=False)
    print("\nVaR / ES backtests (price-only, test 2010-01 onward)")
    print(rk.round(4).to_string(index=False))
    print("\nFZ0: GJR-FHS vs each model (Diebold-Mariano, h = 1)")
    print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
