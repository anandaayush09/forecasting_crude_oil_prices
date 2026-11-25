"""Deep-learning addendum, stage A (pre-registered in docs/adr/0004-deep-learning-addendum.md).

    python experiments/vol_dl_addendum.py --asset brent --h 22 --device cuda --seeds 5 --placebo
    python experiments/vol_dl_addendum.py --smoke      # 13 monthly splits, 1 seed, 5 epochs, CPU is fine

Anchor: the headline benchmark (leverage HAR + OVX of the previous trading day, QLIKE variance
correction, refit monthly). Every learned variant is benchmark * exp(alpha * correction), so alpha = 0
is the benchmark. Nets train on the QLIKE-consistent loss with a bounded, zero-start correction;
ridge (the linear control) keeps squared error. Residual models refit every 12 monthly splits, with
the last 20% of each training window held out to choose alpha. Development data only.
Writes results/kaggle_dl_addendum/ (predictions per origin, a per-cell summary and the run settings);
`experiments/dl_addendum_report.py` computes the pre-registered Holm and SPA tests across cells.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.vol import CHANNELS, daily_channels, lag_window  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, leverage_design  # noqa: E402
from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402
from oilfc.models.ml.combine import equal_weight, gated_combination  # noqa: E402
from oilfc.models.ml.residual import residual_forecast  # noqa: E402

ANCHOR = ["log_rv_5", "log_rv_22", "log_rv_63", "neg_ret_1", "neg_ret_5", "neg_ret_22", "log_ovx"]
NETS = ("mlp_seq", "gru_seq", "transformer_seq")


def load(asset, panel_path):
    panel = pd.read_parquet(panel_path)
    panel.index = pd.to_datetime(panel.index)
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    return panel, np.log(price).diff().dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="brent", choices=["brent", "wti"])
    ap.add_argument("--h", type=int, default=22)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--residual-every", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=22)
    ap.add_argument("--bound", type=float, default=0.5)
    ap.add_argument("--placebo", action="store_true", help="also run the shuffled-OVX transformer")
    ap.add_argument("--test-start", default="2012-01-01")
    ap.add_argument("--holdout-start", default="2025-07-01", help="never evaluated here")
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel_daily.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "kaggle_dl_addendum"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    seeds = tuple(range(1 if a.smoke else a.seeds))
    epochs = 5 if a.smoke else a.epochs
    print(f"device={device} seeds={seeds} epochs={epochs} h={a.h} asset={a.asset}", flush=True)

    panel, r = load(a.asset, a.panel)
    r = development_only(r, a.holdout_start)
    assert_development_only(r.index, a.holdout_start)
    frame = daily_channels(panel, r, ovx_lag=1).join(leverage_design(r))
    window = lag_window(frame, CHANNELS, a.seq_len)
    X = frame[ANCHOR].join(window)
    y = np.log(forward_realized_vol(r, a.h))
    splits = expanding_splits(r.index, pd.Timestamp(a.test_start), a.h, 1)
    if a.smoke:
        splits = splits[: a.residual_every + 1]
    print(f"{len(splits)} monthly splits, {len(window.columns)} window columns", flush=True)

    log_bench = forecast_log_linear(X, y, splits, ANCHOR, variance_correction=True)
    deep = dict(
        seq_len=a.seq_len, n_channels=len(CHANNELS), epochs=epochs, seeds=seeds, device=device,
        loss="qlike", bound=a.bound, patch_len=2,
    )  # fmt: skip
    factories = {
        "ridge_window": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
        "mlp_seq": lambda: SeqResidualRegressor(arch="mlp", mlp_hidden=48, **deep),
        "gru_seq": lambda: SeqResidualRegressor(arch="gru", **deep),
        "transformer_seq": lambda: SeqResidualRegressor(arch="transformer", **deep),
    }

    def run(name, features, factory):
        t0 = time.time()
        res = residual_forecast(
            features, y, splits, ANCHOR, list(window.columns), factory,
            variance_correction=False, residual_refit_every=a.residual_every,
        )  # fmt: skip
        secs = time.time() - t0
        print(f"{name}: {secs:.0f}s, mean alpha {res['alpha'].mean():.2f}", flush=True)
        return log_bench + res["alpha"] * res["residual"], res["alpha"].dropna(), secs

    logs, alphas, seconds = {}, {}, {}
    for name, factory in factories.items():
        logs[name], alphas[name], seconds[name] = run(name, X, factory)
    if a.placebo:
        shuffled = frame.copy()
        ok = frame["log_ovx"].notna().to_numpy()  # keep the leading NaNs in place
        values = frame["log_ovx"].to_numpy().copy()
        values[ok] = np.random.default_rng(0).permutation(values[ok])
        shuffled["log_ovx"] = values
        X_p = frame[ANCHOR].join(lag_window(shuffled, CHANNELS, a.seq_len))
        logs["placebo_shuffled_ovx"], alphas["placebo_shuffled_ovx"], seconds["placebo_shuffled_ovx"] = (
            run("placebo_shuffled_ovx", X_p, factories["transformer_seq"])
        )  # fmt: skip

    vol = {k: np.exp(v) for k, v in logs.items()}
    realised = np.exp(y).to_numpy()
    bench = np.exp(log_bench).to_numpy()
    tf = vol["transformer_seq"].to_numpy()
    logs["transformer_eqw"] = pd.Series(equal_weight(log_bench, np.log(tf)), index=y.index)
    vol["transformer_eqw"] = np.exp(logs["transformer_eqw"])
    z = (frame["log_ovx"] - frame["log_rv_22"]).to_numpy()
    gate = gated_combination(realised, bench, tf, z, [t for _, t in splits], a.h)
    vol["transformer_gate"] = pd.Series(gate["pred"].to_numpy(), index=y.index)
    weights = pd.Series(gate["weight"].to_numpy(), index=y.index)

    df = pd.concat({"y": np.exp(y), "benchmark": np.exp(log_bench), **vol}, axis=1).dropna()
    q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
    rows = []
    for m in [c for c in df.columns if c not in ("y", "benchmark")]:
        rows.append({
            "variant": m, "n": len(df), "qlike": q[m].mean(), "qlike_benchmark": q["benchmark"].mean(),
            "gain_vs_benchmark_pct": 100 * (1 - q[m].mean() / q["benchmark"].mean()),
            "dm_p_better": dm_hln(q["benchmark"], q[m], a.h)[1],
            "mean_alpha": alphas[m].mean() if m in alphas else np.nan,
            "share_alpha_zero": (alphas[m] == 0).mean() if m in alphas else np.nan,
            "mean_gate_weight": weights.loc[df.index].mean() if m == "transformer_gate" else np.nan,
            "seconds": seconds.get(m, np.nan),
        })  # fmt: skip
    table = pd.DataFrame(rows).set_index("variant")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.asset}_{a.h}d" + ("_smoke" if a.smoke else "")
    table.to_csv(out / f"dl_addendum_{tag}.csv")
    df.to_csv(out / f"dl_addendum_predictions_{tag}.csv")
    (out / f"dl_addendum_{tag}.json").write_text(
        json.dumps({"args": vars(a), "device": device, "seeds": list(seeds), "epochs": epochs,
                    "n_splits": len(splits), "adr": "0004"}, indent=2)
    )  # fmt: skip
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
