"""Transformer residuals on the HAR + OVX anchor (RQ3/RQ4), with an attention-free control.

    python experiments/vol_residual_seq.py                       # Brent, 22-day volatility, auto device
    python experiments/vol_residual_seq.py --device cuda --seeds 5
    python experiments/vol_residual_seq.py --smoke               # 13 monthly splits (2 residual refits), 1 seed, 5 epochs

Anchor: HAR + OVX regression on log volatility with the QLIKE variance correction. Residual models
see the last `seq_len` days of six channels and predict what the anchor misses; alpha (chosen on a
time-ordered validation block) shrinks the correction. Models: ridge on the flattened window (linear
control), a flat MLP (attention-free control) and a patch transformer, with similar parameter counts.
Protocol: expanding window, anchor refit monthly (as the benchmark ladder), residual models refit
every 12 months (--residual-every), training origins stop h + 1 rows before the first test origin,
test from 2012-01.
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
from oilfc.eval.losses import mse_logvol, qlike  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.vol import CHANNELS, daily_channels, lag_window  # noqa: E402
from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402
from oilfc.models.ml.residual import residual_forecast  # noqa: E402

ANCHOR = ["log_rv_5", "log_rv_22", "log_rv_63", "log_ovx"]


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
    ap.add_argument("--refit-months", type=int, default=1, help="anchor refit frequency")
    ap.add_argument(
        "--residual-every",
        type=int,
        default=12,
        help="residual models refit every this many splits (12 = yearly with monthly splits)",
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--test-start", default="2012-01-01")
    ap.add_argument(
        "--holdout-start",
        default="2025-07-01",
        help="clean holdout (configs/protocol.yaml); never evaluated here",
    )
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel_daily.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tables"))
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
    frame = daily_channels(panel, r)
    window = lag_window(frame, CHANNELS, a.seq_len)
    X = frame[ANCHOR].join(window)
    y = np.log(forward_realized_vol(r, a.h))
    splits = expanding_splits(r.index, pd.Timestamp(a.test_start), a.h, a.refit_months)
    if a.smoke:
        splits = splits[: a.residual_every + 1]  # two residual refits
    print(
        f"{len(splits)} anchor refits, residual models every {a.residual_every}; "
        f"{len(window.columns)} window columns",
        flush=True,
    )

    deep = dict(
        seq_len=a.seq_len, n_channels=len(CHANNELS), epochs=epochs, seeds=seeds, device=device
    )
    factories = {
        "ridge_window": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
        "mlp_seq": lambda: SeqResidualRegressor(arch="mlp", mlp_hidden=48, **deep),
        "transformer_seq": lambda: SeqResidualRegressor(arch="transformer", **deep),
    }
    params = {k: SeqResidualRegressor(arch=k.split("_")[0], mlp_hidden=48, **deep).n_parameters()
              for k in ("mlp_seq", "transformer_seq")}  # fmt: skip
    print("parameters per net:", params, flush=True)

    preds, alphas, seconds = {}, {}, {}
    for name, factory in factories.items():
        t0 = time.time()
        res = residual_forecast(X, y, splits, ANCHOR, list(window.columns), factory,
                                variance_correction=True,
                                residual_refit_every=a.residual_every)  # fmt: skip
        preds[name], alphas[name], anchor = res["pred"], res["alpha"].dropna(), res["anchor"]
        seconds[name] = time.time() - t0
        print(
            f"{name}: done in {seconds[name]:.0f}s, mean alpha {alphas[name].mean():.2f}",
            flush=True,
        )
    preds["anchor_har_ovx_bc"] = anchor

    df = pd.concat({"y": np.exp(y), **{k: np.exp(v) for k, v in preds.items()}}, axis=1).dropna()
    models = list(preds)
    ql = pd.DataFrame({m: qlike(df["y"], df[m]) for m in models})
    ms = pd.DataFrame({m: mse_logvol(df["y"], df[m]) for m in models})
    ref = "anchor_har_ovx_bc"
    rows = []
    for m in models:
        rows.append({
            "model": m, "n": len(df), "qlike": ql[m].mean(), "mse_logvol": ms[m].mean(),
            "qlike_gain_vs_anchor_pct": 100 * (1 - ql[m].mean() / ql[ref].mean()),
            "mse_gain_vs_anchor_pct": 100 * (1 - ms[m].mean() / ms[ref].mean()),
            "dm_p_qlike_vs_anchor": dm_hln(ql[ref], ql[m], a.h)[1] if m != ref else np.nan,
            "dm_p_mse_vs_anchor": dm_hln(ms[ref], ms[m], a.h)[1] if m != ref else np.nan,
            "mean_alpha": alphas[m].mean() if m in alphas else np.nan,
            "share_alpha_zero": (alphas[m] == 0).mean() if m in alphas else np.nan,
            "parameters": params.get(m, np.nan), "seconds": seconds.get(m, np.nan),
        })  # fmt: skip
    table = pd.DataFrame(rows).set_index("model")
    try:
        from oilfc.eval.mcs import model_confidence_set

        table = table.join(model_confidence_set(ql, block_size=a.h))
    except (ImportError, RuntimeError) as e:  # keep the other results if the MCS cannot run
        print(f"skipping the Model Confidence Set: {e}")

    # attention vs no attention, head to head (RQ4)
    table.attrs["transformer_vs_mlp_qlike_dm_p"] = dm_hln(
        ql["mlp_seq"], ql["transformer_seq"], a.h
    )[1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.asset}_{a.h}d" + ("_smoke" if a.smoke else "")
    table.to_csv(out / f"vol_residual_seq_{tag}.csv")
    df.to_csv(out / f"vol_residual_seq_predictions_{tag}.csv")
    (out / f"vol_residual_seq_{tag}.json").write_text(json.dumps({
        "args": vars(a), "device": device, "seeds": list(seeds), "epochs": epochs,
        "n_refits": len(splits), "transformer_vs_mlp_qlike_dm_p": table.attrs["transformer_vs_mlp_qlike_dm_p"],
    }, indent=2))  # fmt: skip
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(4).to_string())
    print(
        "transformer vs MLP, QLIKE Diebold-Mariano p:",
        round(table.attrs["transformer_vs_mlp_qlike_dm_p"], 4),
    )


if __name__ == "__main__":
    main()
