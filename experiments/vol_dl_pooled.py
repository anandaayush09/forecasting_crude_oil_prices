"""Stage B of the deep-learning addendum: one net shared by Brent, WTI and six companion series.

    python experiments/vol_dl_pooled.py --asset brent --h 22 --device cuda --seeds 5
    python experiments/vol_dl_pooled.py --smoke      # 13 monthly splits, 1 seed, 5 epochs, CPU is fine

Pre-registered in docs/adr/0004-deep-learning-addendum.md (Stage B and the amendment of 2026-11-27).
Same procedure as `vol_dl_addendum.py`: the anchor of Brent and WTI is the headline benchmark, the
final forecast is benchmark * exp(alpha * correction), the nets train on the QLIKE-consistent loss with a
bounded zero-start correction, residual models refit every 12 monthly splits, alpha is chosen on the last
20% of each training window. What changes: the net is trained on the anchor residuals of all eight
series, each with its own anchor, a one-hot series marker and a missing-indicator channel. The
companions only supply training rows. Needs data/processed/panel_companions.parquet
(`python -m oilfc.data.companions`). Development data only. Writes results/dl_pooled/ (predictions per
origin, a per-cell summary and the run settings); `dl_pooled_report.py` computes the registered tests.
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
sys.path.insert(0, str(ROOT / "experiments"))

from vol_dl_addendum import ANCHOR, load, one_sided_p  # noqa: E402

from oilfc.data.companions import load_companions  # noqa: E402
from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.pooled import (  # noqa: E402
    ANCHOR_WITH_IV,
    ANCHOR_WITHOUT_IV,
    POOL_CHANNELS,
    pooled_channels,
    pooled_table,
)
from oilfc.features.vol import daily_channels, lag_window  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, leverage_design  # noqa: E402
from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402
from oilfc.models.ml.pooled import pooled_residual_forecast  # noqa: E402

EVALUATED = ("brent", "wti")
COMPANIONS = ("ulsd", "gasoline", "jet", "propane", "henryhub", "eur")
SERIES = EVALUATED + COMPANIONS
OWN_IV = {"brent": "ovx", "wti": "ovx", "eur": "evz"}  # every other series has no index
# EVZ is carried at most this long, so that its end in 2025-03 shows up as missing
EVZ_CARRY = pd.Timedelta(days=5)


def series_inputs(name, panel, companions, holdout_start):
    """Development-period returns of one series, its own implied volatility (or None) and VIX."""
    if name in EVALUATED:
        price = panel[f"{name}_spot"]
    else:
        price = companions[name]
    price = price.where(price > 0).dropna()
    r = development_only(np.log(price).diff().dropna(), holdout_start)
    assert_development_only(r.index, holdout_start)
    iv = None
    if OWN_IV.get(name) == "ovx":
        iv = panel["ovx"].reindex(r.index)
    elif OWN_IV.get(name) == "evz":
        iv = companions["evz"].dropna().reindex(r.index, method="ffill", tolerance=EVZ_CARRY)
    vix = panel["vix"].reindex(r.index, method="ffill")
    return r, iv, vix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="brent", choices=list(EVALUATED))
    ap.add_argument("--h", type=int, default=22)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--residual-every", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=22)
    ap.add_argument("--bound", type=float, default=0.5)
    ap.add_argument("--test-start", default="2012-01-01")
    ap.add_argument("--holdout-start", default="2025-07-01", help="never evaluated here")
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel_daily.parquet"))
    ap.add_argument(
        "--companions", default=str(ROOT / "data" / "processed" / "panel_companions.parquet")
    )
    ap.add_argument("--out", default=str(ROOT / "results" / "dl_pooled"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    seeds = tuple(range(1 if a.smoke else a.seeds))
    epochs = 5 if a.smoke else a.epochs
    print(f"device={device} seeds={seeds} epochs={epochs} h={a.h} asset={a.asset}", flush=True)

    panel, r_target = load(a.asset, a.panel)
    companions = load_companions(a.companions)
    channels = POOL_CHANNELS + [f"is_{n}" for n in SERIES]
    tables, anchors = {}, {}
    for name in SERIES:
        r, iv, vix = series_inputs(name, panel, companions, a.holdout_start)
        frame = pooled_channels(r, iv, vix, name, list(SERIES))
        anchors[name] = ANCHOR_WITH_IV if iv is not None else ANCHOR_WITHOUT_IV
        window = lag_window(frame, channels, a.seq_len)
        tables[name] = pooled_table(frame[anchors[name]].join(window), r, a.h)
    r = development_only(r_target, a.holdout_start)
    target = tables[a.asset]
    window_cols = [c for c in target.columns if "__" in c]

    # the benchmark exactly as in Stage A, and a check that the pooled anchor columns are the same
    frame_a = daily_channels(panel, r, ovx_lag=1).join(leverage_design(r))
    pd.testing.assert_frame_equal(frame_a[ANCHOR], target[ANCHOR], check_names=False)
    y = np.log(forward_realized_vol(r, a.h))
    splits = expanding_splits(r.index, pd.Timestamp(a.test_start), a.h, 1)
    if a.smoke:
        splits = splits[: a.residual_every + 1]
    print(
        f"{len(splits)} monthly splits, {len(window_cols)} window columns, "
        + ", ".join(
            f"{n}:{int(t[['y', *anchors[n]]].dropna().shape[0])}" for n, t in tables.items()
        ),
        flush=True,
    )
    log_bench = forecast_log_linear(frame_a, y, splits, ANCHOR, variance_correction=True)

    deep = dict(
        seq_len=a.seq_len, n_channels=len(channels), epochs=epochs, seeds=seeds, device=device,
        loss="qlike", bound=a.bound, patch_len=2,
    )  # fmt: skip
    factories = {
        "pooled_mlp": lambda: SeqResidualRegressor(arch="mlp", mlp_hidden=48, **deep),
        "pooled_transformer": lambda: SeqResidualRegressor(arch="transformer", **deep),
    }
    logs, alphas, seconds = {}, {}, {}
    for name, factory in factories.items():
        t0 = time.time()
        res = pooled_residual_forecast(
            tables, anchors, window_cols, a.asset, splits, factory,
            residual_refit_every=a.residual_every,
        )  # fmt: skip
        seconds[name] = time.time() - t0
        alphas[name] = res["alpha"].dropna()
        logs[name] = log_bench + res["alpha"] * res["residual"]
        print(f"{name}: {seconds[name]:.0f}s, mean alpha {alphas[name].mean():.2f}", flush=True)

    df = pd.concat(
        {"y": np.exp(y), "benchmark": np.exp(log_bench)} | {k: np.exp(v) for k, v in logs.items()},
        axis=1,
    )
    df = df.dropna()
    q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
    rows = []
    for m in factories:
        rows.append({
            "variant": m, "n": len(df), "qlike": q[m].mean(), "qlike_benchmark": q["benchmark"].mean(),
            "gain_vs_benchmark_pct": 100 * (1 - q[m].mean() / q["benchmark"].mean()),
            "dm_p_better": one_sided_p(q["benchmark"], q[m], a.h),
            "mean_alpha": alphas[m].mean(), "share_alpha_zero": (alphas[m] == 0).mean(),
            "seconds": seconds[m],
        })  # fmt: skip
    table = pd.DataFrame(rows).set_index("variant")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.asset}_{a.h}d" + ("_smoke" if a.smoke else "")
    table.to_csv(out / f"dl_pooled_{tag}.csv")
    df.to_csv(out / f"dl_pooled_predictions_{tag}.csv")
    (out / f"dl_pooled_{tag}.json").write_text(
        json.dumps({"args": vars(a), "device": device, "seeds": list(seeds), "epochs": epochs,
                    "n_splits": len(splits), "series": list(SERIES), "adr": "0004 amendment 2026-11-27"}, indent=2)
    )  # fmt: skip
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
