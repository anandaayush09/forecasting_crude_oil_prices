"""Tuned architectures from the earlier thesis as bounded corrections to the headline (ADR 0007).

    python experiments/vol_dl_arch.py --asset brent --h 22 --device cuda --seeds 5
    python experiments/vol_dl_arch.py --smoke      # 13 monthly splits, 1 seed, 5 epochs, CPU is fine

Same procedure as `vol_dl_addendum.py` (ADR 0004): the anchor is the headline benchmark (leverage HAR +
OVX of the previous trading day, QLIKE variance correction, refit monthly), every variant is
benchmark * exp(alpha * correction) with a bounded zero-start net trained on the QLIKE-consistent loss,
residual models refit every 12 monthly splits, alpha chosen on the last 20% of each training window.
What changes: six architectures (nhits, tide, itransformer, tcn, tft, informer), the 12 input channels
of `features.vol.CHANNELS_EXT`, and the hyperparameters of `results/dl_arch/hparams.json`, tuned only on
data before 2012-01 by `vol_dl_tune.py`. A ridge control on the 12-channel window (22 days) is outside
the test family. Development data only. Writes results/dl_arch/ (predictions per origin, a per-cell
summary and the run settings); `dl_arch_report.py` computes the registered Holm and SPA tests.
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

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from vol_dl_addendum import ANCHOR, load, one_sided_p  # noqa: E402

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.vol import CHANNELS_EXT, dl_channels, lag_window  # noqa: E402
from oilfc.models.baselines.vol import forecast_log_linear, leverage_design  # noqa: E402
from oilfc.models.dl.encoders import ARCHITECTURES, regressor_params  # noqa: E402
from oilfc.models.dl.residual_seq import SeqResidualRegressor  # noqa: E402
from oilfc.models.ml.residual import residual_forecast  # noqa: E402

MAX_WINDOW = 66
# only for --smoke when no tuned file exists; the registered runs always read hparams.json
SMOKE_HP = dict(
    window=22,
    width=16,
    depth=1,
    dropout=0.1,
    lr=3e-3,
    weight_decay=1e-2,
    bound=0.5,
    heads=2,
    kernel=2,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="brent", choices=["brent", "wti"])
    ap.add_argument("--h", type=int, default=22)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--residual-every", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--test-start", default="2012-01-01")
    ap.add_argument("--holdout-start", default="2025-07-01", help="never evaluated here")
    ap.add_argument("--hparams", default=str(ROOT / "results" / "dl_arch" / "hparams.json"))
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel_daily.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dl_arch"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    seeds = tuple(range(1 if a.smoke else a.seeds))
    epochs = 5 if a.smoke else a.epochs
    hparams = json.loads(Path(a.hparams).read_text()) if Path(a.hparams).exists() else {}
    if not a.smoke:
        missing = set(ARCHITECTURES) - set(hparams)
        assert not missing, f"no tuned hyperparameters for {missing} in {a.hparams}"
    hparams = {k: hparams.get(k, SMOKE_HP) for k in ARCHITECTURES}
    print(f"device={device} seeds={seeds} epochs={epochs} h={a.h} asset={a.asset}", flush=True)

    panel, r = load(a.asset, a.panel)
    r = development_only(r, a.holdout_start)
    assert_development_only(r.index, a.holdout_start)
    frame = dl_channels(panel, r).join(leverage_design(r))
    window = lag_window(frame, CHANNELS_EXT, MAX_WINDOW)
    X = frame[ANCHOR].join(window)
    y = np.log(forward_realized_vol(r, a.h))
    splits = expanding_splits(r.index, pd.Timestamp(a.test_start), a.h, 1)
    if a.smoke:
        splits = splits[: a.residual_every + 1]
    print(f"{len(splits)} monthly splits, {len(window.columns)} window columns", flush=True)

    n_ch = len(CHANNELS_EXT)
    cols_for = lambda w: list(window.columns[: w * n_ch])  # noqa: E731  (lag-major: first w days)
    log_bench = forecast_log_linear(X, y, splits, ANCHOR, variance_correction=True)
    variants = {
        "ridge_window": (cols_for(22), lambda: make_pipeline(StandardScaler(), Ridge(alpha=1000.0)))
    }
    for arch in ARCHITECTURES:
        hp = hparams[arch]
        params = regressor_params(hp, n_ch)
        variants[arch] = (
            cols_for(hp["window"]),
            lambda arch=arch, params=params: SeqResidualRegressor(
                arch=arch, epochs=epochs, seeds=seeds, device=device, **params
            ),
        )

    logs, alphas, seconds = {}, {}, {}
    for name, (cols, factory) in variants.items():
        t0 = time.time()
        res = residual_forecast(
            X, y, splits, ANCHOR, cols, factory,
            variance_correction=False, residual_refit_every=a.residual_every,
        )  # fmt: skip
        seconds[name] = time.time() - t0
        alphas[name] = res["alpha"].dropna()
        logs[name] = log_bench + res["alpha"] * res["residual"]
        print(f"{name}: {seconds[name]:.0f}s, mean alpha {res['alpha'].mean():.2f}", flush=True)

    df = pd.concat(
        {"y": np.exp(y), "benchmark": np.exp(log_bench), **{k: np.exp(v) for k, v in logs.items()}},
        axis=1,
    ).dropna()
    q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
    rows = []
    for m in [c for c in df.columns if c not in ("y", "benchmark")]:
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
    table.to_csv(out / f"dl_arch_{tag}.csv")
    df.to_csv(out / f"dl_arch_predictions_{tag}.csv")
    (out / f"dl_arch_{tag}.json").write_text(
        json.dumps({"args": vars(a), "device": device, "seeds": list(seeds), "epochs": epochs,
                    "n_splits": len(splits), "hparams": hparams, "adr": "0007"}, indent=2)
    )  # fmt: skip
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
