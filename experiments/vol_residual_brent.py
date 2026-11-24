"""Shrunk residual models on the HAR + OVX anchor (Brent, 22-day volatility).

Run from the project root:  .venv/Scripts/python.exe experiments/vol_residual_brent.py
Anchor: HAR + OVX. Residual models (ridge, LightGBM, a small MLP) predict what the anchor misses from
extra price and macro features; alpha, chosen on a time-ordered validation block, shrinks the correction.
Protocol: expanding window, monthly refits, gap = horizon, test from 2012-01 (configs/protocol.yaml).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.losses import mse_logvol, qlike  # noqa: E402
from oilfc.eval.mcs import model_confidence_set  # noqa: E402
from oilfc.eval.rolling import expanding_splits  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.vol import daily_channels  # noqa: E402
from oilfc.models.ml.residual import residual_forecast  # noqa: E402

HOLDOUT_START = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())[
    "clean_holdout_start"
]
ANCHOR = ["log_rv_5", "log_rv_22", "log_rv_63", "log_ovx"]
FACTORIES = {
    "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "lgbm": lambda: LGBMRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=0,
        min_child_samples=50,
        verbose=-1,
    ),
    "mlp": lambda: make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(16,), alpha=1e-1, early_stopping=True, max_iter=300, random_state=0
        ),
    ),
}


def features(panel, r):
    """Shared daily channels plus a few extra price and macro features (all causal)."""
    X = daily_channels(panel, r)
    extra = pd.DataFrame(index=r.index)
    extra["ret_5"] = r.rolling(5).sum()
    extra["ret_22"] = r.rolling(22).sum()
    extra["neg_ret_22"] = r.clip(upper=0).rolling(22).sum()  # downside move (leverage effect)
    extra["t10y2y"] = panel["t10y2y"].reindex(r.index).ffill()
    extra["log_ovx_over_rv"] = X["log_ovx"] - X["log_rv_22"]
    return X.join(extra)


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    r = np.log(panel["brent_spot"]).diff().dropna()
    r = development_only(r, HOLDOUT_START)  # the clean holdout is never touched in development
    assert_development_only(r.index, HOLDOUT_START)
    X = features(panel, r)
    h = 22
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp("2012-01-01"), h, 1)
    cols = ANCHOR + [
        c for c in X.columns if c not in ANCHOR + ["log_rv_5", "log_rv_22", "log_rv_63"]
    ]
    preds, alphas = {}, {}
    for name, factory in FACTORIES.items():
        res = residual_forecast(X, y, splits, ANCHOR, cols, factory, variance_correction=True)
        preds[name], alphas[name] = res["pred"], res["alpha"].dropna()
        anchor = res["anchor"]
        print(name, "done; mean alpha", round(alphas[name].mean(), 3), flush=True)
    preds["anchor_har_ovx"] = anchor
    df = pd.concat({"y": np.exp(y), **{k: np.exp(v) for k, v in preds.items()}}, axis=1).dropna()
    models = list(preds)
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
                "mse_gain_vs_anchor_pct": 100 * (1 - ms[m].mean() / ms["anchor_har_ovx"].mean()),
                "qlike_gain_vs_anchor_pct": 100 * (1 - ql[m].mean() / ql["anchor_har_ovx"].mean()),
                "dm_p_mse_vs_anchor": dm_hln(ms["anchor_har_ovx"], ms[m], h)[1]
                if m != "anchor_har_ovx"
                else np.nan,
                "dm_p_qlike_vs_anchor": dm_hln(ql["anchor_har_ovx"], ql[m], h)[1]
                if m != "anchor_har_ovx"
                else np.nan,
                "mean_alpha": alphas[m].mean() if m in alphas else np.nan,
                "share_alpha_zero": (alphas[m] == 0).mean() if m in alphas else np.nan,
            }
        )
    table = pd.DataFrame(rows).set_index("model").join(model_confidence_set(ql, block_size=h))
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "vol_residual_brent_22d.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
