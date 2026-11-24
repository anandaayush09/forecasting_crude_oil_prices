"""Forecasts for the latest origin, written once and never changed.

The live test applies exactly the backtest procedure to the last available row: training origins stop
h + 1 rows before the origin (so every training target is realised), OLS refit on all of them, the
QLIKE variance correction, strict OVX timing (OVX of the previous trading day). The record is frozen:
it is written once as JSON with a SHA-256 of its content, listed in a manifest, and committed to git.
Information timing: a forecast at origin t uses prices through t and OVX through t-1 only. EIA publishes
prices about a week late, so the run date is later than the origin and part of the target window may
already have happened when the forecast is made. The code never sees those later returns.
"""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from oilfc.eval.targets import forward_realized_vol
from oilfc.models.baselines.garch import garch_forecast
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design

SPEC_VERSION = "1"
MODELS = ("naive", "har_bc", "lhar_bc", "lhar_ovx_bc_lag1", "garch_t", "gjr_t")
HORIZONS = (5, 22, 66)
HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
LOG_COLS = {
    "har_bc": HAR,
    "lhar_bc": HAR + LEV,
    "lhar_ovx_bc_lag1": HAR + LEV + ["log_ovx"],
}
MANIFEST = "MANIFEST.csv"


def live_forecasts(
    panel: pd.DataFrame,
    asset: str,
    horizons: tuple[int, ...] = HORIZONS,
    origin: pd.Timestamp | None = None,
    code_version: str | None = None,
) -> dict:
    """Annualised volatility forecasts (decimal) for the latest origin, per horizon and model.

    panel: daily panel with `{asset}_spot` and `ovx` (OVX in index points). origin: forecast as of an
    earlier date, using only data up to and including it (used by the tests and for reruns).
    Information timing: prices through the origin, OVX through the previous trading day.
    """
    price = panel[f"{asset}_spot"]
    price = price.where(price > 0).dropna()
    r = np.log(price).diff().dropna()
    if origin is not None:
        r = r.loc[:origin]
    ovx_lag1 = panel["ovx"].reindex(r.index).shift(1)
    base = har_design(r).join(leverage_design(r))
    X = base.join(har_design(r, ovx_lag1)["log_ovx"])
    n = len(r)
    out = {}
    for h in horizons:
        if n - 1 - h < 250:
            raise ValueError(f"only {n} returns: too few training rows for horizon {h}")
        y = np.log(forward_realized_vol(r, h))
        splits = [(np.arange(0, n - 1 - h), np.array([n - 1]))]  # same rule as expanding_splits
        row = {"naive": float(np.exp(base["log_rv_22"].iloc[-1]))}
        for name, cols in LOG_COLS.items():
            pred = forecast_log_linear(X, y, splits, cols, variance_correction=True)
            row[name] = float(np.exp(pred.iloc[-1]))
        for kind in ("garch", "gjr"):
            row[f"{kind}_t"] = float(garch_forecast(r, splits, h, kind)["rv_h"].iloc[-1])
        bad = [m for m, v in row.items() if not np.isfinite(v) or v <= 0]
        if bad:
            raise ValueError(f"no valid forecast for {bad} at horizon {h}")
        out[str(h)] = {m: round(row[m], 8) for m in MODELS}
    return {
        "asset": asset,
        "origin_date": str(r.index[-1].date()),
        "ovx_date_used": str(r.index[-2].date()),
        "ovx_value_used": float(ovx_lag1.iloc[-1]),
        "n_returns": n,
        "horizons": out,
        "code_version": code_version,
        "spec_version": SPEC_VERSION,
    }


def _digest(content: dict) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_forecast(rec: dict, out_dir: Path, run_utc: str | None = None) -> Path:
    """Write the record once as {asset}_{origin}.json with its SHA-256 and add a manifest row.

    Raises FileExistsError if a forecast for this asset and origin already exists: frozen forecasts are
    never overwritten.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_utc = run_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = {**rec, "run_utc": run_utc}
    sha = _digest(content)
    path = out_dir / f"{rec['asset']}_{rec['origin_date']}.json"
    with open(path, "x", encoding="utf-8") as f:  # "x": fail if the file exists
        f.write(json.dumps({"content": content, "sha256": sha}, indent=1, sort_keys=True) + "\n")
    manifest = out_dir / MANIFEST
    new = not manifest.exists()
    with open(manifest, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(["asset", "origin_date", "run_utc", "sha256", "code_version"])
        w.writerow([rec["asset"], rec["origin_date"], run_utc, sha, rec.get("code_version") or ""])
    return path


def verify_forecast(path: Path) -> bool:
    """True if the file's content matches its stored hash and, when listed, the manifest hash."""
    path = Path(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    sha = _digest(stored["content"])
    if sha != stored["sha256"]:
        return False
    manifest = path.parent / MANIFEST
    if manifest.exists():
        m = pd.read_csv(manifest, dtype=str)
        c = stored["content"]
        hit = m[(m["asset"] == c["asset"]) & (m["origin_date"] == c["origin_date"])]
        return len(hit) == 1 and hit["sha256"].iloc[0] == sha
    return True
