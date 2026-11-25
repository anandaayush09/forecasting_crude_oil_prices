"""Final single evaluation on the clean holdout (origins from `clean_holdout_start`).

This module is deliberately separate from the development code: `oilfc.eval.holdout` guards every
development experiment against the holdout, and nothing here weakens that guard. The script that
calls this module (`experiments/holdout_final.py`) runs it once, after the protocol freeze.

Procedure (as in the frozen protocol, `configs/protocol.yaml`): expanding window from the start of the
sample, monthly refits, training origins stop h + 1 rows before the first origin of each block (every
training target is realised), OLS on log volatility with the QLIKE variance correction, OVX of the
previous trading day, Student-t GARCH and GJR. Information timing: a forecast at origin t uses prices
through t and OVX through t - 1. Targets are the annualised realised volatility over t+1..t+h and
exist only for origins at least h rows before the end of the data.
"""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from oilfc.eval.losses import qlike
from oilfc.eval.rolling import expanding_splits
from oilfc.eval.stat_tests import dm_hln
from oilfc.eval.targets import forward_realized_vol
from oilfc.models.baselines.garch import garch_forecast
from oilfc.models.baselines.vol import forecast_log_linear, har_design, leverage_design

HAR = ["log_rv_5", "log_rv_22", "log_rv_63"]
LEV = ["neg_ret_1", "neg_ret_5", "neg_ret_22"]
LOG_LINEAR = {
    "har_bc": HAR,
    "lhar_bc": HAR + LEV,
    "lhar_ovx_bc_lag1": HAR + LEV + ["log_ovx"],
}
KNOWN_MODELS = ("naive", *LOG_LINEAR, "garch_t", "gjr_t")


class NotFrozenError(RuntimeError):
    """Raised when the protocol freeze conditions for the holdout run are not met."""


def check_frozen(
    repo: Path, tag: str = "protocol-frozen", protocol: str = "configs/protocol.yaml"
) -> str:
    """Raise NotFrozenError unless the freeze tag exists, is an ancestor of HEAD, the protocol file is
    unchanged since the tag, and no tracked file has uncommitted edits. Returns the HEAD commit hash.
    """

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )

    if git("rev-parse", "--verify", "--quiet", f"refs/tags/{tag}").returncode != 0:
        raise NotFrozenError(f"tag {tag} does not exist: the protocol is not frozen")
    if git("merge-base", "--is-ancestor", tag, "HEAD").returncode != 0:
        raise NotFrozenError(f"tag {tag} is not an ancestor of HEAD")
    if git("diff", "--quiet", tag, "--", protocol).returncode != 0:
        raise NotFrozenError(f"{protocol} differs from the frozen version at {tag}")
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise NotFrozenError("tracked files have uncommitted changes")
    return git("rev-parse", "HEAD").stdout.strip()


def src_changes_since_freeze(
    repo: Path, tag: str = "protocol-frozen", path: str = "src"
) -> list[str]:
    """Files under `path` that differ between the freeze tag and HEAD.

    check_frozen guards the protocol file only. Model or evaluation code may legitimately change after
    the freeze (the holdout module itself is added later), so the list is recorded with the result and
    printed before the run, for a reader to judge, rather than blocked.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), "diff", "--name-only", tag, "HEAD", "--", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in out.stdout.splitlines() if line)


def holdout_forecasts(
    r: pd.Series, ovx: pd.Series, h: int, start, models: tuple[str, ...] = KNOWN_MODELS
) -> pd.DataFrame:
    """Realised volatility `y` and each model's forecast for origins dated `start` or later.

    r: daily log returns of the whole sample (the clean holdout included). ovx: OVX in index points,
    aligned to any index. Rows with an unrealised target are dropped. Volatilities are annualised
    decimals.
    """
    unknown = set(models) - set(KNOWN_MODELS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    y = np.log(forward_realized_vol(r, h))
    splits = expanding_splits(r.index, pd.Timestamp(start), h, 1)
    base = har_design(r).join(leverage_design(r))
    X = base.join(har_design(r, ovx.reindex(r.index).shift(1))["log_ovx"])
    fc = {}
    if "naive" in models:
        fc["naive"] = np.exp(base["log_rv_22"])
    for name, cols in LOG_LINEAR.items():
        if name in models:
            fc[name] = np.exp(forecast_log_linear(X, y, splits, cols, variance_correction=True))
    for name, kind in (("garch_t", "garch"), ("gjr_t", "gjr")):
        if name in models:
            fc[name] = garch_forecast(r, splits, h, kind)["rv_h"]
    return pd.concat({"y": np.exp(y), **fc}, axis=1).dropna()


def evaluate(df: pd.DataFrame, h: int, headline: str, comparators: list[str]) -> list[dict]:
    """QLIKE of the headline against each comparator: mean losses, gain (%) and one-sided
    Diebold-Mariano p (Harvey correction, rectangular kernel) that the headline is better."""
    q_head = qlike(df["y"], df[headline])
    rows = []
    for c in comparators:
        q_c = qlike(df["y"], df[c])
        rows.append({
            "comparator": c, "n": len(df),
            "qlike_headline": float(q_head.mean()), "qlike_comparator": float(q_c.mean()),
            "gain_pct": float(100 * (1 - q_head.mean() / q_c.mean())),
            "dm_p": float(dm_hln(q_c, q_head, h)[1]),
        })  # fmt: skip
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path: Path, record: dict) -> str:
    """Write the record as JSON with its own SHA-256; never overwrite (raises FileExistsError)."""
    body = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        json.dump({"sha256": digest, "content": record}, f, indent=2, sort_keys=True)
    return digest
