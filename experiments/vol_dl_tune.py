"""Hyperparameter search for the six architectures of ADR 0007, on data before the first test origin.

    python experiments/vol_dl_tune.py [--archs nhits,tide] [--trials 30] [--device auto]
    python experiments/vol_dl_tune.py --smoke      # 2 trials, 1 arch, 5 epochs

In each of the six cells only the training rows of the first monthly split are used (their targets end
before 2012-01-01, checked by `assert_targets_before`). The first 80% of those rows fit the anchor
(leverage HAR + OVX, OLS) and the net, h rows are skipped, and the rest validate. The net learns the
anchor residual with the bounded zero-start QLIKE loss of ADR 0004. The objective is the mean over the
six cells of validation loss with the net's correction divided by validation loss of the anchor alone
(below 1 means the net helps), minimised by Optuna's TPE sampler (seed 0). One set of hyperparameters per
architecture is shared by all cells. Writes results/dl_arch/hparams.json and tuning_trials.csv.
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

from vol_dl_addendum import ANCHOR, load  # noqa: E402

from oilfc.eval.holdout import assert_development_only, development_only  # noqa: E402
from oilfc.eval.rolling import assert_targets_before, expanding_splits  # noqa: E402
from oilfc.eval.targets import forward_realized_vol  # noqa: E402
from oilfc.features.vol import CHANNELS_EXT, dl_channels, lag_window  # noqa: E402
from oilfc.models.baselines.vol import leverage_design  # noqa: E402
from oilfc.models.dl.encoders import ARCHITECTURES, regressor_params  # noqa: E402

CELLS = [(a, h) for a in ("brent", "wti") for h in (5, 22, 66)]
WINDOWS = (22, 44, 66)
TEST_START = pd.Timestamp("2012-01-01")


def qlike_consistent(d: np.ndarray) -> float:
    """Mean of exp(2d) - 2d - 1, d = target minus forecast in log volatility."""
    return float(np.mean(np.exp(2 * d) - 2 * d - 1))


def tuning_data(asset: str, h: int, panel_path: str, holdout_start: str):
    """Anchor columns, 66-day window and target for the rows of the first training block only."""
    panel, r = load(asset, panel_path)
    r = development_only(r, holdout_start)
    assert_development_only(r.index, holdout_start)
    frame = dl_channels(panel, r).join(leverage_design(r))
    window = lag_window(frame, CHANNELS_EXT, max(WINDOWS))
    X = frame[ANCHOR].join(window)
    y = np.log(forward_realized_vol(r, h))
    train, _ = expanding_splits(r.index, TEST_START, h, 1)[0]
    assert_targets_before(r.index, train, h, TEST_START)
    data = X.iloc[train].join(y.iloc[train].rename("y"))
    return data, list(window.columns)


def cell_ratio(data, window_cols, h, hp, arch, seeds, epochs, device, SeqResidualRegressor):
    """Validation loss of anchor + net correction divided by that of the anchor, in one cell."""
    cols = window_cols[: hp["window"] * len(CHANNELS_EXT)]
    d = data[ANCHOR + cols + ["y"]].dropna()
    n_fit = int(0.8 * len(d))
    fit, val = d.iloc[:n_fit], d.iloc[n_fit + h :]
    design = lambda f: np.column_stack([np.ones(len(f)), f[ANCHOR].to_numpy()])  # noqa: E731
    beta = np.linalg.lstsq(design(fit), fit["y"].to_numpy(), rcond=None)[0]
    res_fit = fit["y"].to_numpy() - design(fit) @ beta
    res_val = val["y"].to_numpy() - design(val) @ beta
    net = SeqResidualRegressor(
        arch=arch, epochs=epochs, seeds=seeds, device=device,
        **regressor_params(hp, len(CHANNELS_EXT)),
    ).fit(fit[cols].to_numpy(), res_fit)  # fmt: skip
    return qlike_consistent(res_val - net.predict(val[cols].to_numpy())) / qlike_consistent(res_val)


def suggest(trial, arch: str) -> dict:
    hp = dict(
        window=trial.suggest_categorical("window", list(WINDOWS)),
        width=trial.suggest_categorical("width", [16, 32, 64]),
        depth=trial.suggest_int("depth", 1, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        lr=trial.suggest_float("lr", 3e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        bound=trial.suggest_categorical("bound", [0.25, 0.5]),
    )
    if arch in ("itransformer", "tft", "informer"):
        hp["heads"] = trial.suggest_categorical("heads", [2, 4])
    if arch == "tcn":
        hp["kernel"] = trial.suggest_categorical("kernel", [2, 3])
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default=",".join(ARCHITECTURES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--holdout-start", default="2025-07-01", help="never evaluated here")
    ap.add_argument("--panel", default=str(ROOT / "data" / "processed" / "panel_daily.parquet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "dl_arch"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    import optuna
    import torch

    from oilfc.models.dl.residual_seq import SeqResidualRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    archs = a.archs.split(",")[:1] if a.smoke else a.archs.split(",")
    unknown = set(archs) - set(ARCHITECTURES)
    assert not unknown, f"unknown architectures {unknown}"
    trials, epochs = (2, 5) if a.smoke else (a.trials, a.epochs)
    seeds = (0,) if a.smoke else (0, 1)
    cells = CELLS[:2] if a.smoke else CELLS
    print(
        f"device={device} archs={archs} trials={trials} epochs={epochs} seeds={seeds}", flush=True
    )

    data = {c: tuning_data(*c, a.panel, a.holdout_start) for c in cells}
    for c, (d, _) in data.items():
        print(f"{c}: {len(d.dropna(subset=ANCHOR + ['y']))} anchor rows before 2012-01", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if a.smoke else ""
    hp_path, trials_path = out / f"hparams{suffix}.json", out / f"tuning_trials{suffix}.csv"
    best = json.loads(hp_path.read_text()) if hp_path.exists() else {}
    all_trials = pd.read_csv(trials_path) if trials_path.exists() else None

    for arch in archs:
        t0 = time.time()

        def objective(trial, arch=arch):
            hp = suggest(trial, arch)
            ratios = [
                cell_ratio(d, w, c[1], hp, arch, seeds, epochs, device, SeqResidualRegressor)
                for c, (d, w) in data.items()
            ]
            trial.set_user_attr("ratios", [round(x, 4) for x in ratios])
            return float(np.mean(ratios))

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=0), direction="minimize"
        )
        study.optimize(objective, n_trials=trials)
        df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
        df.insert(0, "arch", arch)
        keep = [] if all_trials is None else [all_trials[all_trials["arch"] != arch]]
        all_trials = pd.concat([*keep, df], ignore_index=True)
        hp = {**study.best_params, "best_value": study.best_value, "trials": trials}
        best[arch] = hp
        hp_path.write_text(
            json.dumps(best, indent=2)
        )  # after each architecture, so a stop loses little
        all_trials.to_csv(trials_path, index=False)
        print(f"{arch}: best mean ratio {study.best_value:.4f} {study.best_params} "
              f"({time.time() - t0:.0f}s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
