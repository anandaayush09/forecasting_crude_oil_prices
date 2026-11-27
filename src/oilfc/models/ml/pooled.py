"""Residual forecasting with one net shared by several series (Stage B of ADR 0004).

Each series keeps its own linear anchor (OLS of log volatility on its own HAR columns). One net is
trained on the anchor residuals of all series together and corrects the anchor of the evaluated
series. The rules of `residual.residual_forecast` hold for that series: alpha is chosen on the last
fraction of its training rows by squared error, and the net and alpha are refit every k-th usable
split and reused in between.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from oilfc.models.ml.residual import ALPHAS, MIN_TRAIN_ROWS, _ols, _predict_ols

MIN_COMPANION_ROWS = 100  # a companion with fewer complete usable rows is left out of that fit


def _usable(table: pd.DataFrame, cols: list[str], before: pd.Timestamp) -> pd.DataFrame:
    """Complete rows of `table` whose target ended before `before`: the only rows a test block may use.

    A row with an infinite value is incomplete: a log volatility is undefined over a window with no
    price change at all, which happens in the thinly traded companion series.
    """
    rows = table[[*cols, "y", "end"]].replace([np.inf, -np.inf], np.nan).dropna()
    return rows[rows["end"] < before]


def _fit_pooled(rows, anchors, feature_cols, target, model_factory):
    """Per-series OLS anchors, then one net on all residuals ordered by date. Returns (net, anchors)."""
    betas, parts = {}, []
    for series, data in rows.items():
        if series != target and len(data) < MIN_COMPANION_ROWS:
            continue
        beta = _ols(data[anchors[series]].to_numpy(), data["y"].to_numpy())
        betas[series] = beta
        resid = data["y"].to_numpy() - _predict_ols(beta, data[anchors[series]].to_numpy())
        parts.append(data[feature_cols].assign(resid=resid))
    stacked = pd.concat(parts).sort_index(
        kind="stable"
    )  # the net's early stopping holds out the latest rows
    net = model_factory().fit(stacked[feature_cols].to_numpy(), stacked["resid"].to_numpy())
    return net, betas


def pooled_residual_forecast(
    tables: dict[str, pd.DataFrame],
    anchors: dict[str, list[str]],
    feature_cols: list[str],
    target: str,
    splits,
    model_factory: Callable,
    val_fraction: float = 0.2,
    residual_refit_every: int = 1,
) -> pd.DataFrame:
    """Learned residual and alpha for every test origin of `target`.

    tables: per series, indexed by date, with the anchor columns, `feature_cols`, `y` (log realised
    volatility) and `end` (date of the last return of the target), as built by `features.pooled`.
    splits: (train, test) positions into tables[target].index; only the test positions are used. Which
    rows train is decided by date: a row of any series is usable by a block only if its target ended
    before the block's first origin, which is the rule `expanding_splits` applies to the target series.
    For each usable split (the target has at least MIN_TRAIN_ROWS complete usable rows):
      1. On a refit, split the target's usable rows in time: the last val_fraction validate. Fit every
         series' anchor and the shared net on all usable rows of all series dated before that block.
      2. Choose alpha on the target's validation rows by squared error of anchor + alpha * residual.
      3. Refit the anchors and the net on all usable rows, and predict the test rows.
    A series other than the target joins a fit only with at least MIN_COMPANION_ROWS usable rows.
    Returns a DataFrame indexed like tables[target] with columns residual and alpha, NaN outside the
    test blocks. The final forecast is the target's own anchor forecast plus alpha times the residual.
    """
    if residual_refit_every < 1:
        raise ValueError("residual_refit_every must be at least 1")
    X = tables[target]
    out = pd.DataFrame(np.nan, index=X.index, columns=["residual", "alpha"])
    cache = None  # (net, alpha) from the last refit
    usable = 0
    for _, test in splits:
        first_test = X.index[test[0]]
        rows = {s: _usable(t, [*anchors[s], *feature_cols], first_test) for s, t in tables.items()}
        data = rows[target]
        if len(data) < MIN_TRAIN_ROWS:
            continue
        if usable % residual_refit_every == 0:
            n_val = max(int(len(data) * val_fraction), 1)
            cutoff = data.index[-n_val]
            fit_rows = {s: d[d.index < cutoff] for s, d in rows.items()}
            net, betas = _fit_pooled(fit_rows, anchors, feature_cols, target, model_factory)
            val = data.iloc[-n_val:]
            anchor_val = _predict_ols(betas[target], val[anchors[target]].to_numpy())
            corr_val = net.predict(val[feature_cols].to_numpy())
            errors = [
                np.mean((val["y"].to_numpy() - anchor_val - a * corr_val) ** 2) for a in ALPHAS
            ]
            alpha = float(ALPHAS[int(np.argmin(errors))])
            net, _ = _fit_pooled(rows, anchors, feature_cols, target, model_factory)
            cache = (net, alpha)
        usable += 1
        net, alpha = cache
        te = X.iloc[test][feature_cols]
        ok = np.isfinite(te.to_numpy()).all(axis=1)
        corr = np.full(len(te), np.nan)
        corr[ok] = net.predict(te.loc[ok].to_numpy())
        out.loc[out.index[test], "residual"] = corr
        out.loc[out.index[test], "alpha"] = alpha
    return out
