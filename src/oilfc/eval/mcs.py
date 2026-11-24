"""Model Confidence Set (Hansen, Lunde and Nason 2011) via the `arch` package."""

import numpy as np
import pandas as pd
from arch.bootstrap import MCS


def model_confidence_set(
    losses: pd.DataFrame, size: float = 0.10, block_size: int = 22, reps: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Models that cannot be separated from the best one at level `size`.

    losses: one column per model, one row per forecast origin (all the same rows, no NaN).
    block_size: block length of the bootstrap; use at least the forecast horizon (overlapping errors).
    Models with exactly identical losses (for example a residual model whose correction was shrunk
    to zero, so it equals its anchor) cannot be compared by the bootstrap (zero variance of the loss
    difference), so one of each identical group is tested and the others share its result.
    Returns a table indexed by model with the MCS p-value and an `in_set` flag.
    """
    first = {}  # column -> the first column with identical losses
    for col in losses.columns:
        first[col] = next(c for c in losses.columns if np.array_equal(losses[c], losses[col]))
    unique = list(dict.fromkeys(first.values()))
    if len(unique) == 1:
        pvals = pd.Series(1.0, index=losses.columns)
        in_set = pd.Series(True, index=losses.columns)
    else:
        mcs = MCS(losses[unique], size=size, reps=reps, block_size=block_size, seed=seed)
        mcs.compute()
        p_unique = mcs.pvalues["Pvalue"].reindex(unique)
        in_unique = pd.Series(unique, index=unique).isin(mcs.included)
        pvals = pd.Series({c: p_unique[first[c]] for c in losses.columns})
        in_set = pd.Series({c: bool(in_unique[first[c]]) for c in losses.columns})
    return pd.DataFrame({"mcs_pvalue": pvals, "in_set": in_set}).astype(
        {"mcs_pvalue": float, "in_set": bool}
    )
