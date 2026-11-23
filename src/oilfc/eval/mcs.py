"""Model Confidence Set (Hansen, Lunde and Nason 2011) via the `arch` package."""

import pandas as pd
from arch.bootstrap import MCS


def model_confidence_set(
    losses: pd.DataFrame, size: float = 0.10, block_size: int = 22, reps: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Models that cannot be separated from the best one at level `size`.

    losses: one column per model, one row per forecast origin (all the same rows, no NaN).
    block_size: block length of the bootstrap; use at least the forecast horizon (overlapping errors).
    Returns a table indexed by model with the MCS p-value and an `in_set` flag.
    """
    mcs = MCS(losses, size=size, reps=reps, block_size=block_size, seed=seed)
    mcs.compute()
    pvals = mcs.pvalues["Pvalue"].reindex(losses.columns)
    in_set = pd.Series(losses.columns.isin(mcs.included), index=losses.columns)
    return pd.DataFrame({"mcs_pvalue": pvals, "in_set": in_set}).astype(
        {"mcs_pvalue": float, "in_set": bool}
    )
