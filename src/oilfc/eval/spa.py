"""Hansen's test of superior predictive ability (SPA) via the `arch` package."""

import numpy as np
import pandas as pd
from arch.bootstrap import SPA


def spa_pvalue(
    benchmark_losses, model_losses: pd.DataFrame, block_size: int, reps: int = 1000, seed: int = 0
) -> float:
    """p-value (consistent version) of the null that no model has a lower expected loss than the benchmark.

    benchmark_losses: loss per origin of the benchmark. model_losses: one column per competing model,
    same rows. block_size: bootstrap block length, at least the forecast horizon (overlapping targets).
    A small p-value means at least one model beats the benchmark; it does not say which. Models whose
    losses equal the benchmark exactly are left out, and if all are, the p-value is 1.
    """
    bench = np.asarray(benchmark_losses, dtype=float)
    models = np.asarray(model_losses, dtype=float)
    differs = [j for j in range(models.shape[1]) if not np.array_equal(models[:, j], bench)]
    if not differs:  # every model equals the benchmark (alpha = 0 everywhere): nothing to test
        return 1.0
    spa = SPA(
        bench,
        models[:, differs],
        block_size=block_size,
        reps=reps,
        seed=seed,
    )
    spa.compute()
    return float(spa.pvalues["consistent"])
