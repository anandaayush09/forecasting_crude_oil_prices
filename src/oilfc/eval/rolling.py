"""Expanding-origin splits with a gap between the training targets and the test block."""

import numpy as np
import pandas as pd


def expanding_splits(
    index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    h: int,
    refit_months: int,
    window: int | None = None,
    refit_days: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_positions, test_positions) pairs of integer positions into `index`.

    Test block k holds the origins dated in [test_start + k*m months, test_start + (k+1)*m months).
    The training origins are the prefix of `index` whose target is known before the block starts:
    origin i has a target that reaches row i + h, so we keep i + h < first_test_position.
    Blocks that contain no origin are skipped.
    window: if given, keep only the last `window` training origins (a rolling window of that many
    rows, ending at the same last usable origin); None keeps every earlier origin (expanding window).
    refit_days: if given, test blocks are this many calendar days long (refit_months is then ignored).
    """
    splits = []
    k = 0
    while True:
        if refit_days is None:
            lo = test_start + pd.DateOffset(months=refit_months * k)
            hi = test_start + pd.DateOffset(months=refit_months * (k + 1))
        else:
            lo = test_start + pd.DateOffset(days=refit_days * k)
            hi = test_start + pd.DateOffset(days=refit_days * (k + 1))
        if lo > index[-1]:
            return splits
        test = np.flatnonzero((index >= lo) & (index < hi))
        if len(test) > 0:
            end = test[0] - h
            splits.append((np.arange(0 if window is None else max(0, end - window), end), test))
        k += 1
