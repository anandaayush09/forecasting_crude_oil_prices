"""Expanding-origin splits with a gap between the training targets and the test block."""

import numpy as np
import pandas as pd


def expanding_splits(
    index: pd.DatetimeIndex, test_start: pd.Timestamp, h: int, refit_months: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_positions, test_positions) pairs of integer positions into `index`.

    Test block k holds the origins dated in [test_start + k*m months, test_start + (k+1)*m months).
    The training origins are the prefix of `index` whose target is known before the block starts:
    origin i has a target that reaches row i + h, so we keep i + h < first_test_position.
    Blocks that contain no origin are skipped.
    """
    splits = []
    k = 0
    while True:
        lo = test_start + pd.DateOffset(months=refit_months * k)
        hi = test_start + pd.DateOffset(months=refit_months * (k + 1))
        if lo > index[-1]:
            return splits
        test = np.flatnonzero((index >= lo) & (index < hi))
        if len(test) > 0:
            splits.append((np.arange(0, test[0] - h), test))
        k += 1
