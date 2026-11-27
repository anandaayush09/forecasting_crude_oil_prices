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
    if refit_days is None and refit_months < 1:
        raise ValueError(f"refit_months must be at least 1, got {refit_months}")
    if refit_days is not None and refit_days < 1:
        raise ValueError(f"refit_days must be at least 1, got {refit_days}")
    if window is not None and window < 1:
        raise ValueError(f"window must be at least 1, got {window}")
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


def assert_targets_before(
    index: pd.DatetimeIndex, train: np.ndarray, h: int, first_test: pd.Timestamp
) -> None:
    """Fail unless every training target is realised before `first_test`.

    A training origin at position i has a target that ends at the return dated `index[i + h]`.
    Used for tuning, which must only see data from before the first test origin.
    """
    if len(train) and index[int(train[-1]) + h] >= first_test:
        raise AssertionError(
            f"a training target ends on {index[int(train[-1]) + h].date()}, on or after {first_test.date()}"
        )
