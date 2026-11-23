"""Guard for the clean holdout: development work must never see data from `clean_holdout_start` on.

The date comes from configs/protocol.yaml. Development experiments call `development_only` on the
returns they load; the final evaluation (one run, after the protocol freeze) is a separate script.
"""

import pandas as pd


class HoldoutError(RuntimeError):
    """Raised when development data reaches into the clean holdout."""


def development_only(series: pd.Series | pd.DataFrame, holdout_start) -> pd.Series | pd.DataFrame:
    """Return only the rows dated before `holdout_start`."""
    return series[series.index < pd.Timestamp(holdout_start)]


def assert_development_only(index: pd.DatetimeIndex, holdout_start) -> None:
    """Raise HoldoutError if any date is on or after `holdout_start`."""
    late = index[index >= pd.Timestamp(holdout_start)]
    if len(late):
        raise HoldoutError(
            f"{len(late)} rows on or after the holdout start {holdout_start} (first: {late[0].date()})"
        )
