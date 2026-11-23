import numpy as np
import pandas as pd
import pytest

from oilfc.eval.holdout import HoldoutError, assert_development_only, development_only


def _series():
    idx = pd.bdate_range("2025-06-25", "2025-07-08")
    return pd.Series(np.arange(len(idx), dtype=float), index=idx)


def test_development_only_keeps_rows_strictly_before_the_holdout_start():
    out = development_only(_series(), "2025-07-01")
    assert out.index.max() == pd.Timestamp("2025-06-30")
    assert len(out) == 4
    assert len(development_only(_series().to_frame("x"), "2025-07-01")) == 4


def test_assert_development_only_raises_on_any_holdout_row_and_passes_otherwise():
    s = _series()
    assert_development_only(development_only(s, "2025-07-01").index, "2025-07-01")
    with pytest.raises(HoldoutError, match="2025-07-01"):
        assert_development_only(s.index, "2025-07-01")
