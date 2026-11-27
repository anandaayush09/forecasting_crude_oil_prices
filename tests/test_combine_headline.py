"""Equal-weight combinations of the headline with other models (ADR 0008)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from vol_combine import COMBINATIONS, combine_log  # noqa: E402


def test_the_combination_of_identical_forecasts_is_that_forecast():
    f = pd.Series([0.2, 0.35, 0.8], index=pd.date_range("2020-01-01", periods=3))
    out = combine_log(pd.DataFrame({"a": f, "b": f, "c": f}))
    assert np.allclose(out, f) and out.index.equals(f.index)


def test_the_combination_is_the_geometric_mean_and_keeps_the_origins():
    idx = pd.date_range("2020-01-01", periods=2)
    parts = pd.DataFrame({"a": [0.2, 0.1], "b": [0.4, 0.4], "c": [0.4, 0.9]}, index=idx)
    assert combine_log(parts[["a", "b"]]).iloc[0] == pytest.approx(np.sqrt(0.08))
    assert combine_log(parts).iloc[1] == pytest.approx((0.1 * 0.4 * 0.9) ** (1 / 3))
    assert combine_log(parts).index.equals(idx)


def test_every_combination_contains_the_headline_and_has_two_or_three_parts():
    assert len(COMBINATIONS) == 3
    for parts in COMBINATIONS.values():
        assert parts[0] == "lhar_ovx_bc_lag1" and 2 <= len(parts) <= 3
