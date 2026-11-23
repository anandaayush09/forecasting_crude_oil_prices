"""V1: volatility targets. See docs/tasks/V1_volatility_targets.md.

Implemented: src/oilfc/eval/targets.py.
"""

import numpy as np
import pandas as pd
import pytest

ANN = np.sqrt(252)


def _returns(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(0, 0.02, size=n), index=idx)


def test_constant_returns_give_the_known_annualised_vol():
    from oilfc.eval.targets import forward_realized_vol, past_realized_vol

    r = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=60))
    expected = 0.01 * ANN
    assert np.allclose(forward_realized_vol(r, 5).iloc[:-5], expected)
    assert np.allclose(past_realized_vol(r, 5).iloc[4:], expected)


def test_forward_vol_matches_a_hand_computed_value():
    from oilfc.eval.targets import forward_realized_vol

    r = pd.Series([0.0, 0.0, 0.02, 0.04, 0.0, 0.0], index=pd.bdate_range("2020-01-01", periods=6))
    # origin index 1: the window is the next two returns, 0.02 and 0.04
    expected = np.sqrt(252 * (0.02**2 + 0.04**2) / 2)
    assert forward_realized_vol(r, 2).iloc[1] == pytest.approx(expected)


def test_last_h_forward_values_are_missing_and_index_is_kept():
    from oilfc.eval.targets import forward_realized_vol

    r = _returns()
    out = forward_realized_vol(r, 22)
    assert out.index.equals(r.index)
    assert out.iloc[-22:].isna().all()
    assert out.iloc[:-22].notna().all()


def test_forward_vol_at_t_uses_only_the_next_h_returns():
    from oilfc.eval.targets import forward_realized_vol

    r, h, t = _returns(), 22, 100
    base = forward_realized_vol(r, h).iloc[t]

    past = r.copy()
    past.iloc[: t + 1] = 9.0  # today and everything before: must not matter
    assert forward_realized_vol(past, h).iloc[t] == base

    later = r.copy()
    later.iloc[t + h + 1 :] = 9.0  # beyond the window: must not matter
    assert forward_realized_vol(later, h).iloc[t] == base

    inside = r.copy()
    inside.iloc[t + 1] = 0.5  # inside the window: must matter
    assert forward_realized_vol(inside, h).iloc[t] != base


def test_past_vol_at_t_uses_only_the_last_h_returns_including_today():
    from oilfc.eval.targets import past_realized_vol

    r, h, t = _returns(), 22, 100
    out = past_realized_vol(r, h)
    assert out.iloc[: h - 1].isna().all()
    assert out.iloc[h - 1 :].notna().all()

    future = r.copy()
    future.iloc[t + 1 :] = 9.0
    assert past_realized_vol(future, h).iloc[t] == out.iloc[t]

    older = r.copy()
    older.iloc[: t - h + 1] = 9.0
    assert past_realized_vol(older, h).iloc[t] == out.iloc[t]

    today = r.copy()
    today.iloc[t] = 0.5
    assert past_realized_vol(today, h).iloc[t] != out.iloc[t]
