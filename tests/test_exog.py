import numpy as np
import pandas as pd

from oilfc.features.exog import GROUPS, exogenous_features


def _panel(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    p = pd.DataFrame(
        {
            "wti_fut1": 60 + rng.normal(0, 1, n),
            "wti_fut4": 61 + rng.normal(0, 1, n),
            "wti_fut1_age": 0.0,
            "wti_fut4_age": 0.0,
            "crude_stocks": 400_000 + rng.normal(0, 1000, n),
            "gpr_daily": 100 + rng.uniform(0, 50, n),
            "epu_daily": 100 + rng.uniform(0, 50, n),
            "vix": 20 + rng.uniform(0, 5, n),
        },
        index=idx,
    )
    return p


def test_columns_match_the_groups_and_values_are_as_defined():
    p = _panel()
    f = exogenous_features(p, p.index)
    assert sorted(f.columns) == sorted(c for cols in GROUPS.values() for c in cols)
    t = p.index[300]
    assert np.isclose(f.loc[t, "fut_slope"], np.log(p.loc[t, "wti_fut4"] / p.loc[t, "wti_fut1"]))
    assert np.isclose(
        f.loc[t, "stocks_chg_4w"], np.log(p["crude_stocks"].iloc[300] / p["crude_stocks"].iloc[280])
    )
    assert np.isclose(f.loc[t, "log_gpr"], np.log(p["gpr_daily"].iloc[296:301].mean()))


def test_stale_futures_become_missing():
    p = _panel()
    p.iloc[350:, p.columns.get_loc("wti_fut1_age")] = (
        30.0  # publication stopped, value carried forward
    )
    f = exogenous_features(p, p.index)
    assert f["fut_slope"].iloc[350:].isna().all()
    assert f["fut_slope"].iloc[:350].notna().all()


def test_features_are_causal():
    p = _panel()
    base = exogenous_features(p, p.index)
    changed = p.copy()
    changed.iloc[301:] = changed.iloc[301:] * 3
    again = exogenous_features(changed, p.index)
    assert np.allclose(again.iloc[300], base.iloc[300], equal_nan=True)


def test_a_negative_front_price_gives_a_missing_slope_without_warnings():
    import warnings

    p = _panel()
    p.iloc[200, p.columns.get_loc("wti_fut1")] = -37.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = exogenous_features(p, p.index)
    assert np.isnan(f["fut_slope"].iloc[200]) and f["fut_slope"].drop(f.index[200]).notna().all()
