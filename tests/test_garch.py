import numpy as np
import pandas as pd
import pytest
from arch.univariate import GARCH, ConstantMean, StudentsT

from oilfc.eval.rolling import expanding_splits
from oilfc.models.baselines.garch import garch_forecast


def _simulated(n=2500, seed=0):
    model = ConstantMean(None, volatility=GARCH(1, 0, 1), distribution=StudentsT(seed=seed))
    sim = model.simulate([0.02, 0.05, 0.12, 0.84, 6.0], n, burn=500)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(sim["data"].to_numpy() / 100, index=idx), pd.Series(
        sim["volatility"].to_numpy() / 100, index=idx
    )


def _splits(r, h=5):
    return expanding_splits(r.index, r.index[1800], h, 3)[:3]


def test_forecasts_are_positive_finite_and_only_in_test_blocks():
    r, _ = _simulated()
    splits = _splits(r)
    fc = garch_forecast(r, splits, h=5)
    test_pos = np.concatenate([t for _, t in splits])
    inside = fc.iloc[test_pos]
    assert inside.notna().all().all()
    assert (inside[["sigma1", "sigma_h", "rv_h", "nu"]] > 0).all().all()
    outside = np.setdiff1d(np.arange(len(r)), test_pos)
    assert fc.iloc[outside].isna().all().all()


def test_one_day_sigma_tracks_the_true_conditional_volatility():
    r, true_sigma = _simulated()
    fc = garch_forecast(r, _splits(r), h=5).dropna()
    # true_sigma[t] is the volatility of the return on day t; sigma1 at origin t forecasts day t+1
    truth = true_sigma.shift(-1).reindex(fc.index)
    assert np.corrcoef(fc["sigma1"], truth)[0, 1] > 0.85


def test_forecast_at_an_origin_ignores_later_returns():
    r, _ = _simulated()
    splits = _splits(r)
    base = garch_forecast(r, splits, h=5)
    t = splits[0][1][10]
    perturbed = r.copy()
    perturbed.iloc[t + 1 :] = 0.2
    again = garch_forecast(perturbed, splits[:1], h=5)
    pd.testing.assert_series_equal(again.iloc[t], base.iloc[t], check_names=False)


def test_parameters_come_from_the_training_rows_only():
    r, _ = _simulated()
    splits = _splits(r)
    base = garch_forecast(r, splits[:1], h=5)
    train, test = splits[0]
    changed = r.copy()
    changed.iloc[test[0] + 1 :] *= 3  # only data after the first origin
    again = garch_forecast(changed, splits[:1], h=5)
    assert again["nu"].iloc[test[0]] == pytest.approx(base["nu"].iloc[test[0]])
    assert again["mu1"].iloc[test[0]] == pytest.approx(base["mu1"].iloc[test[0]])


def test_gjr_runs_and_h_day_variance_adds_up():
    r, _ = _simulated()
    fc = garch_forecast(r, _splits(r), h=10, kind="gjr").dropna()
    assert (fc["sigma_h"] > fc["sigma1"]).all()  # ten days carry more variance than one
    # rv_h is the annualised mean variance path; sigma_h**2 is its sum
    assert np.allclose(fc["rv_h"] ** 2 / 252 * 10, fc["sigma_h"] ** 2)


def test_fhs_one_day_var_matches_the_parametric_t_var():
    from oilfc.eval import risk
    from oilfc.models.baselines.garch import garch_fhs_var_es

    r, _ = _simulated(n=3000)
    splits = expanding_splits(r.index, r.index[2200], 5, 3)[:2]
    fhs = garch_fhs_var_es(r, splits, horizons=(1, 5), taus=(0.05,), simulations=4000).dropna()
    par = garch_forecast(r, splits, h=5).reindex(fhs.index)
    var, es = risk.t_var_es(par["mu1"], par["sigma1"], par["nu"], 0.05)
    assert np.corrcoef(fhs["var_h1_0.05"], var)[0, 1] > 0.95
    assert np.allclose(fhs["var_h1_0.05"], var, rtol=0.15)
    assert np.allclose(fhs["es_h1_0.05"], es, rtol=0.20)


def test_fhs_multi_day_tails_are_ordered_and_wider_than_one_day():
    from oilfc.models.baselines.garch import garch_fhs_var_es

    r, _ = _simulated(n=3000)
    splits = expanding_splits(r.index, r.index[2200], 10, 3)[:2]
    fhs = garch_fhs_var_es(r, splits, horizons=(1, 10), taus=(0.01, 0.05), kind="gjr").dropna()
    assert (fhs["es_h10_0.01"] < fhs["var_h10_0.01"]).all()
    assert (fhs["var_h10_0.01"] < fhs["var_h10_0.05"]).all()
    assert (fhs["var_h10_0.05"] < fhs["var_h1_0.05"]).all()  # ten days can lose more than one


def test_fhs_is_reproducible_and_ignores_returns_after_the_origin():
    from oilfc.models.baselines.garch import garch_fhs_var_es

    r, _ = _simulated(n=3000)
    splits = expanding_splits(r.index, r.index[2200], 5, 3)[:1]
    base = garch_fhs_var_es(r, splits, horizons=(5,), taus=(0.05,), simulations=500, seed=1)
    again = garch_fhs_var_es(r, splits, horizons=(5,), taus=(0.05,), simulations=500, seed=1)
    pd.testing.assert_frame_equal(base, again)
    t = splits[0][1][0]
    changed = r.copy()
    changed.iloc[t + 1 :] = 0.2
    other = garch_fhs_var_es(changed, splits, horizons=(5,), taus=(0.05,), simulations=500, seed=1)
    assert other["var_h5_0.05"].iloc[t] == base["var_h5_0.05"].iloc[t]
