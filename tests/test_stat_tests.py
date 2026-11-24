"""V3: Diebold-Mariano test with the Harvey-Leybourne-Newbold correction. See docs/tasks/V3_dm_test.md.


dm_hln(loss_a, loss_b, h) -> (statistic, p_value)
  loss_a, loss_b: per-origin losses of two forecasts (1-D arrays or Series of equal length)
  h:              forecast horizon in origins (sets how many autocovariances enter the variance)
  statistic > 0 means forecast b has the LOWER average loss. p_value is two-sided, from a t distribution with n - 1 d.f.
"""

import numpy as np
import pytest
from scipy import stats


def test_matches_the_hand_formula_for_h_equal_1():
    from oilfc.eval.stat_tests import dm_hln

    a = np.array([1.0, 2.0, 4.0, 3.0, 2.5, 3.5])
    b = np.array([0.5, 2.5, 2.0, 2.0, 2.0, 2.5])
    d = a - b
    n = len(d)
    var_d = np.mean((d - d.mean()) ** 2)  # divide by n, not n - 1
    stat = d.mean() / np.sqrt(var_d / n) * np.sqrt((n - 1) / n)
    p = 2 * (1 - stats.t.cdf(abs(stat), n - 1))
    got_stat, got_p = dm_hln(a, b, 1)
    assert got_stat == pytest.approx(stat)
    assert got_p == pytest.approx(p)


def test_swapping_the_forecasts_flips_the_sign_and_keeps_the_p_value():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(1)
    a, b = rng.normal(1.0, 1, 200) ** 2, rng.normal(1.1, 1, 200) ** 2
    s1, p1 = dm_hln(a, b, 5)
    s2, p2 = dm_hln(b, a, 5)
    assert s1 == pytest.approx(-s2)
    assert p1 == pytest.approx(p2)


def test_a_clearly_better_forecast_is_detected():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(2)
    a = rng.normal(2.0, 1.0, 600)
    b = a - 0.5 + rng.normal(0, 0.5, 600)  # b is lower by 0.5 on average
    stat, p = dm_hln(a, b, 1)
    assert stat > 0
    assert p < 0.01


def test_two_equally_good_forecasts_are_not_distinguished():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(3)
    a, b = rng.normal(1.0, 1, 500) ** 2, rng.normal(1.0, 1, 500) ** 2
    assert dm_hln(a, b, 1)[1] > 0.05


def test_autocorrelated_differences_need_a_larger_h_to_stay_honest():
    from oilfc.eval.stat_tests import dm_hln

    # differences that overlap like h-step forecast errors (moving average of order 4)
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 800)
    d = np.convolve(e, np.ones(5) / 5, mode="valid") + 0.05
    a, b = d + 1.0, np.full_like(d, 1.0)
    p1 = dm_hln(a, b, 1)[1]
    p5 = dm_hln(a, b, 5)[1]
    assert p1 < 0.01  # treating the differences as independent looks highly significant
    assert p5 > 0.05  # allowing for the overlap, the evidence is weak


def test_nan_pairs_are_dropped():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(5)
    a, b = rng.normal(2, 1, 100), rng.normal(1.8, 1, 100)
    a_nan, b_nan = a.copy(), b.copy()
    a_nan[[3, 50]] = np.nan
    b_nan[[10]] = np.nan
    keep = ~(np.isnan(a_nan) | np.isnan(b_nan))
    assert dm_hln(a_nan, b_nan, 3) == pytest.approx(dm_hln(a[keep], b[keep], 3))


def test_degenerate_and_invalid_inputs():
    from oilfc.eval.stat_tests import dm_hln

    x = np.array([1.0, 2.0, 3.0, 4.0])
    stat, p = dm_hln(x, x, 1)  # identical losses: variance is zero
    assert np.isnan(stat) and np.isnan(p)
    with pytest.raises(ValueError):
        dm_hln(x, x[:3], 1)
    with pytest.raises(ValueError):
        dm_hln(x, x + 1, 0)


def test_too_few_pairs_for_the_horizon_is_an_error():
    from oilfc.eval.stat_tests import dm_hln

    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="usable pairs"):
        dm_hln(x, x + 1, 5)


def test_bartlett_kernel_for_h_equal_1_is_the_plain_t_statistic():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(3)
    a = rng.normal(0.1, 1.0, 400)
    b = np.zeros(400)
    stat, p = dm_hln(a, b, 1, kernel="bartlett")
    d = a - b
    expected = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / len(d))
    assert stat == pytest.approx(expected)
    assert 0 < p < 1


def test_bartlett_stays_defined_where_the_rectangular_variance_goes_non_positive():
    from oilfc.eval.stat_tests import dm_hln

    t = np.arange(600)
    d = np.cos(2 * np.pi * t / 30) + 0.05 * np.random.default_rng(1).normal(size=600) + 0.01
    assert np.isnan(dm_hln(d, np.zeros(600), 22)[0])  # rectangular: variance <= 0
    stat, p = dm_hln(d, np.zeros(600), 22, kernel="bartlett")
    assert np.isfinite(stat) and 0 <= p <= 1


def test_bartlett_and_rectangular_agree_in_sign_and_scale_for_overlapping_differences():
    from oilfc.eval.stat_tests import dm_hln

    rng = np.random.default_rng(5)
    e = rng.normal(0, 1, 1500)
    d = np.convolve(e, np.ones(22) / 22, mode="valid") + 0.05  # overlap of 22 like a 22-day target
    rect = dm_hln(d, np.zeros(len(d)), 22)
    bart = dm_hln(d, np.zeros(len(d)), 22, kernel="bartlett")
    assert np.sign(rect[0]) == np.sign(bart[0])
    assert 0.5 < bart[0] / rect[0] < 2.0


def test_unknown_kernel_is_rejected():
    from oilfc.eval.stat_tests import dm_hln

    with pytest.raises(ValueError):
        dm_hln(np.arange(50.0), np.zeros(50), 1, kernel="parzen")


def test_bartlett_weights_match_a_hand_computation_for_h_equal_3():
    from oilfc.eval.stat_tests import dm_hln

    d = np.array([1.0, 2.0, 0.5, 3.0, 2.0, 1.5, 2.5, 0.0, 1.0, 2.0, 3.0, 1.0])
    n, h = len(d), 3
    dbar = d.mean()
    g = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (g[0] + 2 * ((2 / 3) * g[1] + (1 / 3) * g[2])) / n
    expected = dbar / np.sqrt(var)
    stat, p = dm_hln(d, np.zeros(n), h, kernel="bartlett")
    assert stat == pytest.approx(expected)
    assert p == pytest.approx(2 * (1 - stats.t.cdf(abs(expected), n - 1)))
