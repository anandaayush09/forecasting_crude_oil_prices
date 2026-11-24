"""V2: volatility and density losses. See docs/tasks/V2_losses.md.

All three functions work element by element and return an array/Series of the same length as the input.
"""

import numpy as np
import pytest
from scipy import stats


def test_qlike_is_zero_for_a_perfect_forecast_and_positive_otherwise():
    from oilfc.eval.losses import qlike

    x = np.array([0.1, 0.3, 0.8])
    assert np.allclose(qlike(x, x), 0.0)
    assert (qlike(x, x * 1.5) > 0).all()
    assert (qlike(x, x / 1.5) > 0).all()


def test_qlike_hand_values():
    from oilfc.eval.losses import qlike

    # realised 2, forecast 1: z = (2/1)^2 = 4, loss = z - ln z - 1
    assert qlike(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(3 - np.log(4))
    # realised 1, forecast 2: z = 1/4
    assert qlike(np.array([1.0]), np.array([2.0]))[0] == pytest.approx(0.25 + np.log(4) - 1)


def test_qlike_penalises_under_forecasts_more_than_equal_over_forecasts():
    from oilfc.eval.losses import qlike

    r = np.array([1.0])
    assert qlike(r, r / 2)[0] > qlike(r, r * 2)[0]


def test_qlike_is_minimised_at_the_realised_value():
    from oilfc.eval.losses import qlike

    f = np.linspace(0.2, 3.0, 400)
    losses = qlike(np.full_like(f, 1.0), f)
    assert abs(f[np.argmin(losses)] - 1.0) < 0.01


def test_mse_logvol_hand_values_and_scale_invariance():
    from oilfc.eval.losses import mse_logvol

    assert mse_logvol(np.array([np.e]), np.array([1.0]))[0] == pytest.approx(1.0)
    a = mse_logvol(np.array([2.0, 5.0]), np.array([1.0, 4.0]))
    b = mse_logvol(np.array([20.0, 50.0]), np.array([10.0, 40.0]))
    assert np.allclose(a, b)


def test_crps_normal_matches_the_closed_form():
    from oilfc.eval.losses import crps_normal

    # at x = mu: sigma * (2*pdf(0) - 1/sqrt(pi))
    expected = 2 * stats.norm.pdf(0) - 1 / np.sqrt(np.pi)
    assert crps_normal(np.array([0.0]), np.array([0.0]), np.array([1.0]))[0] == pytest.approx(
        expected
    )
    assert crps_normal(np.array([3.0]), np.array([3.0]), np.array([2.5]))[0] == pytest.approx(
        2.5 * expected
    )


def test_crps_normal_grows_with_the_miss_and_matches_a_numeric_integral():
    from oilfc.eval.losses import crps_normal

    mu, sigma, x = 0.5, 1.2, 2.0
    grid = np.linspace(-12, 14, 200001)
    cdf = stats.norm.cdf(grid, mu, sigma)
    numeric = np.trapezoid((cdf - (grid >= x)) ** 2, grid)
    assert crps_normal(np.array([x]), np.array([mu]), np.array([sigma]))[0] == pytest.approx(
        numeric, abs=1e-4
    )
    near = crps_normal(np.array([0.6]), np.array([0.5]), np.array([1.2]))[0]
    far = crps_normal(np.array([4.0]), np.array([0.5]), np.array([1.2]))[0]
    assert far > near
