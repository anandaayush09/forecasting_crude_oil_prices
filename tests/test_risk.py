import numpy as np
import pytest
from scipy import integrate, stats

from oilfc.eval import risk


def test_kupiec_is_zero_when_the_hit_rate_equals_tau():
    hits = np.zeros(100, dtype=int)
    hits[:5] = 1
    lr, p = risk.kupiec_pof(hits, 0.05)
    assert lr == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0)


def test_kupiec_hand_value_for_no_hits():
    n, tau = 250, 0.01
    lr, p = risk.kupiec_pof(np.zeros(n, dtype=int), tau)
    assert lr == pytest.approx(-2 * n * np.log(1 - tau))
    assert p == pytest.approx(stats.chi2.sf(lr, 1))


def test_kupiec_rejects_too_many_hits():
    hits = np.zeros(500, dtype=int)
    hits[:25] = 1  # 5% hits against a 1% VaR
    assert risk.kupiec_pof(hits, 0.01)[1] < 0.001


def test_christoffersen_accepts_independent_hits_and_rejects_clusters():
    rng = np.random.default_rng(0)
    iid = (rng.random(5000) < 0.05).astype(int)
    assert risk.christoffersen_ind(iid)[1] > 0.05
    clustered = np.zeros(5000, dtype=int)
    for start in range(100, 5000, 250):
        clustered[start : start + 5] = 1  # same hit rate, but in bursts
    assert risk.christoffersen_ind(clustered)[1] < 0.001


def test_conditional_coverage_combines_both_tests():
    rng = np.random.default_rng(1)
    hits = (rng.random(2000) < 0.02).astype(int)
    lr_cc, p_cc = risk.christoffersen_cc(hits, 0.05)  # wrong tau on purpose
    assert lr_cc == pytest.approx(risk.kupiec_pof(hits, 0.05)[0] + risk.christoffersen_ind(hits)[0])
    assert p_cc == pytest.approx(stats.chi2.sf(lr_cc, 2))


def test_t_var_es_matches_simulation_and_numerical_integration():
    mu, sigma, nu, tau = 0.001, 0.02, 5.0, 0.05
    var, es = risk.t_var_es(mu, sigma, nu, tau)
    scale = sigma * np.sqrt((nu - 2) / nu)
    rng = np.random.default_rng(2)
    x = mu + scale * rng.standard_t(nu, 2_000_000)
    assert x.std() == pytest.approx(sigma, rel=0.02)
    assert np.quantile(x, tau) == pytest.approx(var, rel=0.02)
    assert x[x <= var].mean() == pytest.approx(es, rel=0.02)
    q = (var - mu) / scale
    numeric = integrate.quad(lambda z: (mu + scale * z) * stats.t.pdf(z, nu), -np.inf, q)[0] / tau
    assert es == pytest.approx(numeric, rel=1e-6)
    assert es < var < mu


def test_normal_var_es_known_values():
    var, es = risk.normal_var_es(0.0, 1.0, 0.05)
    assert var == pytest.approx(-1.6448536)
    assert es == pytest.approx(-2.0627128)


def test_var_hits_counts_strictly_below_the_forecast():
    assert risk.var_hits([-0.03, -0.02, 0.01], [-0.02, -0.02, -0.02]).tolist() == [1, 0, 0]


def test_es_exceedance_test_detects_understated_shortfall():
    rng = np.random.default_rng(3)
    n, tau = 20000, 0.05
    y = rng.standard_normal(n)
    var, es = risk.normal_var_es(np.zeros(n), np.ones(n), tau)
    sigma = np.ones(n)
    assert risk.es_exceedance_test(y, var, es, sigma)[1] > 0.01
    assert risk.es_exceedance_test(y, var, es * 0.6, sigma)[1] < 0.001  # ES too small in magnitude
    assert np.isnan(risk.es_exceedance_test(y[:30], var[:30], es[:30], sigma[:30])[0])


def test_pinball_hand_values():
    assert risk.pinball([1.0], [2.0], 0.1)[0] == pytest.approx(0.9)  # (0.1 - 1) * (1 - 2)
    assert risk.pinball([3.0], [2.0], 0.1)[0] == pytest.approx(0.1)  # 0.1 * 1


def test_fz0_is_minimised_at_the_true_var_and_es():
    rng = np.random.default_rng(4)
    tau = 0.05
    y = rng.standard_normal(400_000)
    v, e = risk.normal_var_es(0.0, 1.0, tau)
    best = risk.fz0(y, v, e, tau).mean()
    for k in (0.7, 0.85, 1.2, 1.5):
        assert risk.fz0(y, k * v, k * e, tau).mean() > best  # both scaled
        assert risk.fz0(y, v, k * e, tau).mean() > best - 1e-4  # ES wrong only
    assert risk.fz0(y[:5], v, e, tau).shape == (5,)
