"""Value-at-Risk and Expected Shortfall: forecasts from a distribution, backtests and a joint loss.

Convention: returns are in decimal; VaR and ES are *quantiles of the return* (negative numbers for
losses). A "hit" is a return below the VaR forecast. tau is the tail probability (0.01, 0.05).
"""

import numpy as np
from scipy import stats


def t_var_es(mu, sigma, nu, tau):
    """VaR and ES (return quantiles) of a Student-t with mean mu, standard deviation sigma and nu d.f.

    arch's Student-t is standardised to unit variance, so a standard t is rescaled by sqrt((nu-2)/nu).
    """
    mu, sigma, nu = (np.asarray(v, dtype=float) for v in (mu, sigma, nu))
    q = stats.t.ppf(tau, nu)
    scale = sigma * np.sqrt((nu - 2) / nu)
    var = mu + scale * q
    es = mu - scale * stats.t.pdf(q, nu) / tau * (nu + q**2) / (nu - 1)
    return var, es


def normal_var_es(mu, sigma, tau):
    """VaR and ES (return quantiles) of a normal with mean mu and standard deviation sigma."""
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    z = stats.norm.ppf(tau)
    return mu + sigma * z, mu - sigma * stats.norm.pdf(z) / tau


def var_hits(returns, var):
    """1 where the realised return is below the VaR forecast, else 0 (NaN pairs are not allowed)."""
    return (np.asarray(returns, dtype=float) < np.asarray(var, dtype=float)).astype(int)


def _xlogy(x, y):
    return 0.0 if x == 0 else x * np.log(y)


def kupiec_pof(hits, tau):
    """Kupiec proportion-of-failures test. Returns (LR statistic, p-value), chi-square with 1 d.f.

    H0: the hit probability equals tau.
    """
    hits = np.asarray(hits, dtype=int)
    n, x = len(hits), int(hits.sum())
    phat = x / n
    ll0 = _xlogy(n - x, 1 - tau) + _xlogy(x, tau)
    ll1 = _xlogy(n - x, 1 - phat) + _xlogy(x, phat)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits):
    """Christoffersen independence test: are hits clustered? Returns (LR, p-value), chi-square 1 d.f."""
    h = np.asarray(hits, dtype=int)
    prev, cur = h[:-1], h[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    ll0 = _xlogy(n00 + n10, 1 - pi) + _xlogy(n01 + n11, pi)
    ll1 = _xlogy(n00, 1 - pi01) + _xlogy(n01, pi01) + _xlogy(n10, 1 - pi11) + _xlogy(n11, pi11)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_cc(hits, tau):
    """Conditional-coverage test: coverage plus independence. Returns (LR, p-value), chi-square 2 d.f."""
    lr = kupiec_pof(hits, tau)[0] + christoffersen_ind(hits)[0]
    return float(lr), float(stats.chi2.sf(lr, 2))


def es_exceedance_test(returns, var, es, sigma, min_hits=5):
    """McNeil-Frey test of Expected Shortfall.

    On hit days the standardised residuals (return - ES) / sigma should average zero if ES is right.
    A negative mean means losses beyond VaR are bigger than ES says. Two-sided one-sample t-test.
    Returns (t statistic, p-value), or (nan, nan) with fewer than `min_hits` hits.
    """
    y, v, e, s = (np.asarray(a, dtype=float) for a in (returns, var, es, sigma))
    hit = y < v
    if hit.sum() < min_hits:
        return float("nan"), float("nan")
    res = (y[hit] - e[hit]) / s[hit]
    t = stats.ttest_1samp(res, 0.0)
    return float(t.statistic), float(t.pvalue)


def pinball(y, q, tau):
    """Pinball (quantile) loss of the quantile forecast q at level tau, element by element."""
    y, q = np.asarray(y, dtype=float), np.asarray(q, dtype=float)
    return (tau - (y < q)) * (y - q)


def fz0(y, var, es, tau):
    """FZ0 joint loss for (VaR, ES) (Patton, Ziegel and Chen 2019), element by element; lower is better.

    Requires es < 0 and es <= var < 0 (return quantiles in the lower tail).
    """
    y, v, e = (np.asarray(a, dtype=float) for a in (y, var, es))
    hit = y <= v
    return -(1 / (tau * e)) * hit * (v - y) + v / e + np.log(-e) - 1
