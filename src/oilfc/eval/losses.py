"""Losses for volatility and density forecasts. All work element by element (arrays or Series).

Volatility inputs are volatilities (not variances) and must be positive.
"""

import numpy as np
from scipy import stats


def qlike(realized, forecast):
    """QLIKE: z - log(z) - 1 with z = (realized / forecast)**2 (the variance ratio).

    Zero only for a perfect forecast, and it punishes under-forecasts more than equal over-forecasts.
    It stays valid when the realised-volatility proxy is noisy (Patton 2011).
    """
    z = (np.asarray(realized, dtype=float) / np.asarray(forecast, dtype=float)) ** 2
    return z - np.log(z) - 1


def mse_logvol(realized, forecast):
    """Squared error on the log of volatility; invariant to the scale of the volatility."""
    return (
        np.log(np.asarray(realized, dtype=float)) - np.log(np.asarray(forecast, dtype=float))
    ) ** 2


def crps_normal(x, mu, sigma):
    """Closed-form CRPS of a N(mu, sigma**2) forecast at the observation x (Gneiting and Raftery 2007)."""
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
