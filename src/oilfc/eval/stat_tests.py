"""Forecast-comparison tests."""

import numpy as np
from scipy import stats


def dm_hln(loss_a, loss_b, h: int) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    d = loss_a - loss_b. The statistic is positive when forecast b has the lower average loss.
    The variance of mean(d) uses the first h autocovariances (h-step forecasts overlap, so the
    differences are autocorrelated). The p-value is two-sided from a t distribution with n-1 d.f.
    Pairs with a NaN in either loss are dropped. Returns (nan, nan) if d has zero variance.
    """
    a, b = np.asarray(loss_a, dtype=float), np.asarray(loss_b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("loss_a and loss_b must be 1-D arrays of equal length")
    if h < 1:
        raise ValueError("h must be at least 1")

    keep = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[keep]
    n = len(d)
    if n <= 3 * h:
        raise ValueError(f"need more than 3 * h = {3 * h} usable pairs, got {n}")
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[: n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n  # long-run variance of the mean
    if not var > 1e-18:
        return float("nan"), float("nan")

    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dbar / np.sqrt(var) * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), n - 1))
    return float(stat), float(p)
