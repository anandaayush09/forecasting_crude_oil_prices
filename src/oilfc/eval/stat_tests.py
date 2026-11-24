"""Forecast-comparison tests."""

import numpy as np
from scipy import stats


def dm_hln(loss_a, loss_b, h: int, kernel: str = "rectangular") -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    d = loss_a - loss_b. The statistic is positive when forecast b has the lower average loss.
    The variance of mean(d) uses the first h autocovariances (h-step forecasts overlap, so the
    differences are autocorrelated). The p-value is two-sided from a t distribution with n-1 d.f.
    Pairs with a NaN in either loss are dropped. Returns (nan, nan) if d has zero variance.

    kernel: "rectangular" (default) weights the first h autocovariances equally and applies the HLN
    correction. "bartlett" uses Bartlett weights 1 - k / h (Newey-West), which keep the variance
    non-negative and are less noisy when h is large (for example 66 days); the HLN factor was derived for
    the rectangular kernel, so it is not applied, and the p-value still uses a t distribution with n-1 d.f.
    """
    if kernel not in ("rectangular", "bartlett"):
        raise ValueError("kernel must be 'rectangular' or 'bartlett'")
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
    weights = np.ones(h) if kernel == "rectangular" else 1 - np.arange(h) / h
    var = (gamma[0] + 2 * sum(w * g for w, g in zip(weights[1:], gamma[1:], strict=True))) / n
    if not var > 1e-18:
        return float("nan"), float("nan")

    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) if kernel == "rectangular" else 1.0
    stat = dbar / np.sqrt(var) * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), n - 1))
    return float(stat), float(p)
