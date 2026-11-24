"""Multiple-testing correction for families of forecast comparisons."""

import numpy as np


def holm_adjust(p_values) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate; no independence needed).

    NaN entries are returned as NaN and do not count towards the number of tests. Input and output
    are in the same order. An adjusted p-value below 0.05 means the comparison survives correction
    for the whole family.
    """
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    adj = np.minimum(1.0, (m - np.arange(m)) * p[ok][order])
    adj = np.maximum.accumulate(adj)
    values = np.empty(m)
    values[order] = adj
    out[ok] = values
    return out
