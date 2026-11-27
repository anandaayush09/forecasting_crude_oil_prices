"""Input series of the foundation-model live track (ADR 0010, protocol v2).

One series per horizon h: the annualised log realised volatility over the last h daily log returns.
Its value at day t uses the returns up to and including t, so it is known at origin t. The value h days
later is exactly the quantity the live track forecasts.
"""

import numpy as np
import pandas as pd

from oilfc.eval.targets import TRADING_DAYS


def ttm_input(r: pd.Series, h: int) -> pd.Series:
    """0.5 * log(252 * mean(r^2 over the last h returns)), indexed like `r`; NaN for the first h - 1 days.

    r: daily log returns. Units: log of annualised volatility (decimal). A window of zero returns gives
    -inf and is left for the caller to reject.
    """
    if h < 1:
        raise ValueError("h must be at least 1")
    var = (r**2).rolling(h).mean()
    with np.errstate(divide="ignore"):
        return 0.5 * np.log(TRADING_DAYS * var)
