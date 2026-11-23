"""GARCH-family forecasts with expanding refits (uses the `arch` package).

Returns are scaled to percent for fitting (arch's optimiser works better) and converted back, so
every output is in decimal (daily return units) or annualised decimal volatility.
"""

import numpy as np
import pandas as pd
from arch import arch_model

from oilfc.eval.targets import TRADING_DAYS

KINDS = {"garch": 0, "gjr": 1}  # the `o` (asymmetry) order


def garch_forecast(returns: pd.Series, splits, h: int, kind: str = "garch") -> pd.DataFrame:
    """Student-t GARCH(1,1) or GJR-GARCH(1,1) forecasts for every test origin.

    For each (train, test) split: fit on returns[train] only, then run the fitted model's filter
    forward (parameters fixed, no refitting) through the end of the test block and read off the
    analytic h-step variance forecasts at each origin. The forecast at origin t uses returns up to
    and including t.

    Columns (indexed like `returns`, NaN outside the test blocks):
      mu1, sigma1  mean and standard deviation of the next day's return (decimal)
      nu           t degrees of freedom
      mu_h, sigma_h  mean and standard deviation of the sum of the next h returns (decimal)
      rv_h         annualised volatility forecast over the next h days: sqrt(252 * mean variance path)
    """
    out = pd.DataFrame(
        np.nan, index=returns.index, columns=["mu1", "sigma1", "nu", "mu_h", "sigma_h", "rv_h"]
    )
    r_pct = returns * 100
    for train, test in splits:
        spec = dict(mean="Constant", vol="GARCH", p=1, o=KINDS[kind], q=1, dist="t")
        fit = arch_model(r_pct.iloc[train], **spec).fit(disp="off", update_freq=0)
        full = arch_model(r_pct.iloc[: test[-1] + 1], **spec).fix(fit.params)
        fc = full.forecast(horizon=h, start=test[0], reindex=False)

        var_path = fc.variance.to_numpy()  # (n_origins, h), percent**2
        mu = fit.params["mu"]
        rows = out.index[test]
        assert len(var_path) == len(rows)
        out.loc[rows, "mu1"] = mu / 100
        out.loc[rows, "sigma1"] = np.sqrt(var_path[:, 0]) / 100
        out.loc[rows, "nu"] = fit.params["nu"]
        out.loc[rows, "mu_h"] = h * mu / 100
        out.loc[rows, "sigma_h"] = np.sqrt(var_path.sum(axis=1)) / 100
        out.loc[rows, "rv_h"] = np.sqrt(TRADING_DAYS * var_path.mean(axis=1)) / 100
    return out


def garch_fhs_var_es(
    returns: pd.Series,
    splits,
    horizons=(1, 10),
    kind: str = "garch",
    taus=(0.01, 0.05),
    simulations: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Filtered historical simulation (Barone-Adesi et al.): VaR and ES of the h-day return.

    For each (train, test) split: fit on returns[train], run the fixed-parameter filter to get the
    next-day variance at every test origin, then simulate `simulations` return paths through the
    GARCH/GJR recursion. The shocks are bootstrapped from the *training* standardised residuals, so
    the 10-day tails keep the fat-tailed shape instead of being forced to a normal.

    Columns: var_h{h}_{tau}, es_h{h}_{tau} (return quantiles in decimal; NaN outside test blocks).
    The forecast at origin t uses returns up to and including t (and the training residuals only).
    """
    rng = np.random.default_rng(seed)
    cols = [f"{k}_h{h}_{tau}" for h in horizons for tau in taus for k in ("var", "es")]
    out = pd.DataFrame(np.nan, index=returns.index, columns=cols)
    r_pct = returns * 100
    horizon_max = max(horizons)
    for train, test in splits:
        spec = dict(mean="Constant", vol="GARCH", p=1, o=KINDS[kind], q=1, dist="t")
        fit = arch_model(r_pct.iloc[train], **spec).fit(disp="off", update_freq=0)
        full = arch_model(r_pct.iloc[: test[-1] + 1], **spec).fix(fit.params)
        sigma2 = full.forecast(horizon=1, start=test[0], reindex=False).variance.to_numpy()[:, 0]
        pool = fit.std_resid.dropna().to_numpy()
        mu, omega = fit.params["mu"], fit.params["omega"]
        alpha, beta = fit.params["alpha[1]"], fit.params["beta[1]"]
        gamma = fit.params.get("gamma[1]", 0.0)

        n = len(sigma2)
        var_t = np.repeat(sigma2[:, None], simulations, axis=1)
        cum = np.zeros((n, simulations))
        rows = out.index[test]
        for step in range(1, horizon_max + 1):
            eps = np.sqrt(var_t) * rng.choice(pool, size=(n, simulations))
            cum += mu + eps
            var_t = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * var_t
            if step in horizons:
                ordered = np.sort(cum, axis=1)
                for tau in taus:
                    k = max(int(tau * simulations), 1)
                    out.loc[rows, f"var_h{step}_{tau}"] = ordered[:, k - 1] / 100
                    out.loc[rows, f"es_h{step}_{tau}"] = ordered[:, :k].mean(axis=1) / 100
    return out
