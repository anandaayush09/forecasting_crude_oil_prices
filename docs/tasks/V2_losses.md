# V2. Loss functions: QLIKE, log-vol MSE, normal CRPS

**Module you create:** `src/oilfc/eval/losses.py`  **Tests:** `tests/test_vol_losses.py`

## Why this matters

A forecast is only "better" under a loss function, and for volatility the choice is not neutral. Patton (2011) showed that a few losses, QLIKE among them, rank volatility forecasts correctly even when the realised-volatility proxy is noisy; plain squared error on the level does not. The paper's primary loss is QLIKE; the others are cross-checks. CRPS scores a whole predictive distribution and is the density criterion in the pre-registered success list.

## Read first

- Patton (2011), "Volatility forecast comparison using imperfect volatility proxies", *Journal of Econometrics*. Read the section on robust loss functions; note which are on the list.
- Gneiting and Raftery (2007), "Strictly proper scoring rules", section on the CRPS and its closed form for a normal.

## Interface

All three work element by element on arrays or Series and return an array of the same length (the harness averages them, and the DM test needs the per-origin values):

```python
from oilfc.eval.losses import qlike, mse_logvol, crps_normal

qlike(realized, forecast)  # z = (realized / forecast)**2 ;  z - log(z) - 1
mse_logvol(realized, forecast)  # (log realized - log forecast)**2
crps_normal(
    x, mu, sigma
)  # CRPS of N(mu, sigma**2) at the observation x (closed form, no integration)
```

`realized` and `forecast` are volatilities (not variances), both positive.

## Decision you must make

QLIKE is written here with volatilities and a squared ratio. Show that this equals the usual variance form `sigma2_realised / sigma2_forecast - log(sigma2_realised / sigma2_forecast) - 1`. Which of the two would you rather store in a results table, and why?

## Hints (not answers)

- The tests include a numerical integral of the CRPS definition. Reading it tells you what the closed form must equal.
- QLIKE punishes forecasts that are too low more than equally too-high ones. The test checks this; sketch why on paper first.

## Done when

- All tests pass and you removed the `xfail` marker.
- Self-check: (1) what does "strictly proper" mean? (2) why is QLIKE less sensitive to a few huge realised values than squared error on the level? (3) why does CRPS reward a sharp forecast only when it is also calibrated?

## Stretch

Add `pinball(y, q, tau)` and `fz0(y, var, es, tau)` for the risk evaluation (needed in weeks 8-10).
