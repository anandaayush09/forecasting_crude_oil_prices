# V1. Realised-volatility targets

**Module you create:** `src/oilfc/eval/targets.py`  **Tests:** `tests/test_vol_targets.py`

## Why this matters

Volatility is now the headline target (ADR 0002). Every number in the paper depends on what "realised volatility over the next h days" means and on never using information from the wrong side of the forecast origin. A target that accidentally includes today's return, or a predictor that peeks forward, is exactly the kind of bug that produced the DDP 26% result. You will write both versions, forward (the thing we predict) and past (the thing we use as an input), and make the tests prove which days each one touches.

## Read first

- Andersen and Bollerslev (1998), "Answering the skeptics" (realised variance from squared returns). Free versions exist on the authors' pages and SSRN; note the link in `literature/lit_matrix.csv`.
- pandas docs: `Series.rolling`, `Series.shift`.

## Interface

```python
from oilfc.eval.targets import forward_realized_vol, past_realized_vol

# r: pd.Series of daily log returns, DatetimeIndex, one row per trading day
forward_realized_vol(r, h)  # value at t uses r[t+1 .. t+h]; the last h values are NaN
past_realized_vol(r, h)  # value at t uses r[t-h+1 .. t] (today included); first h-1 values are NaN
```

Definition (both): `sqrt(252 * mean(r**2))` over the window. No mean is subtracted. The result is annualised volatility, in decimal (0.35, not 35).

## Decision you must make

Mean of squared returns, or the sample standard deviation (subtract the mean, divide by n-1)? Over 22 days they are almost identical for oil. Write down in two lines why the mean of squares is the usual choice for realised variance. Put the answer in ADR 0002.

## Hints (not answers)

- `forward` is `past` moved by a number of rows. Which direction, and by how many?
- Test the boundary: change a return exactly at t, at t+1, at t+h and at t+h+1, and see which outputs move.

## Done when

- All tests pass and you removed the `xfail` marker.
- Self-check: (1) why must the forward window start at t+1 and not t? (2) why is the sample's last stretch of forward values missing, and what would go wrong if you filled it in? (3) why annualise?

## Stretch

Add `forward_realized_vol_garman_klass(ohlc, h)` later if you obtain OHLC data (not free for Brent at intraday frequency, so optional).
