# V5. Expanding-origin splits with a gap

**Module you create:** `src/oilfc/eval/rolling.py`  **Tests:** `tests/test_rolling.py`

## Why this matters

Every model in the paper is evaluated by the same loop: fit on the past, predict the next block, move forward. If one training row's *target* reaches into the test block, the model has seen the answer; with a forward-looking target such as realised volatility over h days, that happens easily. This function is the place where the gap is enforced, once, for every model. The DDP Track B protocol had `train_end == test_start` and no gap; the leakage audit section of the paper rests on you not repeating that.

## Read first

- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, the section on time-series cross-validation.
- Lopez de Prado, *Advances in Financial Machine Learning*, chapter on purging and embargo (the idea, not the code).

## Interface

```python
from oilfc.eval.rolling import expanding_splits

splits = expanding_splits(index, test_start, h, refit_months)
# index: DatetimeIndex of forecast origins; test_start: pd.Timestamp; h: int (rows); refit_months: int
# returns a list of (train_positions, test_positions): integer numpy arrays of positions into `index`
```

The docstring at the top of `tests/test_rolling.py` states the rules exactly. In short: the training origins are the prefix `0 .. first_test - h - 1` (so the last training target is known before the first test origin), and test block k holds the origins dated in `[test_start + k*m months, test_start + (k+1)*m months)`.

## Decision you must make

Blocks with no origins (a holiday-only month) and the final partial block: skip them, or return them? Write the rule and where you test it.

## Hints (not answers)

- `pd.DateOffset(months=...)` adds calendar months correctly.
- Build each block from a boolean mask over `index`; do not loop over dates.
- The second test uses `np.diff` to check the training window is an unbroken prefix.

## Done when

- All tests pass and you removed the `xfail` marker.
- Self-check: (1) draw the timeline of one split and mark the gap; (2) why can a training target overlap the test block even when the training *inputs* do not? (3) what changes if a model also uses a validation block for tuning?

## Stretch

Add a `purge_embargo` argument and test it against the same overlap check.
