# V3. Diebold-Mariano test (Harvey-Leybourne-Newbold form)

**Module you create:** `src/oilfc/eval/stat_tests.py`  **Tests:** `tests/test_stat_tests.py`

## Why this matters

"Model A has lower average loss than model B" is not a result until you know whether the gap could be chance. The Diebold-Mariano test is how this field answers that; the Harvey-Leybourne-Newbold correction makes it reliable in the samples we have; and the HAC variance is what keeps overlapping h-day forecasts from fooling you. The DDP thesis had the test in the code but an empty results table, so no p-value was ever reproducible. This card fixes that for good.

## Read first

- Diebold and Mariano (1995), and Harvey, Leybourne and Newbold (1997) for the small-sample correction.
- Any short explanation of "autocovariance of a loss differential" and "long-run variance" (the Newey-West idea).

## Interface

```python
from oilfc.eval.stat_tests import dm_hln

stat, p = dm_hln(loss_a, loss_b, h)
```

- `loss_a`, `loss_b`: per-origin losses of two forecasts (arrays or Series of equal length).
- `h`: forecast horizon in origins. It sets how many autocovariances enter the variance.
- `stat > 0` means forecast **b has the lower** average loss. `p` is two-sided, from a t distribution with n-1 degrees of freedom.
- Rows where either loss is NaN are dropped together. If the loss differential has zero variance, return `(nan, nan)`.
- Raise `ValueError` for unequal lengths or `h < 1`.

## Decision you must make

A forecast of horizon h made every day overlaps with its neighbours. How many autocovariances does the variance need, and what happens to the p-value if you use too few? (Write the answer in your own words; the autocorrelation test in the test file shows the effect.)

## Hints (not answers)

- Start with h = 1; there the test reduces to a t-test on the mean of the differences (with a variance that divides by n). The first test has the formula.
- Then add the extra autocovariance terms, then the HLN factor.
- Do the arithmetic on paper for a six-element example before coding.

## Done when

- All tests pass and you removed the `xfail` marker.
- Self-check: (1) what is the null hypothesis? (2) why does ignoring autocorrelation make the test too eager? (3) why a t distribution and a correction in small samples?

## Stretch

Add `clark_west(loss_restricted, loss_unrestricted_adjusted, h)` for nested models (the HAR vs HAR + OVX comparison is nested).
