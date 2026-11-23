# V6. Volatility benchmark ladder (naive, EWMA, HAR, OVX, HAR + OVX)

**Module you create:** `src/oilfc/models/baselines/vol.py`  **Tests:** `tests/test_vol_baselines.py`
**Needs first:** V1, V2, V3, V5 (the last test uses all of them).

## Why this matters

This is the card that tells you whether the research direction is real. On 2026-11-23 a throwaway check suggested that the OVX implied-volatility index forecasts 21-day realised Brent volatility about 33% better than the naive forecast and about 24% better than the HAR model (Diebold-Mariano p < 0.001). That check lived in a scratch script. This card puts it in the project, with tests, so the number is reproducible. Every deep and AR-MoE model later is judged against HAR + OVX, the strongest benchmark here.

## Read first

- Corsi (2009), "A simple approximate long-memory model of realized volatility" (HAR): what the three windows capture.
- A short note on the OVX index (CBOE crude oil volatility index): what it measures and when it is published.

## Interface

```python
from oilfc.models.baselines.vol import ewma_vol, har_design, forecast_log_linear

ewma_vol(returns, lam=0.94)  # annualised EWMA volatility; uses returns up to and including t
har_design(
    returns, ovx=None
)  # DataFrame: log_rv_5, log_rv_22, log_rv_63 [, log_ovx = log(ovx/100)]
forecast_log_linear(
    X, y_log, splits, cols
)  # OLS with intercept per split, trained on complete train rows,
# predictions for the test rows; NaN elsewhere
```

The docstring of `tests/test_vol_baselines.py` has the exact rules.

## Decision you must make

Why model the **log** of volatility (and add the log of OVX) instead of volatility itself? What would happen to forecasts and to the QLIKE loss if you regressed the level? Write it in ADR 0002.

## Hints (not answers)

- `har_design` can reuse `past_realized_vol` from V1.
- NumPy's `lstsq` or a small normal-equation solve is enough; the point is the split logic, not the regression.
- Leave rows with missing values out of the *training* fit; do not fill them.

## Acceptance test

`test_reproduces_the_preliminary_brent_result` runs on `data/processed/panel_daily.parquet` (it skips if the file is missing). With the definitions above it should land within 3 percentage points of the scratch result (OVX vs naive about 33%, vs HAR about 24%, DM p < 0.01); a faithful reference implementation gave 31.0% and 21.8%. If your numbers differ by more, the cause is almost always a window, gap or date-alignment slip: find it before you change the tolerance.

## Done when

- All tests pass, including the acceptance test, and you removed the `xfail` marker.
- Self-check: (1) why does OVX help? (2) why is HAR + OVX the benchmark to beat and not OVX alone? (3) what would make you distrust a result that is this large?

## Stretch

Add GARCH(1,1)-t and GJR-GARCH-t with the `arch` package, and compare on the same splits (weeks 8-10).
