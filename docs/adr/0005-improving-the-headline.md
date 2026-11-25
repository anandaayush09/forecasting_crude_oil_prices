# 0005. Improving the headline on development data: candidates and the selection rule, fixed before the run

- Status: accepted (Aayush, 2026-11-25)
- Date: 2026-11-25

## Context

The headline (leverage HAR with OVX of the previous trading day, OLS, QLIKE variance correction) beats naive, HAR, leverage HAR, GARCH-t and GJR-t at 5 days. At 22 days its gain over HAR (22% on Brent, 27% on WTI) is not significant after correction, and at 66 days it does not beat HAR. Deep-learning variants did not help (ADR 0004). Published work points to the estimation and the inputs of the HAR-type model as the places with real gains: QLIKE estimation instead of OLS (Zhang et al. 2025), weighted least squares (Audrino and Chassot 2026), implied-volatility dynamics and the variance risk premium, and combining econometric forecasts.

This note fixes the candidates, the tests and the selection rule before any candidate is run. It concerns development data only (test origins 2012-01 to 2025-06). The holdout is not touched, `configs/protocol.yaml` does not change, and there is no live track for this work.

## Candidates (K = 5)

All use the headline's inputs timing (OVX of the previous trading day), the same expanding window with monthly refits, the same training rule (origins stop h + 1 rows before each test block), and the same six cells (Brent and WTI at 5, 22 and 66 days).

- **C1:** the headline regression (leverage HAR terms plus log OVX) estimated by minimising the mean QLIKE on the training rows, starting from the OLS fit with the variance correction. The forecast is exp of the fitted log volatility, with no further correction.
- **C2:** the same regression by weighted least squares: one OLS fit, then weights equal to 1 over the fitted variance, exp(-2 times the fitted log volatility), then QLIKE variance correction from the residual variance.
- **C3:** the headline inputs plus four OVX-dynamics inputs: the 5-day mean and the 22-day mean of log OVX, and the positive part and the negative part of the gap log OVX minus log 22-day realised volatility (the variance risk premium, split by sign so it is not collinear with the existing inputs). OLS with the QLIKE variance correction.
- **C4:** the C3 inputs estimated by minimising QLIKE as in C1.
- **C5:** the equal-weight average, in log volatility, of the headline, the GJR-t forecast and the leverage HAR forecast.

## Tests and selection rule

- **Primary test:** for each candidate and cell, a one-sided Diebold-Mariano test (Harvey correction, rectangular kernel) that the candidate has the lower mean QLIKE than the headline. 5 candidates times 6 cells give 30 tests, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 5 candidates.
- **A candidate improves the headline if** at least one cell has a Holm-adjusted p below 0.05 and no cell has an unadjusted one-sided p (that the candidate is worse) below 0.05.
- **If several improve,** the one with the largest mean QLIKE gain over the headline across the six cells is chosen.
- **Claim against the benchmarks:** the chosen candidate is compared with naive, HAR, leverage HAR, GARCH-t and GJR-t, one-sided, 6 cells, Holm over the 30 tests. It is said to beat the benchmarks only in the cells that pass this family.
- **If no candidate improves,** that is reported and the headline stays.

## Consequences

- The selection among five candidates is done once, by this rule, on the development period. A winner is reported as a development-data result selected by a pre-registered rule, not as a confirmed result.
- Every candidate and cell is reported, including the failures.
- Departures from this note are recorded as dated amendments here.
