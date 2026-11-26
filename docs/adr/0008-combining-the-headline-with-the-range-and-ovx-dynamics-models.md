# 0008. Combining the headline with the range-based and OVX-dynamics models: candidates and rule, fixed before the run

- Status: accepted (Aayush, 2026-11-27)
- Date: 2026-11-27

## Context

No single change beat the headline (`lhar_ovx_bc_lag1`) on development data. In ADR 0005 the OVX-dynamics model (C3) had its largest gain at Brent 66 days (+3.7%, one-sided p 0.18) and was no better elsewhere. In ADR 0006 the model that replaces the return-based HAR terms by Parkinson range terms (R3) had positive gains in all six cells (+1.1% to +10.2%, smallest one-sided p 0.10) and passed nowhere. Each is a noisy estimate of a small gain. Averaging forecasts built on different information is the usual way to reduce that noise, and the combination in ADR 0005 (C5) mixed the headline with two models that carry no new information (GJR-t and leverage HAR) and was significantly worse in one cell.

This note fixes the combinations, the sample, the tests and the rule before the run. It concerns development data only (test origins 2012-01 to 2025-06). The holdout is not touched, `configs/protocol.yaml` does not change and there is no live track for this work.

## What is stated up front about the design

R3 and C3 were chosen after the results of ADR 0005 and 0006 were seen, because their raw gains were the largest at long horizons. That is a forking-paths risk. The three combinations below are the only ones run, and they use equal weights with nothing estimated, but a pass would still be weaker evidence than a design fixed before any result existed. It would be a development-data finding. The confirmation belongs to the one-time holdout run, which is not part of this work.

## Candidates (K = 3)

All forecasts are volatilities. A combination is the equal-weight average of its parts in log volatility (the geometric mean), as in C5.

- **M1:** the headline and R3.
- **M2:** the headline and C3.
- **M3:** the headline, R3 and C3.

The parts are exactly the forecasts of ADR 0006 (R3: Parkinson 5, 22 and 63 day terms, the three leverage terms and log OVX of the previous day) and ADR 0005 (C3: the headline inputs plus the 5-day and 22-day means of log OVX and the positive and negative parts of the gap between log OVX and log 22-day realised volatility), each by OLS with the QLIKE variance correction, expanding window, monthly refits, test origins and training rule as before.

## Sample

The range data start after OVX, so R3 is available on fewer origins than the headline (Brent 5 days: 3,389 instead of 3,413). Every candidate and the headline are compared on the same rows in a cell, the rows where R3 exists. The headline's QLIKE in this study therefore equals the `qlike_headline` of `range_candidates.csv`, not of `improve_candidates.csv`.

## Tests and selection rule

As in ADR 0005 and 0006, with a family of 18: 3 candidates times 6 cells (Brent and WTI at 5, 22 and 66 days) one-sided Diebold-Mariano tests (Harvey correction, rectangular kernel) of QLIKE against the headline, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 3 candidates.

- A candidate improves the headline if at least one cell has a Holm-adjusted p below 0.05 and no cell has an unadjusted one-sided p (that the candidate is worse) below 0.05.
- If several improve, the one with the largest mean QLIKE gain over the six cells is chosen.
- The chosen candidate is compared with naive, HAR, leverage HAR, GARCH-t and GJR-t, one-sided, Holm over 30 tests, and is said to beat the benchmarks only in the cells that pass that family.
- If none improves, that is reported and the headline stays.

## Consequences

- A winner is a development-data result selected by a pre-registered rule, not a confirmed result.
- Every candidate and cell is reported, including the failures.
- Intraday realised volatility has no usable free source here (see the README limitations), so it is not part of this study.
- Departures from this note are recorded as dated amendments here.
