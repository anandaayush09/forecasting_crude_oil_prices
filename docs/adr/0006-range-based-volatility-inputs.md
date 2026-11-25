# 0006. Range-based volatility inputs: candidates and the selection rule, fixed before any data is downloaded

- Status: accepted (Aayush, 2026-11-25)
- Date: 2026-11-25

## Context

ADR 0005 found that no change of estimator improves the headline with the inputs we have. The remaining lever is better information about past volatility. The headline measures past volatility from squared daily spot returns, a noisy measure. Intraday realised volatility is the best fix and has no free source here. The daily high and low of the futures contract carry much of the same intraday information (Parkinson 1980; Garman and Klass 1980) and can be downloaded free. Published HAR work on equity indices reports gains from Parkinson and Garman-Klass inputs, and I found no direct crude-oil test.

This note fixes the candidates, the timing and the rule before the data is downloaded. It concerns development data only (test origins 2012-01 to 2025-06). The holdout, `configs/protocol.yaml` and the live test do not change.

## Data and timing

- Daily open, high, low and close of the front-month WTI (NYMEX) and Brent (ICE) futures from a free source (Yahoo Finance chart data for CL=F and BZ=F, fetched with `requests`). Rows with a non-positive price, a high below the low, or a missing field are dropped. The series are futures, not the EIA spot prices that the target uses, so they are an additional measurement of the same market.
- Strict timing, as for OVX: the range of day t is known only after the spot price stamp of day t, so features at origin t use the range through day t - 1.
- Daily Parkinson variance: (ln H - ln L)^2 / (4 ln 2). Daily Garman-Klass variance: 0.5 (ln H - ln L)^2 - (2 ln 2 - 1)(ln C - ln O)^2. Past volatility over a window is the log of the square root of 252 times the mean daily variance over that window (as `past_realized_vol`).
- Windows are 5, 22 and 63 days, as in the HAR inputs. Training rows without the range inputs are dropped, as OVX already limits the sample to 2007-05 onward.

## Candidates (K = 3)

All use the headline's other inputs (leverage terms, log OVX of the previous day), the same expanding window with monthly refits, OLS with the QLIKE variance correction, and the same six cells (Brent and WTI at 5, 22 and 66 days).

- **R1:** the headline inputs plus the three lagged Parkinson terms.
- **R2:** the headline inputs plus the three lagged Garman-Klass terms.
- **R3:** the three Parkinson terms replace the three return-based HAR terms (leverage and OVX terms stay).

## Tests and selection rule

As in ADR 0005, with the family size changed: 3 candidates times 6 cells = 18 one-sided Diebold-Mariano tests against the headline (Harvey correction), Holm-adjusted together, plus Hansen's SPA per cell over the 3 candidates (consistent p-value, block length h, 1,000 repetitions, seed 0). A candidate improves the headline if it passes Holm in at least one cell and is significantly worse (unadjusted one-sided p below 0.05) in none. If several improve, the largest mean QLIKE gain over the headline across the six cells decides. The chosen candidate is then compared with naive, HAR, leverage HAR, GARCH-t and GJR-t, one-sided, Holm over 30 tests, and is said to beat the benchmarks only in the cells that pass that family. If none improves, the headline stays and the result is reported.

## Consequences

- A winner is a development-data result selected by a pre-registered rule, not a confirmed result.
- Every candidate and cell is reported.
- Departures from this note are recorded as dated amendments here.
