# 0009. Intraday realised volatility inputs for WTI: candidates, controls and rule, fixed before any model is run

- Status: accepted (Aayush, 2026-11-27)
- Date: 2026-11-27

## Context

The headline (`lhar_ovx_bc_lag1`) measures past volatility from squared daily spot returns, which is a noisy measure. ADR 0006 tested a free substitute, the daily futures high and low. It passed nowhere, but its point gains at long horizons were mostly positive (ADR 0008 repeats the pattern for the combinations). The README lists the missing intraday realised volatility as a limitation that may explain part of the gap to published gains, and records that no free source had been found.

One now exists in part. VOLARE (Cipollini, Cruciani, Gallo, Insana, Otranto and Spagnolo 2026, arXiv 2602.19732, volare.unime.it) publishes daily realised measures for futures, built from Kibot tick data. Its WTI series (symbol `CL`) is free to download. It has no Brent series.

This note fixes the data, the timing, the candidates, the controls and the rule before any model uses the data. It concerns development data only (test origins 2012-01 to 2025-06). The holdout, `configs/protocol.yaml` and the live test do not change.

## Data

- File: the VOLARE futures variance archive downloaded on 2026-11-08 (`realized_variance_futures_20261030.zip`, SHA-256 `f425375e141c3b5b92dc64af1e903d2fac3d20f875e62a2f8a4c0bc2d7119db5`, the CSV inside has SHA-256 `458df82dbb209313e4bffb0b712e29738e3dc4d265700db6ed7340bb922fa9ae`). It is kept in `data/raw/volare/`, which is not committed, because the licence does not say that the file may be redistributed. The archive asks users to cite the paper above. VOLARE refreshes its files once a month, so a later download can differ, and the hashes identify the file used here.
- Series: `CL` only. 4,417 daily rows from 2009-09-28 to 2026-10-30, no duplicate dates, no missing values in the measures used below, no gap over four calendar days. Rows from 2025-07-01 are removed with `oilfc.eval.holdout.development_only` before any use.
- Construction as the VOLARE documentation states it: one-second mid-quote prices from tick bid and ask data, a session from Sunday 18:00 to Friday 17:00 Eastern Time with a daily break from 17:00 to 18:00, days with fewer than 40 observations or under two hours of trading dropped, and no outlier cleaning for futures. The documentation does not state the contract roll rule or how the overnight is treated.
- Date convention, checked against the EIA spot series: the squared WTI spot return correlates most strongly with the VOLARE measure of the same date (Spearman 0.41, against 0.35 one day earlier and 0.38 one day later), and the first price of a day is close to the previous close (median absolute gap 0.006%). So the date of a row is the trading day, the day runs from close to close, and it includes the overnight.
- Measures used: `rv5` (realised variance from 5-minute returns) and `rv5_ss` (the same with subsampling, which lowers the effect of microstructure noise). Both are daily variances, not annualised. The two have a log correlation of 0.995.
- Measures not used: the range-based columns of the file (`pr`, `gkr`, `rr5`) and the realised kernel. A scan of the file before this note was written (no forecasts involved) found impossible ranges in autumn 2009, for example a Parkinson variance of 0.18 on 2009-11-13 against an `rv5` of 0.0004, which fits uncleaned bad quotes. Realised variance from 5-minute returns did not show this, and the range-based inputs are covered by ADR 0006 already.
- What `rv5` leaves out: the weekend gap and the one-hour break gap, which fall outside the session. The target uses spot returns, which include them. The leverage and return terms that stay in the models carry that part.

## Timing

Strict, as for the ranges and OVX. A VOLARE day is complete at 17:00 Eastern, after the EIA spot price stamp of the same day. A feature at origin t therefore uses days up to and including t - 1.

## Features

Log annualised realised volatility over 5, 22 and 63 trading days: `log_rv_<w>` is 0.5 times the log of 252 times the mean daily `rv5` over the window, shifted one day, and `log_rvss_<w>` the same with `rv5_ss`. Window means use the days present, and a window needs at least 80% of its days (4 of 5, 18 of 22, 51 of 63), as in ADR 0006. The builder is a new function in `src/oilfc/features/` and goes into the `FEATURES` registry of `tests/test_leakage.py`. The first origin with all three windows is about 2009-12-07.

## Candidates (K = 3)

All use the headline's remaining inputs (leverage terms and log OVX of the previous trading day), the same expanding window with monthly refits, OLS with the QLIKE variance correction, and WTI at 5, 22 and 66 days.

- **I1:** the headline inputs plus the three lagged `log_rv` terms.
- **I2:** the three `log_rv` terms replace the three return-based HAR terms (leverage and OVX terms stay). It mirrors R3 of ADR 0006.
- **I3:** as I2 with the three `log_rvss` terms. It asks whether the noise reduction of subsampling matters.

## Controls (reported, outside the test family)

- **H-short:** the headline fitted on the same training rows as the candidates, that is only rows with the realised-volatility features available (from about 2009-12). The candidates start with about 2.1 years of training data at the first origin and the headline with about 4.6 years (OVX starts in 2007-05), which works against the candidates. H-short shows how much of any difference comes from the shorter sample and how much from the new input.
- **Range comparison:** R3 of ADR 0006 on the same rows, tested against I2 as a descriptive comparison, to show whether intraday information adds anything over the daily range.

## Sample

Every candidate, the headline and H-short are compared on the same test rows in a cell, the rows where the realised-volatility features exist. The headline's QLIKE in this study therefore equals the `qlike_headline` of `range_candidates.csv` only if the row sets match, and the table reports the number of origins per cell so that this can be checked.

## Tests and selection rule

As in ADR 0005, 0006 and 0008, with a family of 9: 3 candidates times 3 cells (WTI at 5, 22 and 66 days) one-sided Diebold-Mariano tests (Harvey correction, rectangular kernel) of QLIKE against the headline, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 3 candidates.

- A candidate improves the headline if at least one cell has a Holm-adjusted p below 0.05 and no cell has an unadjusted one-sided p (that the candidate is worse) below 0.05.
- If several improve, the one with the largest mean QLIKE gain over the three cells is chosen.
- The chosen candidate is compared with naive, HAR, leverage HAR, GARCH-t and GJR-t, one-sided, Holm over 15 tests (5 comparators times 3 cells), and is said to beat the benchmarks only in the cells that pass that family.
- If none improves, that is reported and the headline stays.

## What is stated up front about the design

- The idea was chosen after the results of ADR 0006 and 0008 were seen, because the long-horizon point gains of the range model were the largest effects found. That is a forking-paths risk. The three candidates above are the only ones run.
- Only WTI is tested, because VOLARE has no Brent. This follows from the data, not from a result, but it halves the family and the evidence, and a pass would say nothing about Brent, the primary target of the project.
- The WTI long-horizon gains of the range model were small in ADR 0006 (R3 +4.6% at 66 days, R1 and R2 worse), so there is no strong prior that WTI is where intraday data help most.
- The 66-day cell has about 450 training rows at the first origin, which is only a handful of non-overlapping 66-day blocks. Tests there have little power.
- A pass would be a development-data result selected by a pre-registered rule, not a confirmed result. It could not enter the live forecasts as it stands, because VOLARE updates once a month and the live track needs a value every day. The confirmation belongs to the one-time holdout run, which is Aayush's decision and not part of this work.

## Consequences

- Every candidate, control and cell is reported, including the failures.
- A null is a plausible outcome and would close the free-data question for intraday inputs, with the stated limit of a short sample and one asset.
- Raw VOLARE data is never committed. The experiment reads it from `data/raw/volare/`, and the tests use synthetic series only.
- Departures from this note are recorded as dated amendments here.

## Amendments

- 2026-11-27, before any model was run: the feature columns are named `log_irv_<w>` (from `rv5`) and `log_irvss_<w>` (from `rv5_ss`), not `log_rv_<w>` and `log_rvss_<w>` as written above, because `log_rv_<w>` is already the return-based HAR column of the headline. The windowing (5, 22 and 63 days, shifted one day, at least 80% of the days) is the code of ADR 0006, now shared in `windowed_log_vol`, and its output for the range features is unchanged. The runner checks the SHA-256 of the archive's CSV against the value above and stops if it differs. H-short sets the headline's inputs to missing on rows where the three `log_irv` terms are missing, so it trains and tests on the candidates' rows.
