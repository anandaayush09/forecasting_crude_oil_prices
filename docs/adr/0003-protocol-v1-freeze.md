# 0003. Protocol v1: what is frozen, and what freezing means here

- Status: **accepted by Aayush on 2026-11-25, drafted by the assistant. The `protocol-frozen` tag marks the commit that carries this status.**
- Date: 2026-11-24 (accepted 2026-11-25)

## Context

`configs/protocol.yaml` was a draft written before the volatility experiments. Since then the headline model changed (leverage HAR + OVX of the previous trading day, D-011), deep models got a monthly anchor (D-009, D-014), and the test list shrank to what was actually run. The file no longer described the work.

## What freezing means in this project

- The development results (test origins 2012-01 to 2025-06) were looked at while the models and tests were chosen. Freezing records what was run. It does not make those results pre-registered, and the paper should not say it does.
- The pre-registered parts are the clean holdout (from 2025-07-01) and the live forward test. After the tag, no rule in the file may change for them.
- The holdout was seen once in aggregate by early baseline runs (B-002, D-004). The live forward test is the only pristine evidence.
- A change after the tag needs protocol v2, a dated line here and a reason.

## What changed in the file

- Added: `version`, `headline_model`, `ovx_timing`, `claim_comparators`, `test_end`, `inference` (tests, Holm family, MCS settings, sub-periods), `holdout_rule`, `live_test` with the success rule from `live/README.md`.
- Renamed to the names used in code: benchmarks (`naive`, `har_bc`, ..., `lhar_ovx_bc_lag1`, `garch_t`, `gjr_t`).
- Removed because no code runs them: `mse_var` as a loss, the `metrics` entries MASE, success ratio, Winkler and coverage, and the `tests` entries Clark-West, Giacomini-White, multi-horizon SPA, Giacomini-Rossi and Pesaran-Timmermann. Kept: `ewma`, the ES exceedance test, CRPS and pinball, which `experiments/vol_risk_brent.py` uses. `daily` and `monthly` blocks are unchanged because the unfinished settings task (E0-2) reads them.
- Changed: deep residual models refit yearly, the anchor monthly (was "deep: yearly").
- Added `inference.window_sensitivity` (rolling windows of 1000, 504 and 252 days). The check is in `results/tables/window_robustness.csv`. It shows the 5-day gain survives rolling windows, while most of the Holm-corrected evidence does not (13 of 30 comparisons with the expanding window, 1 or 2 with rolling ones).
- Disclosed as a limitation in the file: the residual pipeline's anchor uses same-day OVX, the headline uses the lag. Fixing that needs a Kaggle rerun.

## The six items of task card V4: proposed answers

1. **Horizons 5, 22, 66.** Week, month, quarter. These match how a risk desk or a hedger reads volatility. Keep.
2. **Gap = horizon.** A training target at origin s uses returns s+1 to s+h. Without the gap, targets of the last training rows would contain returns after the first test origin, which the model could not know. A larger gap was not tested; if you want it, it is one rerun of `vol_ovx_timing.py` with a larger gap.
3. **OVX test start 2012-01-01.** OVX starts 2007-05, so the first refit has about 4.6 years of training data. No separate check was made that this is enough. The sub-period table `robustness_periods.csv` (2012-2019 against 2020-2025) is the available evidence that the result does not live only in one part of the sample.
4. **Clean holdout.** 2025-07-01 onward, about 16 months. I did not look at the data in it to judge whether it holds a calm and a volatile stretch (looking would break the rule). Proposed rule for breaking it open: never, except to fix a bug in the final script that changes no protocol rule.
5. **Primary loss QLIKE and the stretch criterion.** QLIKE is robust to noise in the volatility proxy and is the standard choice (Patton 2011). The stretch criterion (a residual model inside the 90% MCS at two or more horizons and beating HAR + OVX at p below 0.05) is not met so far, and the file does not claim it. It stays as a stated stretch.
6. **Brent primary, WTI robustness.** OVX is computed from WTI options, so it may fit WTI better. All headline tables show both assets side by side. The mismatch is a stated limitation, not hidden.

## Self-check (from the task card)

- **Why freeze before the final results?** So no rule can be tuned to the answer on the holdout and live data.
- **Cost of freezing too early?** A rule later found to be wrong stays for the confirmation tests. Protocol v2 fixes that but costs credibility.
- **What would a hostile reviewer attack first?** That the headline model was chosen after seeing development results. The answer is the live test and the single holdout run, plus the Holm-corrected tables that show where the claim does not hold (the 22-day gain survives the Holm correction only against some comparators, and the 66-day gain only against naive).

## Outcome of the holdout run (2026-11-27)

`experiments/holdout_final.py --confirm-single-run` was run once, at Aayush's request, at code commit `473ffe0`. The record is `results/holdout/holdout_v1.json` (SHA-256 of its content `fb423f8c77201494b4b0366cfb45339fd83b93aeaec0fac708f3a45fc9b9974d`), written once and never overwritten. Origins run from 2025-07-01 to the data end of 2026-11-17, with 276 to 347 origins per cell with a realised target.

Before the run, 17 source files differed from the freeze tag (the dry run lists them, and the record stores the list). Because of that, the headline path and the five comparators were recomputed on development data with the code of `protocol-frozen` (a separate checkout) and with the code at the run. Every forecast matched exactly (largest relative difference 0.0, 6 cells, 6 models), and so did `holdout_forecasts`, the function the run uses.

Result: the headline `lhar_ovx_bc_lag1` has a lower mean QLIKE than every comparator in all 30 comparisons (naive, HAR, leverage HAR, GARCH-t, GJR-t at 5, 22 and 66 days on both assets), with gains from 1.6% (GARCH-t, Brent 22 days) to 72.2% (naive, WTI 66 days). No comparison is significant. The two-sided Diebold-Mariano p-values are 0.17 to 0.97, and all 30 Holm-adjusted p-values are 1.0. Halving the two-sided p for the one-sided claim gives 0.085 at the smallest, and Holm still gives 1.0.

Reading: the point estimates favour the headline in every cell, but 15 months of holdout data cannot confirm it. With overlapping targets, the 66-day cells hold only about four non-overlapping blocks, and the other cells have low power too. The result does not support the claim that the headline beats the benchmarks, and it does not contradict it. Together with the early aggregate look at these data (BUGS B-002, DECISIONS D-004), the live forward test remains the only pristine evidence.

A docstring of `evaluate()` in `src/oilfc/eval/holdout_final.py` called `dm_p` one-sided. It is the two-sided `dm_hln` p-value, which is what this protocol names as the primary test. The record is correct as written, and the docstring was corrected after the run (documentation only).

## Consequences

- The final holdout script can be written and run once after the tag.
- `tests/test_protocol_draft.py` now also checks that the live test models and horizons equal `oilfc.live.forecast.MODELS` and `HORIZONS`, so the file cannot drift from the code unseen.

## Amendments

- 2026-11-27: protocol v2 is proposed in ADR 0010 (`configs/protocol_v2.yaml`). It leaves v1 unchanged, including the primary live success rule, and adds one secondary live track for a foundation model. It takes effect only when ADR 0010 is accepted and the tag `protocol-v2-frozen` exists.
