# 0010. Foundation-model live track and protocol v2: design fixed before any forecast is made

- Status: proposed (2026-11-27), waiting for Aayush's acceptance. Nothing is run and no tag exists until it is accepted.
- Date: 2026-11-27

## Context

ADR 0004 reserved a secondary live track for a pretrained time-series foundation model, zero-shot, and its equal-weight combination with the headline. It needs protocol v2, because ADR 0003 allows no change to the frozen protocol without one. A historical backtest would say nothing, since the model's weights postdate the development period, so the only evidence is forecasts written after the design is fixed, as in the primary live test.

Published support is thin and mixed. Brini (arXiv 2607.05291, 2026) finds that among nine zero-shot foundation models only TTM beats a log-HAR at every horizon on the VOLARE realised-volatility data, by about 1.3% to 1.8% in QLIKE, that the edge is concentrated in a few assets, and that an equal-weight average of TTM and log-HAR is in the model confidence set for 98% to 100% of assets. Zhao and Qiao 2026 and Lian et al. 2026, as cited in ADR 0004, find that zero-shot foundation models lose to a well-fitted HAR. A null is a plausible outcome.

## What protocol v2 is

`configs/protocol_v2.yaml` extends v1 without changing it. It records v1's identity (the SHA-256 of `configs/protocol.yaml` with LF line endings, equal for the file and for the `protocol-frozen` tag) and `tests/test_protocol_v2.py` fails if v1 changes. The headline, the development protocol, the one-time holdout (already run) and the primary live success rule stay as in v1. v2 adds one secondary track.

## Model

TTM r2.1 (Tiny Time Mixers, IBM Granite, Apache-2.0 per the model card): repository `ibm-granite/granite-timeseries-ttm-r2`, branch `512-96-ft-r2.1`, pinned to commit `3442406f106122f414a11f72b4ebfe4878bec89d` (committed 2025-02-26 14:33 UTC according to the Hugging Face API, checked 2026-11-27). The date of 2025-01-31 in ADR 0004 is superseded by this pin. The branch name is context 512, prediction length 96, frequency prefix tuning ("ft", which the card says to use with the daily resolution that r2.1 added), MSE pretraining and release r2.1. A prediction length of 96 covers the longest horizon of 66 days. Used zero-shot: no fitting, no fine-tuning, no variance correction, no random seeds. The `ttm-l1` variants and the other context lengths are not used, and are not tested.

## Input and forecast

One series per horizon h, built from the same EIA spot prices as the primary track and known at the origin t: the annualised log realised volatility over the last h returns, `0.5 * log(252 * mean(r^2))`, for the last 512 trading days ending at t. OVX is not an input. The forecast of the volatility over t+1..t+h is the exponential of the model output at step h, because the value of that series at t + h is exactly the target. There is no averaging and no recalibration. If any value in the context is not finite, no forecast is written for that origin and the skip is logged (no such window exists in the Brent or WTI history to 2026-11-17: the smallest 5-day mean squared return is 1.7e-6).

Combination: `ttm_headline_mean = exp(0.5 * log(ttm) + 0.5 * log(headline))`, equal weights in log volatility as in ADR 0008. Brini's average may be arithmetic, so the combination here is not a replication of it.

## Live procedure

Same as the primary track: forecasts for the latest origin, written once as JSON with a SHA-256, listed in a manifest and committed, in the separate folder `live/forecasts_ttm/` so that the primary manifest and hashes are untouched. The model version, the revision and the library versions are stored in each record. Only origins dated after the v2 tag count toward the claim. Running the code on development dates is allowed to check that it works (finite, correct shape, no use of later data) and no loss is computed from it.

## Success rule (secondary track)

At least 60 scored origins per horizon. The equal-weight combination has a lower mean QLIKE than the headline alone at 5 and 22 days on both assets, with a Diebold-Mariano p (`dm_hln`, rectangular kernel, two-sided as in v1) below 0.05 after Holm adjustment over those 4 tests. TTM alone against the headline and the 66-day cells are reported and are not part of the claim. A result against this is reported as it comes, and no model, horizon or variant may be added to the claim afterwards.

## Stated limits

- Origins are daily and the targets overlap, so 60 origins is a small number of independent windows, above all at 22 days. The test has little power and a pass is not likely to be clear-cut either way.
- The first scored origins at 22 days need 22 trading days after the first forecast, and the first 66-day scores need about 3 months.
- The EIA prices arrive about a week late, so TTM sees the same information as the headline. It has no OVX, which is part of why a combination, not TTM alone, is the claim.
- The design choices above (branch, series, step h, log combination) were made once, here, without looking at any TTM forecast. They are not tuned.

## Work after acceptance (not part of this note)

1. Add `granite-tsfm` and a CPU `torch` to the live workflow. The install time and the model download in GitHub Actions need checking, and a failure there must not stop the primary track.
2. A builder for the input series in `src/oilfc/features/`, registered in `FEATURES` of `tests/test_leakage.py`, and `src/oilfc/live/ttm.py` with its tests.
3. A scoring step for `live/forecasts_ttm/`, kept apart from `live/scores.csv`.
4. Tag `protocol-v2-frozen` on the commit that carries the accepted status. Pushing the tag needs Aayush's separate yes.

## Consequences

- If the combination does not beat the headline, the foundation-model section of the paper reports a null from a pre-specified, outside-the-sample test.
- The primary live claim cannot be affected by this track.
- Departures from this note are recorded as dated amendments here.
