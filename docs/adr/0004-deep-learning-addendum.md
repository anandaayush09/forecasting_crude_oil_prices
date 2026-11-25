# 0004. Deep-learning addendum: pre-registration of the experiments and their success rule

- Status: accepted (Aayush, 2026-11-25)
- Date: 2026-11-25

## Context

The learned residual models in `results/kaggle_residual_monthly/` do not beat the monthly leverage HAR with OVX. Before trying better designs, this note fixes what will be run and how it will be judged, so the outcome cannot be tuned after the fact. It is written before any run of the experiments below. The literature behind the choices (anchored and bounded corrections, QLIKE training, combination with the benchmark, pooling across assets) is summarised at the end.

This addendum is a development-period study (test origins 2012-01 to 2025-06). It does not change the headline, `configs/protocol.yaml` or the holdout rule. A positive result is exploratory and can only be confirmed by the live forward test.

## Stage A: per-asset fixes (Brent and WTI, 5, 22 and 66 days)

- **Anchor:** the headline benchmark itself, `lhar_ovx_bc_lag1`, refit monthly. The final forecast of a learned variant is the benchmark times exp(alpha times the learned correction), so alpha equal to zero reproduces the benchmark exactly.
- **Correction:** target is log realised volatility minus the log of the anchor regression forecast. The neural nets output a bounded correction, 0.5 times tanh of a zero-initialised last layer, so an untrained net changes nothing. Alpha is chosen by squared error on the last 20% of each training window, as in `models/ml/residual.py`. Residual models refit every 12 monthly splits.
- **Loss:** the neural nets train and early-stop on the QLIKE-consistent loss exp(2(r - c)) - 2(r - c) - 1, with r the target and c the correction. Ridge keeps squared error (it is the linear control).
- **Inputs:** the six channels of `features.vol.CHANNELS` over 22 days, with OVX and VIX lagged one trading day.
- **Seeds:** 5 per net (fewer than first planned, to fit the free GPU budget). Seeds are averaged in correction space, never picked.
- **Variants (K = 6):** ridge on the window, MLP, GRU, patch transformer, equal-weight combination of the transformer forecast and the benchmark in log volatility, and a logistic gate between the two with weight 1 / (1 + exp(-(a + b z))), z = log OVX minus log 22-day realised volatility. The gate parameters are fitted by QLIKE on forecasts of earlier test blocks whose targets are realised. With fewer than 250 such origins the gate weight is 0.5.
- **Controls:** ridge is the linear control. A transformer with the OVX channel shuffled over time is a placebo, reported but outside the test family.
- **Test family:** 6 variants times 6 cells = 36 one-sided Diebold-Mariano tests (Harvey correction, rectangular kernel) of QLIKE against the benchmark, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 6 variants.
- **Success rule:** a variant helps in a cell only if its Holm-adjusted p is below 0.05 and the SPA p of that cell is below 0.05. Anything else is reported as a null. Every variant and cell is reported.

## Stage B: pooled panel

- One shared net with an asset embedding, trained on about 8 to 10 free daily series (EIA petroleum spot prices, FRED exchange rates, each with its own HAR anchor and its own implied volatility where a free index exists). Evaluated only on Brent and WTI.
- Variants (K = 2): pooled MLP and pooled transformer against their Stage A per-asset versions, in a separate Holm family of 12 tests.
- The series list is fixed in a dated amendment to this ADR before the first Stage B run, after each series is confirmed free and each new feature builder has a leakage test.
- Stated limit: 8 to 10 series is below the roughly 40 where pooling has paid off in published work, so a null is the expected outcome.

## Foundation model track (needs protocol v2)

TTM r2.1 (IBM Granite, Apache-2.0, weights of 2025-01-31, pinned by Hugging Face revision hash), zero-shot, and its equal-weight combination with the headline, as a secondary live track. The primary live success rule (`configs/protocol.yaml`) does not change. A historical backtest of foundation models is not evidence, because their weights postdate the test period. Protocol v2 is a separate file and tag, created only with Aayush's approval.

## Evidence used for the design

- Anchored, bounded corrections are the only transformer form reported to beat HAR (Zhao and Qiao 2026; Heidari et al. 2026; Ye and Borde 2026), with gains of 0.4% to 3.7% QLIKE against HARs that were frozen or had no implied volatility.
- QLIKE training: 1.8% to 3.2% lower QLIKE at up to one week and about 0.3% at one month (Zhang et al. 2025). A network started at the HAR coefficients trains stably (Reisenhofer et al. 2022).
- Combination with the benchmark: the TTM and Log-HAR average is in the confidence set for 98% to 100% of assets (Brini 2026).
- Pooling: global networks overtake local ones from about 40 series (Liu et al.); naive pooling can hurt (Teller et al. 2025).
- Against: unanchored transformers, free residual LSTMs and zero-shot foundation models lose to a well-fitted HAR (Zhao and Qiao 2026; Lian et al. 2026; Brini 2026). Benchmark refit frequency and a linear control matter (Audrino and Chassot 2026; Zeng et al. 2023).

## Consequences

- If every variant is null, the deep-learning section of the paper reports a controlled null with the fixes tried.
- Any positive cell is labelled exploratory and goes to the live track for confirmation, not into the headline.
- Departures from this note are recorded as dated amendments here, not by rewriting it.

## Amendments

- 2026-11-25, after the Stage A run: `dm_hln` returns two-sided p-values, and the per-cell summary files written by the Kaggle runs (`dl_addendum_<asset>_<h>d.csv`) therefore carry two-sided values in `dm_p_better`. The registered one-sided p-values (halved when the model is better, one minus half otherwise) are computed by `experiments/dl_addendum_report.py` from the per-origin predictions and are the ones in `results/tables/dl_addendum.csv`. The runner now writes one-sided values too. No registered choice changed.
