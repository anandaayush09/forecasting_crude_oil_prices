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
- 2026-11-27, before any Stage B run: series list and design of Stage B, as required by the Stage B section. Nothing here changes Stage A. The ADR 0007 result (no architecture helps) is known and does not alter this design.

  **Series (8).** Evaluated: Brent (`RBRTE`) and WTI (`RWTC`) from EIA, as in the rest of the study. Companions, used only as training rows, all free: NY Harbor ultra-low-sulfur diesel (EIA `EER_EPD2DXL0_PF4_Y35NY_DPG`), NY Harbor conventional gasoline (`EER_EPMRU_PF4_Y35NY_DPG`), US Gulf Coast kerosene-type jet fuel (`EER_EPJK_PF4_RGC_DPG`), Mont Belvieu propane (`EER_EPLLPA_PF4_Y44MB_DPG`), Henry Hub natural gas (`RNGWHHD`, an EIA file under the `ng` path, not `pet`) and EUR per USD (FRED `DEXUSEU`). The yen and Canadian dollar rates were considered and left out: they have no free implied volatility index and are the weakest link to crude. With 8 series the list is inside the 8 to 10 of the Stage B section.

  **Implied volatility.** Brent and WTI use OVX. EUR per USD uses the euro currency volatility index (FRED `EVZCLS`), which exists only from 2007-11-01 to 2025-03-11. Every other series has none. Where a series has no index on a date, its implied-volatility channel is set to zero and a 0/1 indicator channel marks it as missing, so the net can tell a missing input from a real zero. Implied-volatility channels are lagged one trading day, as in Stage A.

  **Data checked on 2026-11-27**, for 2007-05-10 (the OVX start) to 2025-06-30. All series start before 2007-05-10, except EVZ. No value is zero or negative. The longest run of missing weekdays is 2 for every series except gasoline (3) and jet fuel (5, 2011-07-11 to 2011-07-15, which is at the limit of the 5 weekdays allowed and is kept).

  | series | first date | last date used by the study |
  |---|---|---|
  | ULSD | 2006-06-14 | 2025-06-30 |
  | gasoline | 1986-06-02 | 2025-06-30 |
  | jet fuel | 1990-04-02 | 2025-06-30 |
  | propane | 1992-07-09 | 2025-06-30 |
  | Henry Hub | 1997-01-07 | 2025-06-30 |
  | EUR per USD | 1999-01-04 | 2025-06-30 |
  | EVZ | 2007-11-01 | 2025-03-11 |

  **Conditions before the first run** (each is a code change with a test, committed before the run): the fetcher takes the EIA path so that Henry Hub can be read; release lags for the new series go into `configs/data.yaml` (spot prices lag 0 business days, the exchange rate 5 as for the broad dollar index); the pooled channel builder is added to the `FEATURES` registry in `tests/test_leakage.py` and passes; companion series are cut at the holdout with `oilfc.eval.holdout.development_only` like every other input. A series that fails these is dropped, not replaced, and the amendment is re-dated.

  **Design.** The anchor of Brent and WTI is the headline `lhar_ovx_bc_lag1`, unchanged. Each companion gets the same leverage HAR with the QLIKE variance correction (expanding window, monthly refits), without the implied-volatility term where it has none. One shared net with a learned embedding per series, 22-day window, the Stage A settings (width 32, bounded zero-start QLIKE correction, last linear layer zero-initialised, alpha chosen on the last 20% of each training window, 5 seeds averaged in correction space, refit every 12 monthly splits). No new tuning, so any difference comes from pooling and not from settings. Inputs per day are the six channels of `features.vol.CHANNELS` computed for the series itself, plus the missing-indicator channel. At each monthly origin the net trains on all series with every training target ending before that origin. Test origins are Brent and WTI only, 2012-01 to 2025-06.

  **Tests.** Variants (K = 2): pooled MLP and pooled transformer. Family: 2 variants times 6 cells (Brent and WTI at 5, 22 and 66 days) = 12 one-sided Diebold-Mariano tests (Harvey correction, rectangular kernel) of QLIKE against the Stage A per-asset version of the same architecture, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 2 variants. A variant helps in a cell only if its Holm-adjusted p and the SPA p of that cell are both below 0.05. Outside the family, each pooled variant is also compared with the headline, with gain and one-sided p. Beating a per-asset net that itself did not beat the headline is not a gain over the benchmark, and the write-up says so.

  **Stated limit.** 8 series is well below the roughly 40 where pooling has paid off in published work, and the companions share a volatility regime only loosely with crude. A null is the expected outcome. A positive cell would be labelled exploratory.
- 2026-11-27, before any Stage B run: implementation details of the amendment above, written with the code (`src/oilfc/models/ml/pooled.py`, `src/oilfc/features/pooled.py`, `experiments/vol_dl_pooled.py`). None changes the series, the design or the tests.
  - **Series embedding.** Each series has a constant one-hot channel over the whole window. Through the first layer of the net this acts as a learned embedding of the series, so the existing MLP and patch transformer are used unchanged. Each day has 15 channels: the 7 of `features.pooled.POOL_CHANNELS` (return, absolute return, log realised volatility at 5 and 22 days, own implied volatility or 0, VIX, missing indicator) and 8 one-hot columns.
  - **VIX** is the same market series for every net input, carried forward from the last observation on or before each series' own dates and delayed one trading day. It is not a series-specific input.
  - **Usable rows.** A row of any series is used by a test block only if it is complete and the date of the last return in its target lies before the block's first origin. A companion joins a fit only with at least 100 such rows, otherwise it is left out of that fit. The evaluated asset needs at least 250, as in Stage A.
  - **Undefined log volatilities.** Realised volatility over 5 days is zero, so its logarithm undefined, on 16 days in the development sample (15 for propane, 1 for Henry Hub, runs of unchanged prices). Every row whose window or target contains such a value is left out, not filled.
  - **Implied volatility.** EVZ is carried forward for at most 5 calendar days, so the series' end on 2025-03-11 shows as missing. EUR per USD has the EVZ term in its anchor, and its rows without an EVZ value are not used. The other companions have no implied-volatility term and no such restriction.
  - **Fits.** One net is trained on the anchor residuals of all series stacked in date order, and its early stopping holds out the latest 15% of those rows (as in Stage A, where the same share of the rows is held out). Alpha is chosen by squared error on the last 20% of the evaluated asset's usable rows, with the net and the anchors fitted on all rows dated before that validation block. The net and alpha are refit every 12th monthly split and reused in between.
