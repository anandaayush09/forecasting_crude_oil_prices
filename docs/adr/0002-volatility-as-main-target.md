# 0002. Volatility and risk are the headline target; price forecasts are secondary

- Status: **drafted by the assistant at your request; read it, edit it, and make it yours**
- Date: 2026-11-23

## Context

- The DDP thesis reported a 26% lower 7-day MAE than the naive forecast. A controlled re-run without three whole-series VMD columns (`vmd_*_leaky`) gave 0.2 to 0.6% (Diebold-Mariano p 0.19 to 0.65) and direction accuracy fell from 72-75% to about 50%. The archive audit found no honest DDP price model that beats naive at any horizon (`prior_work/ddp/resume-numbers-justification.md`).
- `ROADMAP.md` already warned that daily price gains over the random walk would be tiny.
- (Numbers in this paragraph and the update below were computed on the full sample, which included the clean holdout from 2025-07; they are superseded by the development-only numbers in `ROADMAP.md` and `results/tables/SUMMARY.md`.)
- A volatility check, now reproduced in project code (tasks V1-V3, V5, V6), shows that the OVX index forecasts 22-day realised Brent volatility 31.0% better than the naive forecast (p = 0.0008) and 21.8% better than HAR (Brent, test 2012-2026, n = 3,718).

## Options considered

1. Keep daily price point forecasts as the headline and accept a likely null result.
2. Monthly price forecasts with fundamentals (futures curve, inventories, forecast combination); gains in the literature are small and need more data work.
3. Volatility and risk as the headline, price forecasts as an honest secondary result.
4. Both as two parts of one paper (broader, more work in six months).

## Decision

Option 3. `ROADMAP.md` and `configs/protocol.yaml` now put forward realised volatility (5, 22 and 66 trading days), VaR/ES and densities first. Deep, transformer and AR-MoE models predict a shrunk residual on the HAR + OVX anchor. Stretch criterion: a residual model inside the 90% Model Confidence Set on QLIKE at 2 or more horizons and beating HAR + OVX with p < 0.05 at 1 or more horizon (or in the top-volatility tercile).

## Consequences

- The benchmark is strong, so deep models may lose; the paper then reports a rigorous null for deep learning next to the OVX result and the leakage audit.
- "OVX predicts realised volatility" is known (literature check, 2026-11-23: Haugom et al. 2014, Gilder and Tsiaras 2020, Degiannakis and Filis 2017; raw unadjusted OVX already beats naive on QLIKE at 22 and 66 days in our data); the contribution is the leakage-free deep-residual test, the risk evaluation and the audit. Phase 0 must add oil implied-volatility papers to `literature/lit_matrix.csv`.
- Demoted: multi-horizon SPA, Pesaran-Timmermann direction and certainty-equivalent return become secondary price checks.
- **Update (baseline tables, 2026-11-23).** With a variance correction for QLIKE, the OVX regression beats naive, HAR and GARCH-family forecasts at 5 and 22 days (p < 0.03) and ties them at 66 days; learned residuals (ridge, LightGBM, MLP) add nothing detectable at 22 days. The headline must name the loss function, and the variance correction was introduced after seeing uncorrected results (both are reported). See the ROADMAP baseline section for the numbers and caveats.
- **Holdout correction (2026-11-23).** The first baseline runs saw data through 2026-09, including the holdout from 2025-07. All reported numbers were recomputed on development data only and a tested code guard now enforces this, but the holdout has been seen in aggregate. Treat the live forward test as the only fully clean evaluation, and say so in the paper.
- **Update (transformer run, 2026-11-23).** The Kaggle transformer / MLP / ridge residuals on HAR + OVX show no gain at 22 days; the single significant 5-day cell vanishes against monthly-refit benchmarks (yearly-anchor confound). The deep-learning stretch claim is currently unsupported; the fallback paper (OVX-based volatility and risk results, a rigorous null for deep residuals, the leakage audit) is the working plan.
- Revisit if, at the week-10 baseline table, OVX loses to HAR on the clean 2025-07 holdout, or if Brent and WTI disagree on the sign of the result.

## Notes from the task cards

- **V1, mean of squares.** Realised variance is the sum of squared returns because, for a martingale-difference return, its expected value is the integrated variance without needing a mean estimate; subtracting a sample mean from 22 noisy returns adds estimation error and changes almost nothing for oil.
- **V6, log of volatility.** Volatility is positive, right-skewed and its changes are multiplicative. Regressing log volatility on logs keeps forecasts positive, makes errors roughly symmetric and stabilises the variance; levels would be dominated by a few crisis periods.
- **V4, protocol review.** Items 1-6 of `docs/tasks/V4_protocol_review.md` are not answered here. They are your decisions before you tag `protocol-frozen`.
