# Literature check: OVX, HAR and neural residuals for oil volatility (2026-11-23)

**How this was done.** Web searches (abstracts, repository pages and search snippets); the full text of almost no paper was read. `literature/lit_matrix.csv` now has 22 rows and each row says what was read. Everything below is a lead to verify, not a finding. Open the free links in the matrix before citing anything.

## 1. Is "implied volatility predicts oil volatility" already known? Yes.
- Implied-volatility forecasts of crude oil volatility have been compared with GARCH and historical measures for decades: Agnolucci (2009, *Energy Economics*) reports GARCH-type forecasts beating at-the-money implied volatility (restricted text; second-hand summary); older work cited in the searches reports implied volatility beating GARCH at about two weeks and historical volatility at about four weeks.
- OVX in particular: Haugom, Langeland, Molnar and Westgaard (2014) are reported to find that OVX improves realised-volatility forecasts for WTI (second-hand only; full citation not verified). Mazzeu et al. (2019, *Journal of Forecasting*) model and forecast OVX itself with HAR extensions. Gilder and Tsiaras (2020, *Journal of Futures Markets*) compare option-implied measures with OVX and find a narrow-corridor measure best and OVX the *least accurate of the implied measures*, with a simple risk-premium adjustment helping all of them.
- HAR with exogenous channels for oil: Degiannakis and Filis (2017, *J. Int. Money and Finance*) for Brent at 1 to 66 days; Luo, Klein, Walther and Ji (2024, *J. Forecasting*) with machine-learning variable selection for WTI.

**So the contribution cannot be "OVX helps".**

## 2. How our result relates (Brent, test 2012-2026, QLIKE unless stated)
| | 5 days | 22 days | 66 days |
|---|---|---|---|
| Raw OVX as published (no fitted parameters) vs naive | 11.4% (p = 0.11) | **35.5% (p = 0.0014)** | 38.6% (p = 0.015) |
| OVX regression with variance correction (`ovx_bc`) vs naive | 16.4% (p = 0.008) | 33.3% (p = 0.0001) | 28.9% (p < 0.001) |
| `ovx_bc` vs raw OVX | raw is worse (p = 0.014) | tied (p = 0.72) | tied (p = 0.34) |
| `ovx_bc` vs GARCH-t | better (p = 0.021) | better (p = 0.005) | tied (p = 0.95) |

- **The edge comes from OVX itself.** Unadjusted OVX already beats naive and GARCH at 22 days on QLIKE. The fitted regression adds value at 5 days and on MSE(log vol) (at 5 days raw OVX is 15.5% *worse* than naive on MSE(log vol)), and is otherwise tied with raw OVX. This agrees with the risk-premium-adjustment point in Gilder and Tsiaras. WTI shows the same pattern with weaker significance.
- **Gilder and Tsiaras are not contradicted.** "OVX is least accurate" is a ranking among option-implied measures; we never tested a corridor measure (it needs option prices, which are not free). We only show OVX beats historical-volatility forecasts.
- **Agnolucci (2009) points the other way** (time-series beats ATM implied volatility) on a pre-OVX, futures-options sample; sample, loss and adjustment differ. Worth a short discussion, not a conflict to hide.
- **QLIKE favours upward-biased forecasts** (it punishes under-forecasts harder). OVX, a risk-neutral measure, is typically above realised volatility, which may flatter it on QLIKE; MSE(log vol) tells a less rosy story at 5 days. Report both.

## 3. Closest prior work on the deep-residual idea (not oil)
- **arXiv 2610.03019** ("Signal simplification is not predictive simplification: diagnosing residual neural forecasting in short-horizon volatility"): a HAR-style anchor with an LSTM on its residuals did not beat a pure LSTM; the anchor cut residual variance but did not give the network a residual it could usefully correct. Authors and asset not retrieved. **Our residual null so far agrees with it.**
- **arXiv 2608.25369** ("Forecasting global equity volatility: econometric anchoring and cross-market attention"): an econometric anchoring mechanism under a neural model, with cross-market attention (equities). Abstract and results not retrieved.
- Kılıç (FEDS 2025-061, revised): ML sometimes beats HAR, not systematically. Rodikov and Antulov-Fantulin (2022): LSTMs show no substantial gain over econometric volatility models.

**Do not claim novelty for "a neural residual on a HAR anchor".** The defensible novelty is narrower: a leakage-controlled test on oil with an OVX-augmented anchor, an attention-vs-attention-free comparison at matched size, filtered-simulation VaR/ES evaluation, and the leakage audit (the DDP case study). Check the two arXiv papers in full before writing the related-work section.

## 4. Backlog from this check
1. Read in full: Haugom et al. 2014, Gilder and Tsiaras 2020, arXiv 2610.03019, arXiv 2608.25369, Luo et al. 2024 (is OVX one of their channels?).
2. Add a leverage HAR (negative-return term, as in Mazzeu et al.) to the benchmark ladder (cheap).
3. Add a HAR with exogenous volatility channels (stocks, FX) using daily realised proxies; Degiannakis and Filis use intraday data, which we lack. S&P 500 in the FRED panel starts only in 2016, so another free source is needed.
4. Keep raw OVX as a permanent benchmark (done in `experiments/vol_risk_brent.py`).
5. Re-state the success criteria so the stretch claim is "beat the best of HAR + OVX and raw OVX", not "beat HAR".
