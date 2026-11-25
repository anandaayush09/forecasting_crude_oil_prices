# Crude Oil Volatility and Risk Forecasting

This project asks which forecasts of **crude oil volatility** and **tail risk** beat the standard benchmarks, tested out of sample and without look-ahead. It covers Brent and WTI at **5, 22 and 66 trading days**, scores forecasts with **QLIKE** and formal tests, and reports negative results as plainly as positive ones.

Status: research in progress (release `v0.2.0`). The headline result so far is that a **leverage HAR regression with the OVX implied-volatility index** (OVX lagged one day) beats naive, HAR and GARCH forecasts for Brent at 5 days and for most benchmarks at 22 days. No learned model (ridge, LightGBM, neural network, transformer) beats a well-specified HAR + OVX regression yet. The research plan is in [ROADMAP.md](ROADMAP.md).

## Research problem

Price forecasts for oil are hard to improve on the no-change forecast, and a published gain is often a data leak. An earlier thesis in this archive (see Prior work) reported a 26% error reduction at 7 days that disappeared once three features computed from the whole series were removed. This project therefore moves to targets with more predictable structure:

- **Volatility:** the annualised realised volatility over the next 5, 22 and 66 trading days.
- **Risk:** 1-day and 10-day Value at Risk (VaR) and Expected Shortfall (ES) at 1% and 5%.
- **Price:** reported as a secondary result, with a null result allowed.

## Data and provenance

Every series is free and needs no key. Each is aligned to its **release date** so that a value is used only once it was public (`src/oilfc/data/align.py`, lags in `configs/data.yaml`).

| Series | Source | Release lag used |
|---|---|---|
| Brent, WTI spot | EIA (daily, from 1987 and 1986) | 0 days |
| OVX, VIX, 10Y-2Y spread | FRED | 0 days |
| Broad dollar index | FRED | 5 business days |
| NYMEX futures, contracts 1 and 4 | EIA | 0 days, **ends 2024-04-05** |
| US crude stocks | EIA weekly | 6 days |
| Geopolitical risk (GPR), policy uncertainty (EPU) | authors' sites, newspaper-based | 7 and 2 days |

The panel also holds the S&P 500 (FRED keeps about ten years) and CPI (45-day lag). They are not used in the experiments. The built panel has 9,102 rows from 1987-05-20 to 2026-10-06. OVX starts in 2007-05, so every OVX-based test starts in 2012-01.

Data quality handling covered by code and tests: release-date alignment (`tests/test_alignment.py`), the WTI price below zero on 2020-04-20 (removed before taking returns), futures values older than 10 days masked and non-positive futures prices set to missing (`src/oilfc/features/exog.py`), and parsers tested on small fixtures (`tests/test_parsers.py`). There is no exploratory notebook yet. GPR and EPU are the current published files, not real-time vintages, and the downloads are not versioned, so a rebuild can pick up revisions.

**A timing caveat that matters.** OVX for day t is computed at the New York close of t, after the Brent assessment (London) and after the WTI settlement. Using OVX(t) at origin t is therefore slightly more favourable than a forecaster could have had. The headline results use **OVX(t-1)**, and the same-day numbers are kept as the optimistic alignment.

## Method

- **Target:** forward realised volatility, `sqrt(252 * mean(r^2))` over the next `h` returns (`src/oilfc/eval/targets.py`).
- **Evaluation:** expanding window with monthly refits. Training origins stop `h + 1` trading days before the first test origin, so every training target (which looks `h` days ahead) is fully realised before the first test origin and no training target overlaps the test block. There is no further embargo beyond that (`src/oilfc/eval/rolling.py`). Test period 2012-01 to 2025-06 for volatility, 2010-01 to 2025-06 for the price-only risk backtests.
- **Clean holdout:** data from 2025-07-01 are not used in development. Helper functions (`src/oilfc/eval/holdout.py`, unit-tested) trim and check the data, and each experiment script calls them. Nothing forces a new script to do so, and the date comes from `configs/protocol.yaml`, which is still a draft.
- **Loss:** **QLIKE** is primary, because it ranks volatility forecasts correctly even when the realised-volatility proxy is noisy (Patton 2011). MSE of log volatility is reported next to it.
- **Tests:** Diebold-Mariano with the Harvey-Leybourne-Newbold correction (a rectangular window of `h` autocovariances, standard for h-step forecasts), and the 90% Model Confidence Set.
- **Risk backtests:** Kupiec coverage, Christoffersen independence and conditional coverage, an Expected Shortfall exceedance test, and the FZ0 joint loss.

## Models compared

| Family | Models |
|---|---|
| Naive and smoothing | last-window volatility, EWMA |
| HAR family | HAR on log volatility (5, 22, 63 days), leverage HAR |
| Implied volatility | raw OVX, OVX regression, HAR + OVX, leverage HAR + OVX |
| GARCH family | GARCH(1,1)-t, GJR-GARCH(1,1)-t |
| Learned residuals | ridge, LightGBM, a small MLP and a patch transformer, each predicting a shrunk correction on top of HAR + OVX |
| Risk | GARCH and GJR with t errors and with filtered historical simulation, EWMA-normal, plain historical simulation |

Log-linear forecasts use a variance correction (training residual variance only) so they are consistent with QLIKE. It was added after the uncorrected forecasts looked worse on QLIKE than on MSE, so both versions are reported in the tables.

## Findings (development data)

**Brent, leverage HAR + OVX(t-1), QLIKE:**

| | 5 days | 22 days | 66 days |
|---|---|---|---|
| Gain over naive | 16.1% (p < 0.001) | 32.4% (p < 0.001) | 26.3% (p = 0.001) |
| Better than HAR (corrected) | p = 0.002 | p = 0.066 | tied |
| Better than leverage HAR | p < 0.001 | p = 0.011 | tied |
| Better than GARCH-t / GJR-t | p < 0.001 / < 0.001 | p = 0.002 / 0.007 | tied |

- **WTI** shows the same direction at 5 days (22.8% over naive, p < 0.001, better than HAR, leverage HAR and GARCH with p <= 0.001) and a weaker 22-day result (42.3% over naive, p = 0.032, but HAR p = 0.073). At 66 days no model is separable. The Brent result does not replicate cleanly at longer horizons.
- **The OVX-only regression is weaker under strict timing:** 11.9% over naive at 5 days (p = 0.089) and 28.8% at 22 days (p = 0.002, not significantly better than GJR-t).
- **Robustness of the headline** (sub-periods, Holm correction, calibration; `results/tables/robustness_*.csv`). The 5-day result holds in 2012-2019, in 2020-2025 and without 2020 for both assets, and all 10 comparisons survive Holm correction across the 30 headline comparisons (5 benchmarks, 2 assets, 3 horizons). The 22-day Brent result is weaker after 2020: it still beats naive (p = 0.001) but not the other benchmarks (p = 0.065 to 0.19), while without 2020 every p is below 0.001. After Holm correction only the comparisons with naive (adjusted p = 0.0007) and GARCH-t (0.035) survive; WTI at 22 days survives none (smallest adjusted p = 0.26). At 66 days only the comparison with naive separates (Brent adjusted p = 0.02). The forecasts sit above realised volatility on average (the headline by 5% to 19%, as the QLIKE variance correction implies), calibration slopes are 0.67 to 0.90, and a joint test of slope 1 and intercept 0 rejects in every cell. They rank volatility well but are not calibrated in level. As a sensitivity analysis (added after seeing the results, so not the primary test) a Bartlett-weighted variance for the Diebold-Mariano test, which is less noisy at long horizons, gives smaller p-values at 22 days: for Brent, GJR-t now also survives Holm correction (adjusted p = 0.032) while HAR still does not (0.30), and at 66 days nothing changes except the comparison with naive (Brent adjusted p = 0.009).
- **Estimation window (`experiments/vol_window.py`).** Published work that finds no OVX gain (Haukvik et al. 2024) uses a 504-day rolling window, so the headline was refitted with rolling windows of 1,000, 504 and 252 days. At 5 days, HAR + OVX(t-1) still beats HAR on both assets with 1,000 and 504 days (p from 0.005 to 0.032 in all four cells), so the window does not explain a null at short horizons. At 22 and 66 days no rolling window gives a significant gain over HAR and the 90% confidence set keeps HAR in 11 of 12 cells, which is the pattern of a low-power test, not evidence of no gain. The wider claim is weaker with rolling windows: 13 of the 30 headline comparisons survive Holm correction with the expanding window, against 2, 1 and 1 with 1,000, 504 and 252 days. With 252 days and 66-day targets a few origins with very low forecasts dominate the mean QLIKE (Brent, headline 49.2 against naive 0.84), so those cells say nothing about the model. Refitting the log-linear models weekly instead of monthly (`experiments/vol_refit.py`, following Audrino and Chassot 2024) changes no result: gains move by less than 0.5 points and 11 of 18 comparisons are significant at 5% in both settings (8 after Holm).
- **Economic value, exploratory (volatility targeting, `experiments/vol_targeting.py`).** Sizing a spot position to a 25% annual risk target (cap 2, fixed before running) with each model's forecast, the headline model delivers a volatility closest to the target in all four cells (24.5% to 25.0%, against 27% to 28% for the naive forecast, 23% to 25% for the others, and 44% to 49% unmanaged). How much the rolling volatility wanders around the target is lowest for the headline in three cells (2% to 9% below naive) and level with naive on Brent at 5 days. Sharpe ratios are negative for every variant, because spot returns were negative over the period, and carry no information here. This is a risk-management test with no return forecast and no significance test, so it is descriptive only.
- **The loss matters.** QLIKE favours upward-biased forecasts, and OVX typically sits above realised volatility. At 5 days GARCH has a positive QLIKE gain and a negative MSE(log volatility) gain, and the ranking of corrected and uncorrected HAR flips between the two. Both are reported in the tables.
- Unadjusted OVX already beats naive at 22 and 66 days. The edge is the information in OVX, which earlier literature points to, so the contribution is not "OVX helps". See [the literature note](literature/notes/ovx_volatility_check_2026-11-23.md).
- **Risk (Brent):** at 1 day, GARCH and GJR with filtered simulation pass Kupiec and Christoffersen at 1% and 5%. The t versions also pass at 1% and miss the 5% coverage (hit rates about 6.0%). On the joint FZ0 loss, GJR with simulation is not significantly different from GJR with t errors (p = 0.69 at 1%, 0.64 at 5%). At 10 days GJR with simulation passes at 5% and fails at 1%, while plain historical simulation passes 10-day 1% coverage but has by far the worst FZ0. **WTI (`risk_backtests_wti.csv`):** the same backtests are cleaner than for Brent. At 1 day all four GARCH-type models (t errors and filtered simulation, GARCH and GJR) pass Kupiec and Christoffersen at 1% and 5% (all p above 0.10), including the t versions that miss the 5% coverage on Brent. The EWMA normal model and plain historical simulation fail at 1 day. At 10 days the filtered-simulation models pass at both levels, while the t versions pass Christoffersen but miss 5% coverage on Kupiec (p = 0.046). On FZ0, GJR with simulation is not significantly different from the t models (p from 0.11 to 0.83) and beats EWMA normal at all four horizon and level pairs (p from 0.001 to 0.035).
- **Negative results:** learned residual models do not beat HAR + OVX. The one nominal transformer gain (Brent, 5 days, +2.5% against its own anchor, p = 0.014) does not survive a correction for nine comparisons (Holm-adjusted p = 0.13), and added to the monthly HAR + OVX benchmark it is +1.6% (p = 0.13). Adding futures slope, crude stocks, GPR, EPU or VIX improves QLIKE in none of 12 cells, and makes it significantly worse in some (for example VIX for Brent at 22 days). Small MSE gains appear for WTI at 5 days (policy uncertainty and all groups together). The exchange-rate channel (the broad dollar index, its 22-day change and its 22-day volatility, added to the strict-timing headline) also adds nothing: no QLIKE improvement is significant, none survives Holm correction, and at 5 days the volatility input is significantly worse on MSE(log volatility) (p <= 0.019 in four cells). Its large QLIKE changes at 66 days (+11% to +16%, p > 0.3) come with worse MSE and are not credible. The S&P 500 could not be tested because FRED holds it only from 2016.

Full tables: [results/tables/SUMMARY.md](results/tables/SUMMARY.md). Decisions and caveats: [ROADMAP.md](ROADMAP.md), [docs/adr/](docs/adr/).

## Limitations

- **Multiple testing is corrected only for the 30 headline comparisons** (Holm). The wider model ladder, the new-data and residual tables are not corrected, so treat p-values near 0.05 there as suggestive.
- **Few independent windows at long horizons.** At 66 days the 3,350 origins contain about 50 non-overlapping windows, and the Diebold-Mariano variance uses 66 unweighted autocovariances, which is noisy.
- **One development window of 13.5 years,** a few regimes. Sub-period and ex-2020 results exist for the headline model only. The first 2012 refit has about 1,150 OVX rows.
- **The evidence depends on the expanding window.** With rolling estimation windows (1,000, 504, 252 days) the 5-day gain remains but most other comparisons lose significance after Holm correction (see Findings). The expanding window uses all history since 1987, which is also how the live test is fitted.
- **The headline specification was refined after seeing results** (variance correction, leverage terms, strict OVX timing). All versions are in the tables, but the choice is a forking path.
- **The holdout from 2025-07 was seen in aggregate** by an early run, so only the planned live forward test is a pristine test.
- **Association, not causation.** OVX is a forward-looking market measure, so its link to later volatility is mechanical. The only economic-value check is the exploratory volatility-targeting exercise above, and hedging cost has not been measured.
- **10-day backtests have 391 non-overlapping periods,** which gives low power at the 1% level.
- Futures data end in April 2024, and GPR and EPU are not real-time vintages.

## Repository layout

```
src/oilfc/data/        downloaders and release-date alignment
src/oilfc/features/    daily channels, exogenous features
src/oilfc/eval/        targets, losses, tests, rolling splits, risk, holdout, calibration, Holm
src/oilfc/models/      HAR, OVX and GARCH baselines, residual models
src/oilfc/live/        frozen live forecasts and the scorer
src/oilfc/api/         read-only FastAPI over the live forecasts (docs/api_spec.md)
experiments/           scripts that write results/tables/*.csv
live/                  frozen live forecasts, manifest and scores (see live/README.md)
notebooks/, kaggle/    Kaggle GPU notebook and bundle builder
results/tables/        generated tables (SUMMARY.md is the readable one)
configs/               series and release lags, protocol draft
docs/, literature/     decisions, task cards, free-resource register, literature matrix
tests/                 unit tests (174 pass, 22 are expected failures for unfinished task cards)
prior_work/            reference notes on the earlier thesis
```

## Reproduce

Python 3.12 (the development environment used 3.12.0rc3).

```bash
python -m venv .venv
.venv/Scripts/activate                       # Windows; on Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dl.txt -r requirements-dev.txt
PYTHONPATH=src python -m oilfc.data.build    # add --offline to reuse cached raw files
python -m pytest -q
python experiments/vol_risk_brent.py ladder-only   # Brent ladders (about 10 minutes)
python experiments/vol_risk_brent.py wti ladder-only
python experiments/vol_risk_brent.py wti risk-only        # WTI VaR and ES backtests
python experiments/vol_risk_brent.py risk-only
python experiments/vol_residual_brent.py     # ridge, LightGBM, MLP residuals
python experiments/vol_exog.py               # new data groups
python experiments/vol_ovx_timing.py         # OVX(t) versus OVX(t-1)
python experiments/vol_robustness.py         # sub-periods, Holm correction, calibration (about 10 minutes)
python experiments/vol_window.py             # rolling-window check (about 15 minutes)
python experiments/vol_refit.py              # weekly versus monthly refit (a few minutes)
python experiments/vol_targeting.py          # exploratory volatility targeting
python experiments/vol_fx.py                 # dollar-index channel
python experiments/residual_vs_monthly.py kaggle_residual_monthly   # needs results/kaggle_residual_monthly/ (committed)
python experiments/summarise_vol.py          # writes results/tables/SUMMARY.md
python experiments/live_run.py && python experiments/live_score.py   # live forecasts (the daily workflow does this)
python experiments/holdout_final.py         # dry run only: checks the freeze, evaluates nothing
```

The transformer experiment runs on a free Kaggle GPU: see [docs/kaggle_transformer_residual.md](docs/kaggle_transformer_residual.md). Its outputs are committed: `results/kaggle_residual_monthly/` (monthly anchor, the reported run) and `results/kaggle_residual/` (the first run with a yearly anchor). Seeds: ridge is deterministic, LightGBM and the MLP use `random_state=0`, the transformer averages seeds 0 to 4. Lint with `python -m ruff check . && python -m ruff format --check .`.

Verified on Windows: the test suite, lint, the offline data build and every experiment script. Verified on GitHub Actions (Ubuntu, Python 3.12, 2026-11-24, run 45411467186): a fresh install of `requirements.txt` and `requirements-dev.txt`, lint, the format check and the test suite (the torch tests are skipped there because CI installs no deep-learning packages). Not verified: a macOS run, a fresh install of `requirements-dl.txt`, and the data build and experiment scripts on Linux (CI does not run them, they need the network).

## Project stages

- `v0.1.0`: first working volatility and risk benchmark (Brent and WTI ladders, filtered-simulation VaR/ES, residual models, evaluation library with holdout helpers).
- `v0.2.0`: the headline result stated under strict OVX timing and checked for robustness (sub-periods, Holm correction, Bartlett variance, calibration, volatility targeting), a monthly-anchor rerun of the learned residuals (null), more data tests (futures, inventories, GPR, EPU, VIX, dollar index: null), leakage guards for every feature builder, the live forward test with a daily GitHub Actions run, a read-only API, and a green CI run.
- `protocol-frozen` (tag): the evaluation protocol in `configs/protocol.yaml` and ADR 0003, fixed before the holdout run and the live forward test.
- The earlier thesis work this project grew out of is kept outside the repository as a reference, so there is no separate original-code checkpoint: the first tag is `v0.1.0`.

## Live forward test

Because the clean holdout was seen in aggregate by an early run, the only pristine test of the headline is a forward one. Since 2026-11-24, `experiments/live_run.py` writes frozen, SHA-256-stamped forecasts of volatility at 5, 22 and 66 days for Brent and WTI from six models, and `experiments/live_score.py` scores them once the windows have ended. The first forecasts have origin 2026-11-17, and a GitHub Actions workflow now writes and commits new ones automatically. Nothing is scored yet, and a significance test is not meaningful before roughly 60 origins per horizon. How it works and what it can and cannot show: [live/README.md](live/README.md). A read-only API serves the frozen forecasts and scores (`python -m uvicorn oilfc.api.app:app`, spec in [docs/api_spec.md](docs/api_spec.md)).

## Prior work

The earlier thesis on WTI price forecasting at 1, 7, 90 and 180 days is archived with notes in `prior_work/`. It is reference only: its 7-day gain came from look-ahead features, and nothing from it is used here except lessons. The free-resource policy is in [docs/FREE_RESOURCES.md](docs/FREE_RESOURCES.md).

## Licence

MIT, see [LICENSE](LICENSE).
