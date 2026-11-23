# Research Roadmap: Multi-Horizon Crude Oil Price Forecasting with Transformers and Agentic AI

## Context

The goal is publication-level work that beats existing benchmarks for crude oil price forecasting across horizons.

**Your choices:**
- Venues: a Q1 forecasting/energy journal **and** an ML workshop, with no publication fees (subscription route, arXiv/SSRN preprints, free workshop submission).
- Horizons: daily (1–66 trading days) and monthly (1–24 months).
- Timeline and compute: about 6 months (week 1 = 19 Oct 2026, ending mid-Apr 2027) on a single GPU (Colab/Kaggle).
- Agentic AI plays two roles: (a) **a forecasting component**, the main focus of the workshop paper and one evaluated module in the journal paper, and (b) **research infrastructure**, which supports the work but is not claimed as a contribution.
- **Cost constraint: everything is free-tier.** No paid APIs, no paid data, no paywalled-only papers, no card required. The register of every resource, its limit and its fallback is `docs/FREE_RESOURCES.md`.
- LLMs: open-weight models on free Kaggle/Colab GPUs for all backtests; a free-tier API (Gemini or Groq via LiteLLM, with failover) only for the live test.

The project folder is empty, so everything starts from scratch.

**Gaps in the field this work targets:**
1. **Weak benchmarks.** DL oil papers rarely compare against the no-change or futures forecasts, which the econometric literature shows are hard to beat (Alquist, Kilian & Vigfusson 2013; Baumeister & Kilian 2012, 2015).
2. **Look-ahead leakage.** Decomposition hybrids (EMD/VMD-LSTM) usually decompose the full series before splitting. General critiques exist (arXiv 2509.15394, 2512.06932), but there is **no audit of the oil literature**.
3. **Untested transformers.** No study compares modern transformers (PatchTST, iTransformer, TimeXer) and TSFMs (Chronos-2, TimesFM 2.5, Moirai 2, TiRex) on crude oil with exogenous drivers, against econometric benchmarks and with formal tests. The closest analogue covers agricultural commodities (arXiv 2601.06371).
4. **Agentic forecasting is new, and its leakage is unaddressed.** Recent frameworks (TimeCopilot 2509.00616, TimeSeriesScientist 2510.01538, KairosAgent 2605.30002, last-mile agents 2606.02497) have not been applied to oil. **LLM look-ahead bias** (memorized history) makes most historical LLM backtests unreliable (Look-Ahead-Bench 2601.13770; DatedGPT 2603.11838; Chronologically Consistent LLMs 2502.21206).
5. **Structural breaks** (2014–16, COVID with negative WTI, 2022, the 2025 shocks) break models, and point forecasts lack calibrated uncertainty.

**Being honest about "beating benchmarks":** daily 1–5 day MSPE gains over the random walk will be tiny. The realistic wins are at monthly 1–6 month horizons, in high-volatility regimes, in directional accuracy and in probabilistic accuracy (CRPS and calibration). Gains are claimed only where formal tests support them.

**Update 2026-11-23 (week 6): the main claim moves to volatility and risk.** The DDP re-run showed that its apparent 7-day price gain (26%) came from look-ahead features and falls to 0.2-0.6% (not significant) without them. An audit of the archive found no honest DDP price model that beats the naive forecast. A quick read-only check, still to be reproduced in project code, found a real edge elsewhere: the OVX implied-volatility index forecasts 21-day realised Brent volatility about 33% better than the naive forecast and about 24% better than the standard HAR model (Diebold-Mariano p < 0.001, 2012-2026). So:
- The **headline claim** is leakage-free volatility, risk (VaR/ES) and density forecasting. Deep, transformer and AR-MoE models predict a shrunk residual on top of the HAR + OVX anchor.
- **Price point forecasts** are reported at every horizon as the secondary result. A null is acceptable and is stated as one.
- The **leakage audit** (DDP 26% to 0.2%) is a paper section.
Decision record: `docs/adr/0002-volatility-as-main-target.md`. Task cards: `docs/tasks/V1-V6_*.md`.

---

## 0. Prior work (DDP thesis) and how this project builds on it

The earlier DDP thesis (WTI at 1/7/90/180 days, Track A "leaky paper reproduction" vs Track B "honest walk-forward") is archived at `prior_work/ddp` (moved into this folder; the old `C:\Users\anand\Documents\DDP1` path is a link to it). A summary with reusable pieces and pitfalls is in `prior_work/DDP_KNOWLEDGE_BASE.md`. This project starts fresh; nothing is copied into `src/` unless you decide to port it.

- **RQ2 head start:** DDP already contains a leaky-vs-honest comparison (Track A vs B) and causal rolling-VMD code. Its leakage premium was planned but never reported, and Track B itself still leaks (no purge gap, full-series winsorization and HMM regimes). This project must do the audit properly and report the premium.
- **Re-run result (2026-11-23):** removing the three whole-series VMD columns from the 7-day feature set turned Informer / TCN-BiLSTM / TCN gains of 27.0 / 26.0 / 24.7% into 0.2 / 0.4 / 0.6% (Diebold-Mariano p 0.65 / 0.32 / 0.19) and direction accuracy from 72-75% to about 50%. Evidence: `prior_work/ddp/resume-numbers-justification.md`, `prior_work/ddp/rerun/`. The DDP 90d and 180d "wins" used the same columns and are unverified.
- **Internal baseline row:** DDP Track B results (naive, Informer, TFT, PatchTST, etc.) become a reference row in the benchmark tables, labelled as produced under the old protocol.
- **Horizons:** DDP used 1/7/90/180 calendar days. Report those in an appendix for continuity with the thesis.
- **Motivation for new signals:** the thesis concludes that long horizons lack information, not model capacity. That motivates the term structure, fundamentals and agentic news signals here.
- **Do not repeat:** captions that call a 70/30 split "walk-forward"; audit scripts that check files but not leakage; empty Diebold-Mariano tables; unverified citations; models that silently collapse to the naive forecast.

---

## 1. Research questions and contributions

| RQ | Question | Contribution |
|---|---|---|
| RQ1 | **(reframed)** Under a leakage-free, rolling-origin protocol, which forecasts of Brent realised volatility (5, 22 and 66 trading days) beat the naive, GARCH and HAR benchmarks, and does anything beat HAR + OVX? Price forecasts vs the no-change benchmark are the secondary result | Rigorous volatility benchmark plus a price-forecast null or gain, with formal tests |
| RQ2 | How much of the reported gain of decomposition hybrids comes from leakage? | Literature audit, the DDP case study (26% to 0.2%) and automated leak tests |
| RQ3 | **(reframed)** Does AR-MoE, a regime-gated, shrunk residual on the HAR + OVX anchor, add significant gains, mainly in high-volatility regimes? | **AR-MoE Transformer** (§2) |
| RQ4 | For oil, does attention (temporal and exogenous) add value over attention-free backbones of equal capacity, on the residual task? | Controlled transformer study that answers the DLinear critique (§3) |
| RQ5 | **(narrowed)** Does point-in-time event intensity (news, OPEC, geopolitical risk) improve volatility forecasts around events, and how much of an LLM's apparent value is look-ahead bias? | **OilAgent** M1 first; M2-M4 only if time allows (§4) |
| RQ6 | **(reframed)** Are 1- and 10-day VaR and Expected Shortfall, and 22-day densities, calibrated through breaks, and better than RW + GARCH-t? | Calibrated risk forecasts; online conformal calibration |

**Pre-registered success criteria (revised 2026-11-23, before the week-6 freeze):**
- **Volatility, expected (preliminary evidence exists):** QLIKE and MSE(log vol) ratios below 1 against naive, GARCH-t and HAR, with Diebold-Mariano (HLN) p < 0.05 at 2 or more of the 3 horizons.
- **Volatility, stretch (the deep-learning claim):** a residual model inside the 90% Model Confidence Set on QLIKE at 2 or more horizons, and beating HAR + OVX with p < 0.05 at 1 or more horizon, or in the top-volatility tercile.
- **Risk:** 1% and 5% VaR pass Kupiec and Christoffersen; Expected Shortfall passes its backtest; FZ0 joint loss below GARCH-t and historical simulation.
- **Density:** CRPS better than a RW + GARCH-t density.
- **Price:** MSPE ratio vs no-change reported at all horizons with Clark-West tests; a null result is stated as one.
- **Dropped as headline criteria:** multi-horizon SPA, Pesaran-Timmermann direction and certainty-equivalent return become secondary price-forecast checks.

**Priorities (MoSCoW), to protect the 6-month window:**
- **Must:** benchmark, leakage audit, AR-MoE Transformer, the transformer study, and LLM event features as covariates (agent mode M1).
- **Should:** agent adjudication (M2/M3), memorization probes, live forward test.
- **Could:** chronological-vintage LLM comparison, domain pretraining on an energy-commodity panel.
- **Won't, in this window:** RL fine-tuning of the agent.

---

## 2. Proposed model: AR-MoE Transformer (Anchor-Residual, Regime-routed Mixture-of-Experts)

**Update 2026-11-23:** for the headline volatility task the anchor is **HAR + OVX** (not the no-change price forecast), and the model predicts a shrunk log-volatility residual; for the secondary price task the anchor stays no-change or futures-implied.

The design principle is that the random walk is hard to beat, so the model **learns a state-dependent correction on top of a strong anchor**. Its backbone is an exogenous-aware patch transformer derived from TimeXer.

**Inputs at origin t:**
- Endogenous lookback: log price over L = 128–512 trading days.
- Exogenous series, each at its native frequency, forward-filled to its *release date*, with a "release age" channel.
- State vector s_t: realized volatility, OVX, futures slope, GPR, plus agent event intensity from M1.
- Horizon set H.

**Architecture:**
1. **Anchor:** ŷᵃ(h) = no-change forecast, or the futures-implied forecast where a matching maturity exists (EIA NYMEX contracts 1–4).
2. **Endogenous tokens:** RevIN (reversible instance normalization) → patches (length P ∈ {5, 10, 22}) → linear embedding plus RoPE or learned positions.
3. **Exogenous tokens:** each covariate's whole lookback becomes one variate token (iTransformer style), plus a type embedding (financial / fundamental / uncertainty / text-event). This handles mixed frequencies and missing values naturally.
4. **Global token** (TimeXer) bridges the exogenous information into the endogenous patches.
5. **Encoder (2–3 layers, d_model 64–128, 4 heads):** patch self-attention, then cross-attention from the global and endogenous tokens to the exogenous tokens, then an **MoE feed-forward layer**. The FFN has K = 4 experts with top-2 routing, the router sees [token, s_t], and a load-balancing loss keeps experts in use. The experts specialize by regime.
6. **Multi-horizon decoder:** learned horizon-query tokens q_h (an embedding of h plus the target-date calendar) cross-attend to the encoder. This produces all horizons jointly and keeps them consistent.
7. **Heads:**
   - Location residual μ_h, gated by the shrinkage gate α_h = σ(w·[q_h, s_t]) ∈ [0, 1].
   - A separate scale/shape head with non-crossing quantiles. Uncertainty does not collapse when α → 0.
   - Forecast: log ŷ_h^τ = log ŷᵃ(h) + α_h·μ_h + spread_h^τ.
8. **Loss:** pinball loss over the quantile grid + λ_α·mean(α) (shrinkage prior: deviate from the anchor only with evidence) + load-balancing loss + an optional directional auxiliary loss.
9. **Calibration:** Adaptive Conformal Inference (Gibbs & Candès 2021) runs online on the test stream.
10. **Training:** pooled Brent + WTI (a global model). Optionally, pretrain first on an energy-commodity panel (Brent, WTI, Dubai, RBOB, heating oil, Henry Hub), then fine-tune.
11. **Monthly variant:** monthly patches plus aggregated daily tokens (multi-resolution). Compare with temporally aggregated daily forecasts to test cross-frequency consistency.
12. **Variant:** frozen TSFM encoder (Chronos-2 or TiRex embeddings) with the AR-MoE head.

**Novelty check in Phase 0:** search for "residual over naive/futures + learned shrinkage" and "regime-routed MoE forecasting" so the contribution is framed correctly.

---

## 3. Transformer deep dive

### 3.1 Taxonomy and role in the study

| Family | Models | Key mechanism | Role |
|---|---|---|---|
| Legacy long-horizon | Informer, Autoformer, FEDformer | Sparse, auto-correlation and frequency attention | Baselines, since oil papers use them; expected to lose to DLinear (Zeng et al. 2023) |
| Token redesign | PatchTST, Crossformer, iTransformer | Patches as tokens, channel independence, variates as tokens | Core baselines; tokenization ablations |
| Exogenous-aware | **TimeXer**, **TFT** | Endogenous patches + exogenous cross-attention; variable selection + known-future inputs + quantiles | Strongest DL baselines; TimeXer is the origin of the AR-MoE backbone |
| Foundation transformers | Chronos-Bolt, **Chronos-2** (group attention, covariates), TimesFM 2.5, Moirai 2, Sundial, Toto, Time-MoE, Kronos (finance) | Pretrained, zero-shot, in-context covariates | Zero-shot + fine-tuned baselines; frozen-encoder variant |
| Attention-free controls | DLinear/NLinear, TSMixer(x), TiDE, N-HiTS, TiRex (xLSTM), S-Mamba (SSM) | Linear, MLP, recurrent, state-space | Controls that test whether attention adds value (RQ4) |

### 3.2 Transformer-specific experiments (RQ4)

- **Attention vs no attention:** at matched parameter count, swap the attention blocks for MLP-Mixer or linear blocks in AR-MoE.
- **Tokenization:** point vs patch vs variate tokens; patch length and stride; channel independence vs channel mixing.
- **Exogenous pathway:** cross-attention vs concatenation vs none; drop one covariate block at a time.
- **Lookback scaling:** L ∈ {64, 128, 256, 512}. Transformers claim to exploit long context; test whether that holds for oil.
- **Normalization under shift:** RevIN vs standard scaling vs none, compared specifically across break periods.
- **MoE:** K ∈ {1, 2, 4, 8}; routing on state vs on token only; expert usage over time vs known events.
- **TSFM adaptation:** zero-shot vs linear probe vs LoRA/adapter vs full fine-tune (small models); Chronos-2 native covariates vs a residual model on top.
- **Domain pretraining:** a small transformer pretrained on the energy panel vs a general TSFM, at equal fine-tuning budget.
- **Interpretability:** cross-attention maps over exogenous variables by horizon and regime, triangulated with integrated gradients (attention ≠ explanation); MoE routing timelines.
- **Efficiency:** parameters, training minutes per refit on a T4/L4, inference latency, reported for every model.

### 3.3 Training recipe for small data on Colab

- Mixed precision, small widths (d_model ≤ 128), dropout 0.1–0.3, weight decay, early stopping on a rolling validation block.
- 5-seed ensembles, with optional stochastic weight averaging.
- Optuna with a capped budget, run on the validation period only.
- Refit yearly over about 15 test years: roughly minutes per refit, which is feasible on a single GPU.

---

## 4. Agentic AI, part A: OilAgent forecasting system (RQ5; workshop focus, journal module)

### 4.1 Architecture: a multi-agent graph over a point-in-time tool layer

```
  Point-in-time store (documents and data with timestamp <= t, enforced server-side)
        │
  Curator ──► Event Extractor ──► structured events (pydantic JSON) ──► [M1] covariates into AR-MoE
        │                                   │
        └──► Quant Analyst ── MCP tools: get_prices / get_futures_curve / get_inventory_report /
                                         search_news / run_model / get_regime_state / get_past_performance
                                            │
             Adjudicator ([M2] model weights and shrinkage, or [M3] bounded adjustment) ◄── Memory (resolved past forecasts only)
                                            │
             Critic (temporal validity, bounds, cites real event IDs) ──► final quantile forecast + rationale
```

- **Event schema:**
  - Type: supply disruption / OPEC+ decision / sanctions / conflict / demand shock / inventory surprise / macro-policy / weather.
  - Also: region, direction, magnitude, expected duration, confidence, source document IDs, timestamp.
- **Leakage prevention lives in infrastructure, not prompts.** Every MCP tool takes `as_of` and filters server-side by *publication* timestamp. The agent cannot see the future even if it tries.
- **Bounded adjustment (M3):** δ is capped at |δ| ≤ κ·IQR of the model's forecast, so hallucinated extremes cannot do damage. Every adjustment must cite event IDs, and the Critic vetoes uncited or out-of-bounds outputs.
- **Reproducibility:** temperature 0, fixed seeds, and **every LLM call cached by prompt hash**, so historical backtests replay deterministically.

### 4.2 Integration modes compared (the core experiment)

| Mode | Agent role | Scale |
|---|---|---|
| M0 | None (AR-MoE alone) | Full history |
| M1 | Feature engineer: daily event features feed AR-MoE | Full 2015–2026 backtest with an open-weight model |
| M2 | Router/combiner: picks model weights and shrinkage per date | Clean windows + live test |
| M3 | Last-mile adjuster with bounded δ (cf. 2606.02497) | Clean windows + live test |
| M4 | LLM forecasts the number directly | Baseline; expected to be overconfident |

### 4.3 LLM look-ahead control (a contribution in itself)

1. **Chronological vintages:** event extraction with ChronoGPT-Instruct (vintages 2015/2020/2024) and DatedGPT (annual 2013–2024). Each forecast date uses a vintage whose cutoff precedes it. Compare against a current open-weight model to *measure* the bias.
2. **Memorization probes:**
   - Price recall on specific dates.
   - Entity and date anonymization ("a major Gulf producer", shifted dates), comparing outputs with and without masking.
   - Performance decay before vs after each model's cutoff (Look-Ahead-Bench style).
3. **Post-cutoff windows:** general LLMs are evaluated only after their documented cutoff.
4. **Live forward test:** a frozen v1 system runs from week 12 to week 24 (~60 trading days, h ∈ {1, 5, 10, 22}). A scheduled job logs the forecasts, and they are **hash-committed to git before the outcomes exist**. It is explicitly underpowered, so report descriptive statistics plus DM where feasible.
5. **Reporting rule:** historical M2/M3 results from general LLMs are labelled "upper bound, possibly contaminated". Claims rest on vintages, post-cutoff windows and the live test.

### 4.4 Models and costs

- **Extraction (M1):** a 7–9B open-weight instruct model (e.g., Qwen3-8B / Llama-3.1-8B), 4-bit, served with vLLM on Colab. GDELT 2.0 runs from 2015, giving about 4,000 daily prompts (top-N oil-relevant headlines per day plus EIA/OPEC documents), which is feasible.
- **Adjudicator/critic:** the open-weight model for clean historical windows; a free-tier API (Gemini or Groq via LiteLLM, with failover, exact model id logged) for the live test (~60 days × a few calls/day, low cost).
- **Comparators:** TimeCopilot (open source) run as an off-the-shelf agentic baseline; TemporalBench (2602.13272) as a reference evaluation design.

### 4.5 Agent-specific evaluation

- All §8 forecast metrics.
- Adjustment hit rate: the share of adjustments that reduced error.
- Event study: errors around OPEC+ meetings, WPSR releases, sanctions and conflict onsets.
- Rationale faithfulness: human audit of 100 sampled rationales. Does each cited event exist with timestamp ≤ t, and does its direction match the adjustment?
- Ablations: no retrieval, no memory, no critic; model size (vintage 1.3B / 8B / API).
- Cost per forecast, latency, and rerun variance.

---

## 5. Agentic AI, part B: research infrastructure (supports the work, not claimed)

| Agent role | Does | Guardrail |
|---|---|---|
| Data steward | Scheduled refresh; schema, gap, outlier and release-timestamp validation; data-card diffs | Raw data immutable and versioned |
| Experiment runner | Generates Hydra sweep configs, launches Colab/Kaggle runs, logs to MLflow | Runs only through `make`/CLI entry points, never ad hoc |
| Leakage auditor | Runs the pytest leakage suite and the statistical tests; **auto-flags any model beating the random walk by > 20% MSPE at h = 1 as likely leakage** | CI blocks merges on failure |
| Literature scout | Weekly arXiv/Semantic Scholar digests, screening against inclusion criteria into `lit_matrix.csv` | Every extracted field human-verified (hallucination risk) |
| Live-test operator | Daily scheduled job: pull data, run frozen v1, commit forecast hash before the close | Append-only log |
| Reporting | Builds tables and figures *only from result files*; no hand-typed numbers | Figures regenerated in CI |

- **Test-set lock:** test-split access is gated on a `protocol-frozen` git tag. A guard in the evaluation code (covered by a pytest test and CI) refuses to load the test split unless that tag exists. Protocol changes need human sign-off.
- **Leakage test suite (pytest):**
  - `test_no_future_rows`
  - `test_scaler_fit_window`
  - `test_release_alignment`
  - `test_decomposition_causal`
  - `test_tool_as_of_filter`
  - `test_sequence_after_split`
- **Disclosure:** include an AI-use statement per Elsevier/IJF policy.

---

## 6. Data and information sets

| Block | Series (frequency) | Source |
|---|---|---|
| Targets | Brent spot (daily, 1987–), WTI spot (daily, 1986–); monthly real Brent/WTI/RAC deflated by US CPI | EIA, FRED |
| Term structure | NYMEX WTI contracts 1–4 (daily); no full curve (paid Datastream/Bloomberg data is out of scope) | EIA |
| Financial | OVX, VIX, broad dollar index, S&P 500, 10Y–2Y spread, gold, Henry Hub | FRED / CBOE |
| Fundamentals | WPSR stocks and production (weekly); world production, OECD stocks (monthly); RBOB and heating oil spreads | EIA, JODI |
| Global demand | Kilian real activity index, Baumeister–Korobilis–Lee indicators, OECD+6 industrial production | FRED / authors' sites |
| Uncertainty and positioning | Daily GPR, daily EPU, CFTC managed-money positions | authors' sites, CFTC |
| **Text corpus** | GDELT 2.0 events/GKG + headlines (2015–); EIA WPSR/STEO text; OPEC press releases and MOMR summaries; all timestamped by publication | GDELT, EIA, OPEC |

**Alignment rules:**
- Align every series to its publication timestamp.
- Use real-time vintages (ALFRED / Baumeister–Kilian) for revised series.
- Align Brent (London close) and WTI (New York close) by close time.
- WTI went negative on 2020-04-20, so Brent is the primary log target and WTI a robustness check.

**Small monthly sample:** use a panel of Brent, WTI and RAC, TSFMs, and cross-frequency aggregation.

---

## 7. Benchmark suite

- **Volatility ladder (headline):** naive realised vol (last h days), EWMA (lambda 0.94), GARCH(1,1)-t, GJR-GARCH-t, log-HAR (1/5/22 or 63-day), OVX-only regression, **HAR + OVX (the anchor any deep model must beat)**.
- **Naive/statistical:** no-change (the denominator for every ratio), drift, historical mean, AutoARIMA, ETS, Theta; RW+GARCH-t (density).
- **Econometric:** futures-based forecasts (Alquist & Kilian 2010), product-spread models, monthly VAR/BVAR, Baumeister–Kilian forecast combination.
- **ML:** Ridge/LASSO/Elastic Net, LightGBM, Random Forest.
- **Deep learning and transformers:** the full §3.1 table, plus LSTM/GRU/TCN (common in oil papers).
- **Decomposition hybrids:** VMD-LSTM and CEEMDAN-LSTM, each leaky vs causal (RQ2).
- **Foundation models:** zero-shot and fine-tuned, with a **post-release clean holdout** (e.g., 2025-07 onward) to control pretraining contamination.
- **Agentic:** M1–M4 and TimeCopilot.

---

## 8. Evaluation protocol (frozen in week 6, including the agent protocol and the live-test design)

- **Design:** expanding-window, rolling-origin evaluation. Train from 1987, use a validation block for tuning, and run the out-of-sample test from about 2010 to the latest data. The text-based modes are tested 2015+. Refit deep models yearly. Never tune on the test set.
- **Volatility targets (headline):** forward realised volatility from daily squared Brent returns over t+1..t+h, h in {5, 22, 66}. WTI is a robustness check (OVX is built from WTI options; WTI went negative in 2020). Free intraday data is not available.
- **Test start:** 2012 for anything using OVX (it starts 2007-05); about 2010 for price-only models.
- **Gap:** at least h trading days between the last training target and the first test origin (no overlap).
- **Volatility losses:** QLIKE (primary), MSE(log vol), MSE(variance). **Risk backtests:** Kupiec, Christoffersen, an Expected Shortfall test and FZ0 loss.
- **Clean holdout:** the last 12 months, untouched until the final run.
- **Horizons:** daily h ∈ {1, 5, 10, 22, 66} (cumulative log returns, also reported on price levels); monthly h ∈ {1, 3, 6, 12, 24} (log real monthly-average prices).
- **Sub-periods:** calm vs crisis (2014–16, 2020, 2022, 2025); volatility terciles.
- **Metrics:**
  - Point: MSPE and MAE ratios vs random walk, MASE.
  - Direction: success ratio.
  - Probabilistic: CRPS, pinball loss, coverage and width, Winkler score, PIT histograms.
- **Statistical tests:** DM with the HLN correction and HAC variance; Clark–West; Giacomini–White; MCS (`arch`); **multi-horizon SPA (Quaedvlieg 2021)**; Giacomini–Rossi fluctuation test; Pesaran–Timmermann.
- **Economic value:** sign-based trading with transaction costs (Sharpe ratio, maximum drawdown, certainty-equivalent return); a monthly hedging decision.
- **Reproducibility:** ≥ 5 seeds reporting mean ± std; LLM call cache; scripts that download the raw data.

---

## 9. Timeline (26 weeks; week 1 = 19 Oct 2026)

| Weeks | Phase | Key deliverables |
|---|---|---|
| 1–3 | **0. Literature and infrastructure bootstrap** | Lit matrix of ~100–140 papers (oil econometrics/DL, transformers, agentic forecasting, LLM look-ahead); gap statement; repo, CI leakage suite, MLflow, literature-scout agent |
| 2–6 | **1. Data, text corpus and tool layer** | Price and covariate pipeline; GDELT/EIA/OPEC corpus with publication timestamps; point-in-time store; MCP tool server v0 with `as_of` enforcement; EDA (unit roots, Bai–Perron breaks) |
| 5–8 | **2. Protocol and baselines** | Protocol (including agent and live test) **frozen and tagged in week 6**; rolling engine; tests module; statistical and econometric baseline tables |
| 7–10 | **3. Transformer, TSFM and leakage experiments** | §3.1 baselines, TSFM zero-shot and fine-tuning, leaky-vs-causal decomposition. **Fallback:** publishable as a benchmark/audit paper on its own |
| 9–17 | **4a. AR-MoE Transformer** | Development on validation only; §3.2 ablations; **frozen in week 17** |
| 10–17 | **4b. OilAgent** | Weeks 10–12: event extraction (M1) over the corpus. **Week 12: freeze v1, start the live test.** Weeks 13–16: adjudicator and critic (M2/M3), memorization probes, vintage comparison. Frozen in week 17 |
| 12–24 | **Live forward test** | Daily scheduled run of frozen v1 with hash-committed forecasts |
| 16–20 | **5. Final evaluation** | Single test run; all tests; regime and event studies; economic value; interpretability; robustness (Brent vs WTI, test start, lookback, refit frequency) |
| 19–26 | **6. Writing and submission** | **Journal** (IJF / Energy Economics): benchmark + leakage audit + AR-MoE Transformer + OilAgent M1 module, submitted around week 26. **Workshop** (non-archival, agentic focus: OilAgent + LLM look-ahead control + live test): target an ICML 2027 workshop (deadlines usually around May; verify). An optional early non-archival ICLR 2027 workshop submission could cover the transformer benchmark. arXiv/SSRN preprint; GitHub + Zenodo DOI |

**Baseline results (2026-11-23). Source of truth: `results/tables/SUMMARY.md`, generated by `experiments/summarise_vol.py` from the CSVs written by `experiments/vol_risk_brent.py` and `experiments/vol_residual_brent.py`. Expanding window, monthly refits, gap = horizon, development data only (test 2012-01 to 2025-06; risk from 2010-01), QLIKE primary.**

**Correction of 2026-11-23 (holdout).** The first baseline runs used data through 2026-09, i.e. they included the clean holdout (2025-07 onward). Every number below was recomputed on development data only; the earlier, larger-sample numbers are superseded. The holdout has now been seen **in aggregate** by the people running this project, so it is no longer a pristine final test; the live forward test (weeks 12-24) is the only fully clean evaluation. A code guard (`oilfc/eval/holdout.py`, tested) now trims every development experiment to before `clean_holdout_start`.

*Volatility ladder, Brent (n = 3,413 / 3,396 / 3,352 origins at 5 / 22 / 66 days).* `ovx_bc` = regression of log volatility on log OVX with a QLIKE-consistent variance correction (training residuals only); `ovx_raw` = OVX as published, no fitted parameters; `lhar_bc` = leverage HAR (Corsi and Reno 2012) with the same correction.

| Brent | 5 days | 22 days | 66 days |
|---|---|---|---|
| `ovx_bc` QLIKE gain vs naive | 15.4% (p = 0.020) | 32.4% (p < 0.001) | 25.6% (p < 0.001) |
| `ovx_raw` (unadjusted) vs naive | 9.9% (p = 0.19) | 34.0% (p = 0.004) | 36.0% (p = 0.032) |
| `ovx_bc` better than HAR (corrected) | p = 0.006 | p = 0.031 | tied (p = 0.82) |
| `ovx_bc` better than leverage HAR | p = 0.10 (not sig.) | p = 0.008 | tied (p = 0.66) |
| `ovx_bc` better than GARCH-t / GJR-t | p = 0.037 / 0.052 | p = 0.005 / 0.037 | tied (p = 0.85 / 0.77) |
| 90% MCS (QLIKE) | `ovx_bc`, `har_ovx_bc`, `lhar_ovx_bc` only | 7 models incl. raw OVX and plain HAR | 11 models (everything but naive and EWMA) |

- **Reading.** OVX-based forecasts are significantly better than the non-OVX benchmarks at 22 days (p <= 0.037 against corrected HAR, leverage HAR, GARCH-t and GJR-t) and mostly at 5 days (leverage HAR p = 0.10 and GJR p = 0.052 are not significant), and tied with them at 66 days. Adding leverage terms to HAR + OVX changes nothing significant. Raw OVX is tied with the fitted versions at 22 and 66 days; the fitted regression helps at 5 days and on MSE(log vol).
- **Caveats.** (1) The variance correction was added after seeing uncorrected results; both are reported. (2) QLIKE favours upward-biased forecasts and OVX is typically above realised volatility, so MSE(log vol) must be reported next to it (at 5 days raw OVX is 18% *worse* than naive on MSE(log vol)). (3) No multiplicity correction across three horizons and many comparisons. (4) One asset and one development period.
- **WTI robustness (`vol_ladder_wti_*d.csv`; negative price of 2020-04-20 removed, so one return spans several days).** Same ranking, much weaker separation: OVX-based models are in the MCS at 5 days (with raw OVX) and beat naive at 5 days (p <= 0.02), but at 22 and 66 days nearly every model, including naive, stays in the MCS and no OVX model is significantly better than the HAR or GARCH variants (p >= 0.07). The Brent result does not replicate cleanly on WTI.
- **Density (22-day return, zero-mean normal scaled by the volatility forecast).** `ovx_bc` has lower CRPS than naive at 5 and 22 days (0.0276 vs 0.0279, p = 0.019; 0.0617 vs 0.0627, p = 0.0001) and than GARCH-t (p = 0.0009, 0.033); at 22 days it is not significantly better than GJR-t (p = 0.058) or leverage HAR (p = 0.090). At 66 days nothing is significant. Effects are about 1-2%.
- **Raw OVX and the literature** (`literature/notes/ovx_volatility_check_2026-11-23.md`): the edge is OVX's own information, already suggested by prior work (Haugom et al. 2014, Gilder and Tsiaras 2020, Degiannakis and Filis 2017), so **the paper cannot claim "OVX helps"**. The closest prior work on neural residuals is not oil (arXiv 2610.03019: HAR + LSTM residual did not help; arXiv 2608.25369: econometric anchoring, equities).
- **Residual models on the HAR + OVX anchor (22 days, shrunk).** Ridge, LightGBM and a small MLP: QLIKE gain vs anchor 4.9% (p = 0.43), 1.5% (p = 0.62), -2.4% (p = 0.54); MSE(log vol) worse for all. All inside the MCS. **No detectable gain from learned residuals; the stretch criterion is not met.**
- **VaR/ES (Brent; 1 day n = 3,917, 10 days non-overlapping n = 391).** 1 day, 1%: GARCH-t, GJR-t and the simulation versions (FHS) all pass Kupiec, Christoffersen and the ES test (hit rates 0.87-1.07%). 1 day, 5%: the t versions fail coverage (hit rates 6.0%, Kupiec p = 0.005 and 0.008) and FHS passes (5.18%, p = 0.60). 10 days: GJR-FHS passes at 5% (hit rate 6.65%, Kupiec p = 0.15, Christoffersen p = 0.29, ES test p = 0.23) and fails at 1% (2.56%, p = 0.0097, ES test passes). Plain historical simulation passes 10-day coverage but has by far the worst FZ0 (e.g. -0.39 vs -1.08 at 1%). EWMA-normal fails everywhere.
- **FZ0 (GJR-FHS vs each model, Diebold-Mariano).** Significantly better than EWMA-normal and historical simulation at every horizon and level (p <= 0.03); tied with GARCH-t and GJR-t at 1 day for 1% (p = 0.69, 0.69) and not distinguishable from GJR-t anywhere (p >= 0.64 at 1 day 5%; 0.11 at 10 days 1%); better than GARCH-FHS at 1 day 5% (p = 0.001) and 10 days 5% (p = 0.043). At 10 days, 1% the gain over GARCH-t and GARCH-FHS is borderline (p = 0.057, 0.057).
- **Transformer residuals on the HAR + OVX anchor (RQ3/RQ4), run on Kaggle (T4/P100 GPU, 5 seeds, 14 yearly refits, development data only; tables in `results/kaggle_residual/`).** Patch transformer (19,009 parameters), flat MLP control (20,881), ridge on the 64-day window; QLIKE gain vs the yearly-refit anchor:

| | Brent 22 days (primary) | Brent 5 days (extra) | WTI 22 days (extra) |
|---|---|---|---|
| transformer | -1.2% (p = 0.63) | **+2.6% (p = 0.009)**, MSE(log vol) +2.3% (p = 0.003) | -0.5% (p = 0.58) |
| MLP (attention-free) | -1.1% (p = 0.57) | +0.4% (p = 0.73) | -2.1% (p = 0.18) |
| ridge | -0.8% (p = 0.56) | +2.3% (p = 0.32) | +1.7% (p = 0.73), MSE(log vol) -8.3% (p = 0.047) |
| transformer vs MLP (DM, QLIKE) | p = 0.99 | p = 0.060 | p = 0.22 |

  - **Reading.** At the primary setting (Brent, 22 days) no learned residual helps and attention makes no difference to the MLP (p = 0.99): a null for RQ3 and RQ4. The one significant cell (Brent 5 days, transformer) is an extra, unplanned run, one of nine model-setting comparisons (a Bonferroni-adjusted p of about 0.08), and the transformer-vs-MLP gap there is borderline (p = 0.060).
  - **Confound found and checked.** The deep models and their anchor are refit yearly, the benchmark ladder monthly. Against the monthly-refit HAR + OVX on the same origins the transformer's 5-day edge is gone (0.4% better, p = 0.72) and it is worse than monthly leverage HAR + OVX (-1.1%, p = 0.77); at 22 days it is 2.4% to 5.8% worse than the monthly benchmarks (p >= 0.40); on WTI it is significantly worse (p = 0.001 vs monthly HAR + OVX). **The apparent 5-day gain is the stale yearly anchor, not the transformer.**
  - **To do before any claim:** refit the anchor monthly while the residual network refits yearly (or give the benchmarks the same yearly refits), then rerun. Until then the honest status is: no evidence that a transformer residual beats a well-specified HAR + OVX.
  - Smoke-test and 5-day/WTI files are in the same folder; the Kaggle runs used the holdout guard (`holdout_start` is in each run's JSON).

**Status against the pre-registered criteria (revised draft, before the freeze):**

| Criterion | Status |
|---|---|
| Volatility, expected: beat naive, GARCH-t and HAR with p < 0.05 at 2+ horizons (QLIKE) | **Met on Brent** at 22 days (all non-OVX benchmarks, p <= 0.037) and at 5 days against naive, corrected HAR and GARCH-t (leverage HAR and GJR-t not significant); tied at 66 days. **Not replicated on WTI** |
| Volatility, stretch: a learned residual inside the MCS and beating HAR + OVX at 1+ horizon | **Not met.** Ridge, LightGBM, MLP and transformer residuals: no gain at 22 days; the one significant 5-day transformer cell disappears against monthly-refit benchmarks (anchor-refit confound). Foundation models not run |
| Risk: 1% and 5% VaR pass Kupiec and Christoffersen; ES test; FZ0 below GARCH-t | **Met at 1 day** (GARCH/GJR with t or simulation at 1%; simulation at 5%). **Partly at 10 days** (GJR-FHS passes 5%, 1% rejects). FZ0 gain over GARCH-t not significant |
| Density: CRPS better than RW + GARCH-t | **Met on Brent at 5 and 22 days** (small effect, about 1-2%) |
| Price: MSPE vs no-change at all horizons, null allowed | Not started in the new protocol (DDP result retracted) |

**Revised schedule from week 6 (2026-11-23), replacing the phase 2-5 rows above where they differ:**

| Weeks | Work |
|---|---|
| 6 | Protocol update and freeze; ADR 0002; task cards V1-V4 |
| 7 | V5-V6. **Gate:** project code reproduces the preliminary table (Brent, 22 days, OVX vs naive about 33%, p < 0.01). **Reached 2026-11-23** in `src/oilfc/` (tasks V1-V3, V5, V6; full-sample numbers, which included the clean holdout and are superseded by the development-only numbers below): OVX 31.0% better than naive (DM p = 0.0008) and 21.8% better than HAR; HAR + OVX 21.1% better than HAR (p = 0.002); HAR alone 11.8% better than naive (p = 0.14, not significant). Brent, 22-day log-vol MSE, test 2012-01 to 2026-09, n = 3,718 |
| 8-10 | GARCH family, MCS, VaR/ES backtests, CRPS. First baseline results table (workshop-ready) |
| 10-14 | Deep residual models and TSFMs on the HAR + OVX anchor; attention vs attention-free (RQ4) |
| 14-18 | AR-MoE regime gate; ablations on new data; event intensity (RQ5, M1) if time allows |
| 18-20 | Leakage-audit section; secondary price result (monthly MSPE vs no-change) |
| 20-22 | Final holdout run; robustness on WTI, sub-periods, excluding 2020 |
| 22-26 | Writing: workshop paper first, then the journal paper |

**Journal outline:**
1. Introduction
2. Related work
3. Data and real-time information
4. AR-MoE Transformer
5. Agentic module (M1)
6. Evaluation design
7. Results: daily and monthly
8. Transformer analysis (RQ4)
9. Leakage audit
10. Regimes and economic value
11. Interpretability
12. Robustness
13. Conclusion

**Workshop outline:**
1. Problem: agentic forecasting under look-ahead bias
2. OilAgent architecture
3. M0–M4 comparison
4. Memorization probes and vintages
5. Live test
6. Limitations

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| No gain at daily h = 1–5 | Expected and reported; the anchor bounds losses; claims rest on monthly and regime gains, probabilistic accuracy, the audit and the agent findings |
| Deep residuals don't beat HAR + OVX | The paper still has three honest results: OVX-based volatility and risk forecasts beat the standard benchmarks by large margins; a rigorous null for deep learning; the leakage audit |
| "OVX predicts realised vol" is already known | Done 2026-11-23: 22 papers in `lit_matrix.csv` and a note (OVX-volatility links are established in the literature). The contribution is the leakage-controlled deep-residual test with an attention-free control, the FHS risk evaluation and the audit, not "OVX helps" and not "a neural residual on HAR" (read arXiv 2610.03019 and 2608.25369 in full first) |
| Preliminary volatility numbers are not project results | Week 7 gate: reproduce them in `src/oilfc/` with tests before any claim |
| Transformers don't beat attention-free controls | Still a publishable answer to RQ4; AR-MoE keeps whichever backbone wins |
| LLM look-ahead contaminates agent results | Vintages, post-cutoff windows, live test, plus the reporting rule in §4.3 |
| Hallucinated events | Pydantic schema, mandatory source IDs, critic veto, bounded δ, human audit |
| Live test too short | Framed as supplementary; daily horizons only; v1 frozen early (week 12) to maximize length |
| Free-tier limits (Kaggle GPU hours, API rate limits) | Small backbones, yearly refits, checkpoint every epoch, 4-bit vLLM, LLM cache, two API providers with failover, API used only for the live test |
| Scope overload (one person, 6 months) | MoSCoW in §1; Phase 3 is a standalone fallback paper; agent M2/M3 can slip to the workshop only |

---

## 11. Tooling and project structure

**Stack:**
- Python 3.11, PyTorch
- Forecasting libraries: Nixtla `neuralforecast` / `statsforecast` / `mlforecast` / `utilsforecast`
- Econometrics: `statsmodels`, `arch`
- Decomposition and conformal: `vmdpy`, `PyEMD`; `MAPIE` or a custom ACI
- TSFM packages: `chronos-forecasting`, `timesfm`, `uni2ts`, `tirex`
- Agents and LLM serving: `vllm`, `langgraph` (or `smolagents`), `pydantic`, MCP Python SDK, LanceDB/Chroma with timestamp-filtered metadata, `sentence-transformers`
- Experiment tooling: Optuna, Hydra, MLflow, pytest + GitHub Actions
- GitHub Actions scheduled workflows (free on public repos) or local cron/Task Scheduler for the daily live job

```
Forecasting Crude Oil Prices/
  ROADMAP.md  README.md  requirements.txt  .gitignore  Makefile
  configs/        data.yaml, protocol.yaml, models/*.yaml, agents/*.yaml
  data/           raw/ interim/ processed/            (gitignored)
  corpus/         text documents with publication timestamps (gitignored)
  literature/     lit_matrix.csv, notes/
  src/oilfc/
    data/         eia.py, fred.py, gpr_epu.py, cftc.py, gdelt.py, align.py
    features/
    eval/         rolling.py, metrics.py, tests.py (DM-HLN, CW, GW, MCS, MH-SPA, GR, PT), conformal.py
    models/       baselines/, econ/, ml/, dl/, transformers/, tsfm/, armoe/
    agents/       graph.py, schemas.py, prompts/, memory.py, llm_cache.py, probes.py
  mcp_server/     point-in-time tools with as_of enforcement
  tests/          leakage guard suite
  live/           forward-test logs + committed forecast hashes
  experiments/    one entry point per phase
  notebooks/      EDA
  results/        tables/, figures/
  paper/          journal/, workshop/
```

---

## 12. Execution on approval (this session)

1. Write this roadmap to `ROADMAP.md` in the project folder.
2. Create the §11 folder skeleton, `.gitignore`, a `requirements.txt` with pinned versions, and a `Makefile` with placeholder targets.
3. Create the `literature/lit_matrix.csv` template. Columns: citation, year, venue, target, frequency, horizons, model family, transformer?, agent/LLM?, RW benchmark?, futures benchmark?, stat tests, decomposition?, decomposition causal?, scaler/split leakage risk, LLM look-ahead control?, reported gain, notes.
4. Add stub leakage tests in `tests/` (marked xfail until implemented) and run `git init`, so protocol and model freezes become tagged commits.

Data download scripts, the evaluation harness, the MCP tool server and the agents follow as Phase 1–4 work in later turns.

## Verification

- **After steps 1–4:** the folder tree matches §11; `ROADMAP.md` renders; `pytest` collects the stub tests (as xfail); requirements resolve (dry run).
- **Project quality gates:**
  - Week 6: protocol tagged.
  - Week 8: the random walk beats most naive DL baselines at h = 1, confirming a leakage-free pipeline.
  - Week 10: the leaky VMD-LSTM's gains disappear when run causally.
  - Week 12: live-test forecasts are hash-committed daily.
  - Week 17: model and agent frozen and tagged before the single test run.

## Key references (verify each in Phase 0)

- **Oil econometrics:** Alquist, Kilian & Vigfusson (2013); Alquist & Kilian (2010, JAE); Baumeister & Kilian (2012, 2015, JBES); Kilian (2009, AER); Degiannakis & Filis (2018, Energy Econ.)
- **Oil ML/text:** Yu, Wang & Lai (2008, Energy Econ.); Li, Shang & Wang (2019, IJF); Bai et al. (2022, IJF); arXiv 2603.11408 (multi-dimensional LLM sentiment for WTI); arXiv 2410.12473 (GPT oil supply news)
- **Evaluation:** Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997); Clark & West (2007); Giacomini & White (2006); Hansen, Lunde & Nason (2011); Giacomini & Rossi (2010); Quaedvlieg (2021); Pesaran & Timmermann (1992); Gneiting & Raftery (2007); Gibbs & Candès (2021)
- **Transformers:**
  - Long-horizon and token redesign: Informer (AAAI'21); Autoformer (NeurIPS'21); FEDformer (ICML'22); PatchTST (ICLR'23); Crossformer (ICLR'23); iTransformer (ICLR'24); TimeXer (NeurIPS'24)
  - Covariates, critique and normalization: TFT (IJF'21); DLinear (AAAI'23); RevIN (ICLR'22)
  - Foundation models: Time-MoE (ICLR'25); Chronos (TMLR'24) and Chronos-2; TimesFM (ICML'24); Moirai (ICML'24); GIFT-Eval (2410.10393)
- **Agentic forecasting:** TimeCopilot (2509.00616); TimeSeriesScientist (2510.01538); TimeART (2601.13653); TemporalBench (2602.13272); Cast-R1 (2602.13802); KairosAgent (2605.30002); last-mile LLM agents (2606.02497)
- **LLM look-ahead:** Chronologically Consistent LLMs (2502.21206); Look-Ahead-Bench (2601.13770); DatedGPT (2603.11838)
- **Leakage in decomposition and LSTM pipelines:** VMDNet (2509.15394); 2512.06932
