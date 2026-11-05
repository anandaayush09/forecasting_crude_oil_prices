# DDP thesis: knowledge base for this project

Archive: `prior_work/ddp` (the real archive, moved here on 2026-11-05; `C:\Users\anand\Documents\DDP1` is now a link to it). Start with `prior_work/ddp/README.md` there, then `MANIFEST_files.csv` to locate any file by its old or new path.
All paths below are relative to `prior_work/ddp/`.

## What was done

- **Question:** forecast WTI spot at 1, 7, 90 and 180 days with calibrated uncertainty; test improvement over naive and over the Yu (2025) and Huang (2024) published pipelines using Diebold-Mariano tests.
- **Data:** EIA WTI spot 1986-2026; exogenous Brent, S&P, gas, gold, DXY, 10Y yield, VIX, OVX, FRED macro, Cushing and US stocks, rig count, OPEC output, EPU/GPR, CFTC positioning (indexed by release date), Google Trends, GDELT/FinBERT sentiment; a 4-state HMM regime feature. Feature sets of 48 / 61 / 69 variables per horizon.
- **Target:** forward log-return converted to price; a shrinkage factor chosen on validation data caps every model at naive.
- **Track A ("leaky paper reproduction"):** Yu (WOA-VMD-FE) and Huang (VMD-EMD) with whole-series decomposition and a 70/30 split; about 15-18 architectures each, plus linear/statistical heads (LHYu, LHHuang).
- **Track B ("honest walk-forward"):** expanding window, tests from 2007, refit yearly (24 months for 90d/180d), RobustScaler on train only, rolling 360-day VMD recomputed every 5 days where used. Eleven finished neural models: iTransformer, Autoformer, Informer, FEDformer, TFT, TimeMixer, N-HiTS, TiDE, TCN, TCN-BiLSTM, CNN-BiLSTM-Attn; plus a PatchTST run. Failed or unfinished: SegRNN, FEDformerW, FreEformer (cuFFT/AMP bug), FITS.
- **Extras:** Chronos and TimesFM zero-shot, TimeRAG analog retrieval, split-conformal 80% intervals, GPD-POT EVT, GARCH(1,1)-EVT, Kupiec/Christoffersen tests, PIT/CRPS, SLSQP ensemble.
- **Not implemented:** Time-LLM and the agentic RAG forecaster were proposals only.

## Results (MAE, USD/bbl)

| Horizon | Naive | Best Track B | Best Track A (leaky) |
|---|---|---|---|
| 1d | 1.22 | 1.22 (all models shrink to naive) | 0.69 |
| 7d | 3.21 | 2.39 (Informer, DM p<0.05) | 1.25 |
| 90d | 12.37 | 12.56 (TFT, worse) | 5.23 |
| 180d | 16.78 | 15.62 (PatchTST) | 11.59 |

Also: about 72-75% directional accuracy at 7d passes the Pesaran-Timmermann test; no 1d model beats naive significantly; calibration at 90d/180d is poor (reliability 0.015-0.033, KS-PIT 0.11-0.16); EVT VaR99 is $6.05 / $13.20 / $65.51 / $75.29; linear heads beat neural heads on decomposed components; Chronos zero-shot is worse than naive.
Sources: `01_thesis/final/Final Thesis Report.md`, `07_notes/defense_guide.md`, `04_results/final/comparator_table.csv`.

## Lessons and pitfalls (what the new project must avoid)

1. Whole-series VMD plus a 70/30 split inflates results enormously (1d MAE 0.69 vs 1.22 naive). Treat all Track A numbers as an upper bound from leakage.
2. Track B still leaks: no purge between train and test (`train_end == test_start`, up to h days look-ahead at 180d); winsorization floor and HMM regimes fitted on the full series; at least one run included `vmd_*_leaky` features.
3. `03_code/pipeline_final/17_audit_pipeline.py` checks files and schemas, not leakage.
4. Diebold-Mariano result tables are empty in every copy.
5. Many 1d models collapse to exactly the naive forecast (MAE 1.22): a symptom of over-strong shrinkage, not skill.
6. The 90d VMD result (MAE 3275) and `RAGForecaster` (MAE about 37) are broken runs; seven `predictions_*_fused_1d.csv` files are identical empty copies.
7. Thesis captions call Track A "walk-forward"; the logs show a 70/30 split.
8. The Xu & Niu 2024 "decomposition lookahead bias" citation is unverified.

## Reusable pieces (each needs the stated fix before use)

| Piece | Where | Needed fix |
|---|---|---|
| Walk-forward harness | `03_code/pipeline_final/_forecast_template.py`, `_run_wf_arch.py` | add an h-day purge/embargo; fit winsorization and HMM on train only |
| Causal rolling VMD | `03_code/pipeline_final/01a_vmd_features.py`, `09_vmd_hybrid.py`, `B_honest_decomp.py` | none known; use as the "causal" arm of the RQ2 experiment |
| Yu/Huang leaky replications | `03_code/pipeline_final/_paper_helpers.py`, `A_*.py`, `T_thesis_leaky_repro.py` | none; they are the "leaky" arm (leakage-premium control) |
| Conformal and GARCH-EVT | `03_code/pipeline_final/13_probabilistic_forecast.py`, `_conformal_plus.py`, `07b_garch_evt.py` | recheck calibration on the new protocol |
| Text/RAG features | `03_code/pipeline_final/10_text_features.py`, `R_timerag.py` | restrict news to dates on or before the forecast origin |
| Feature construction | `03_code/pipeline_final/00_setup_data_features.py` | the new project has its own release-aligned pipeline (`src/oilfc/data`); compare, don't copy |
| Audit script | `03_code/pipeline_final/17_audit_pipeline.py` | extend with real leakage checks |
| News data | `04_results/final/data/CrudeOilNewsData.csv` | only about 400 articles (Apr-Aug 2025): too short to train on |
| CrudeBERT weights | `03_code/pipeline_v1_may/crudebert_model/crude_bert_model.bin` (418 MB, only copy) | none |

## Stated future work in the thesis

Add futures term structure, OPEC schedules and forward-looking macro data for long horizons; refresh rolling decomposition more often; fine-tune foundation models; extend to Brent and natural gas; fine-tune a language model on oil text with a structured event database; test decision value (hedging/procurement) with the full predictive distribution.

## Where the documents are

Final thesis `01_thesis/final/Final Thesis Report.pdf` (+ `.md` export); LaTeX `01_thesis/latex_source/`; stage reports `01_thesis/stage_reports/`; final deck `02_presentations/thesis_defense_v22.pptx`; working notes `07_notes/` (`DDP_Session_Summary.md`, `Track A and B.txt`, `defense_guide.md`); methodology `03_code/pipeline_final/THESIS_METHODOLOGY.md`; critique of the predecessor thesis `07_notes/New_folder_2/Thesis_Critique_and_Improvements.md`.
