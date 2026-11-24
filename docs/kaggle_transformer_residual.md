# Running the transformer residual experiment on Kaggle

**Question (RQ3/RQ4):** does a transformer that reads the last 64 days add anything to the HAR + OVX volatility anchor, and does attention matter compared with an MLP of similar size?

**What runs:** `experiments/vol_residual_seq.py` (Brent, 22-day volatility). The anchor (HAR + OVX regression) is refit every month, like the benchmark ladder. Every 12th month it also fits three residual models on the flattened window of six daily channels (`ret_1, abs_ret, log_rv_5, log_rv_22, log_ovx, log_vix`): ridge (linear control), a flat MLP (attention-free control, 20,881 parameters) and a patch transformer (19,009 parameters, 2 layers, d_model 32, patch 8). A weight alpha, chosen on a time-ordered validation block, shrinks each correction toward the anchor. Seeds are averaged (5 in the full run).

## Steps

1. **Build the bundle (once, and again if code changes):**
   ```bash
   .venv/Scripts/python.exe kaggle/build_upload.py
   ```
   This writes `kaggle/oilfc-residual.zip` (0.3 MB: `src/oilfc`, the script, `panel_daily.parquet`).
   (Rebuild it after any code change. The script now trims to development data before the clean holdout; an older upload does not have that guard.)
2. **Upload it** on kaggle.com: Datasets, New Dataset, upload the zip, name it `oilfc-residual`, Private.
3. **Create a notebook:** New Notebook, File, Import Notebook, choose `notebooks/kaggle_transformer_residual.ipynb`. Add Input and attach `oilfc-residual`. In Settings set Accelerator to GPU (P100 or T4). Internet is optional; it only enables the Model Confidence Set (`arch`).
4. **Run the cells in order.** Cell 3 is a smoke test (13 monthly splits with 2 residual refits, 1 seed, 5 epochs; results are meaningless, it only proves the pipeline runs). Cell 4 is the full Brent 22-day run. Cell 5 is optional (5-day horizon, WTI).
5. **Download `residual_results.zip`** from the Output panel as soon as the last cell finishes; `/kaggle/working` is cleared when the session ends.
6. **Unzip into `results/kaggle_residual_monthly/`** (a new folder; `results/kaggle_residual/` keeps the first run with a yearly anchor) and send the tables for review.

## What to read in the results

- `qlike_gain_vs_anchor_pct` and `dm_p_qlike_vs_anchor`: a gain only counts if p < 0.05. Compare with the earlier ridge/LightGBM/MLP residuals, which added nothing detectable at 22 days.
- `mean_alpha`, `share_alpha_zero`: how much of the correction survived shrinkage. Alpha near 0 means the model found nothing.
- `transformer vs MLP, QLIKE Diebold-Mariano p` (printed and in the `.json`): the attention question. A significant p is needed to claim attention helps.
- `mcs_pvalue` / `in_set`: whether a model is separable from the anchor in the 90% Model Confidence Set.

## Known limits

- Training rows start with OVX (2007-05), so the first 2012 refit has only about 1,150 rows; deep models are data-starved early on.
- Timing is a guess: the CPU smoke test took 45 s for 2 refits, 1 seed, 5 epochs; the full run (15 refits, 5 seeds) is expected to take tens of minutes on a GPU, not yet measured.
- Early stopping and alpha use a validation block taken from the end of the training rows; its targets overlap the fit part by up to h days, which affects model selection only, never the test.
- One asset, one horizon per run; no correction for testing several models.

## Results of the first run (2026-11-23)

Files: `results/kaggle_residual/` (Brent 22 and 5 days, WTI 22 days, plus the smoke test). Summary and the refit-frequency confound are in `ROADMAP.md` (baseline results, transformer residuals). Short version: no gain at the primary setting; the single significant cell (Brent, 5 days) is explained by comparing against a yearly-refit anchor and disappears against monthly-refit benchmarks. Next step before any claim: monthly anchor refits with yearly residual refits.

## Rerun with a monthly anchor (2026-11-24)

The first run refit the anchor yearly, so its one significant cell (Brent, 5 days) compared a yearly anchor with a model that also moved the anchor. The script now refits the anchor monthly (`--refit-months 1`) and the residual models every 12 splits (`--residual-every 12`, the notebook sets both). Rebuild the bundle (`kaggle/build_upload.py`), upload it again as a new version of `oilfc-residual`, rerun cells 3 to 5 and unzip the outputs into `results/kaggle_residual_monthly/`. Then run `experiments/residual_vs_monthly.py kaggle_residual_monthly`.

**Result of the rerun (2026-11-24):** see `ROADMAP.md` (baseline results, transformer residuals). The 5-day transformer gain against its own anchor stays nominally significant (p = 0.014) but not after correction for nine comparisons, and nothing beats the monthly leverage HAR + OVX.
