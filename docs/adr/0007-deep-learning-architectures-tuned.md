# 0007. Deep-learning architectures from the earlier thesis, tuned, with more inputs: design and success rule, fixed before any tuning or run

- Status: accepted (Aayush, 2026-11-26)
- Date: 2026-11-26

## Context

ADR 0004 tested a ridge control, an MLP, a GRU and a patch transformer as bounded corrections to the headline (`lhar_ovx_bc_lag1`). None helped: 0 of 36 Holm tests, and the shuffled-OVX placebo did about as well as the real input. Two objections to that null remain. The settings were fixed and small (22-day window, width 32, no tuning), and the inputs were six channels. The earlier thesis also used other architectures that were never run under a leakage-safe protocol.

This note fixes the architectures, the inputs, the tuning procedure and the success rule before any tuning or run. It concerns development data only (test origins 2012-01 to 2025-06). The holdout is not touched, `configs/protocol.yaml` does not change, and there is no live track for this work. A null is the expected outcome.

## Architectures (K = 6)

Each is a compact pure-PyTorch version of the core block of the named architecture, not a port of its reference code. Each plugs into the anchored framework of ADR 0004: the final forecast is the headline times exp(alpha times the correction), the correction is bounded by `bound * tanh`, the last linear layer starts at zero, and the loss is the QLIKE-consistent loss. An untrained net therefore reproduces the benchmark exactly.

| key | after | core block kept |
|---|---|---|
| `nhits` | N-HiTS (Challu et al. 2023) | three MLP stacks on max-pooled copies of the window (pool sizes 1, 4 and 11), outputs summed |
| `tide` | TiDE (Das et al. 2023) | per-day feature projection, then a dense residual-MLP encoder over the flattened window |
| `itransformer` | iTransformer (Liu et al. 2024) | each channel's whole window as one token, attention across channels |
| `tcn` | TCN (Bai et al. 2018) | dilated causal convolutions with residual blocks, last time step to the head |
| `tft` | TFT (Lim et al. 2021) | gated variable selection, LSTM encoder, interpretable multi-head attention, gated output |
| `informer` | Informer (Zhou et al. 2021) | ProbSparse self-attention with a distilling convolution between layers |

Models of the thesis that are not run, with the reason:

- PatchTST: the Stage A patch transformer is the same design.
- Autoformer and FEDformer: built to decompose long seasonal series. A window of 22 to 66 trading days has no seasonality to decompose.
- TimeMixer: its multi-scale MLP mixing is the idea already covered by N-HiTS.
- TCN-BiLSTM and CNN-BiLSTM-Attn: hybrids of blocks that are already covered (TCN, the Stage A GRU, attention).
- SegRNN, FreEformer, FITS: unfinished in the thesis and not part of this design.

## Inputs (12 channels per day)

- The six channels of `features.vol.CHANNELS` (return, absolute return, log 5-day and 22-day realised volatility, log OVX, log VIX), with OVX and VIX lagged one trading day, as in Stage A.
- Four from `features.exog.exogenous_features`: `stocks_chg_4w`, `stocks_dev_1y`, `log_gpr`, `log_epu`.
- Two from `features.exog.fx_features`: `usd_chg_22`, `log_usd_vol_22`.
- Not used: `log_vix` from the exogenous builder (a duplicate of the channel above) and `fut_slope` (futures data end in 2024-04).
- Every extra series starts no later than OVX (2007-05-10), so the test sample does not shrink. The new channel builder is registered in the leakage test.

## Tuning (uses only data before the first test origin)

- **Data:** in each of the six cells, the training rows of the first monthly split only (their targets end before 2012-01-01). The first 80% of those rows fit the net, h rows are skipped, and the rest validate.
- **Objective:** the mean over the six cells of the ratio of the net's validation loss to the anchor's validation loss, with the QLIKE-consistent loss. Each architecture gets one shared set of hyperparameters. Pooling the cells reduces the noise of the short pre-2012 sample.
- **Search:** Optuna TPE sampler with seed 0, 30 trials per architecture, 2 net seeds per trial.
- **Space:** window {22, 44, 66}, width {16, 32, 64}, depth {1, 2, 3}, dropout in [0, 0.3], learning rate in [3e-4, 1e-2] (log scale), weight decay in [1e-4, 1e-1] (log scale), bound {0.25, 0.5}, plus at most one architecture-specific choice: attention heads {2, 4}, or kernel size {2, 3} for TCN.
- **Output:** `results/dl_arch/hparams.json` and `results/dl_arch/tuning_trials.csv`, committed before the main run starts.
- **Stated limit:** the validation block at 66 days holds about 150 overlapping rows, so tuning at that horizon is noisy.

## Main run

- Anchor, splits, refit rule, seeds, loss, alpha rule and cells are those of ADR 0004 (monthly anchor refits, residual models refit every 12 splits, alpha chosen on the last 20% of each training window, 5 seeds averaged in correction space). The tuned hyperparameters replace the fixed Stage A settings.
- Run on a free Kaggle GPU.
- A ridge control on the 12-channel window (as in Stage A, alpha 1000) shows what the extra inputs give a linear model. It is outside the test family.

## Tests and success rule

- **Test family:** 6 architectures times 6 cells = 36 one-sided Diebold-Mariano tests (Harvey correction, rectangular kernel) of QLIKE against the headline, Holm-adjusted together. Hansen's SPA (consistent p-value, block bootstrap with block length h, 1,000 repetitions, seed 0) is reported per cell over the 6 architectures.
- **Success rule:** an architecture helps in a cell only if its Holm-adjusted p is below 0.05 and the SPA p of that cell is below 0.05. Anything else is reported as a null. Every architecture and cell is reported.
- **If an architecture helps:** it is compared with naive, HAR, leverage HAR, GARCH-t and GJR-t (one-sided, Holm over 30 tests, as in ADR 0005) and reported as a development-data result selected by a pre-registered rule, not as a confirmed result.

## Consequences

- If every architecture is null, the deep-learning section of the paper reports a second controlled null, with tuning and wider inputs as the answers to the two objections to the first.
- The tuning period is the only place where the net's settings are chosen. No setting is changed after the first main run.
- Departures from this note are recorded as dated amendments here, not by rewriting it.

## Amendments

- 2026-11-26, before any tuning or run: the N-HiTS pooling uses ceil mode, so any window length works and the window choice applies to N-HiTS like to the other architectures (this replaces the sentence in the search space that said N-HiTS ignores the window choice). The iTransformer adds a learned embedding per channel to its tokens, because attention and the mean over tokens would otherwise not tell the channels apart. Neither change affects the success rule.
