# prior_work: the DDP thesis archive

The earlier DDP thesis (WTI crude oil forecasting at 1/7/90/180 days) lives in `ddp/` inside this folder, as a self-contained project of its own. It was reorganised and moved here on 2026-11-05 (files moved, none deleted). The old location `C:\Users\anand\Documents\DDP1` is now just a link to `ddp/`, so old paths still work.

| File or folder | What it is |
|---|---|
| `DDP_KNOWLEDGE_BASE.md` | What the thesis did, its results, pitfalls, and which code is reusable |
| `ddp/` | The real DDP archive (about 6 GB, including 4.7 GB awaiting your review in `_review_before_delete/`). It is gitignored, so it never enters this project's git history |

## Map of the archive (`ddp/`)

| Folder | Contents |
|---|---|
| `README.md` | Index, key results and known problems. Start here |
| `01_thesis/final/` | Final Thesis Report (PDF, signed) and text export |
| `01_thesis/latex_source/` | LaTeX source |
| `01_thesis/stage_reports/` | Stage 0, 1, 2 reports |
| `02_presentations/` | Final defense deck, older versions |
| `03_code/pipeline_final/` | Canonical code (Track A and B) |
| `03_code/pipeline_v1_may/` | Earlier pipeline, CrudeBERT weights |
| `04_results/final/` | Canonical results (comparator tables, predictions, figures) |
| `05_data/` | WTI/Brent/FRED data, news CSVs |
| `06_literature/` | Papers by topic |
| `07_notes/` | Working notes, session summary, defense guide |
| `_review_before_delete/` | About 4.7 GB of duplicates, venvs and caches awaiting your deletion |
| `MANIFEST_files.csv` | Old and new path of every file |
| `undo_moves.py` | Restores the original DDP1 layout |

## Caveats in one line each

- Track A results are leaky (whole-series decomposition, 70/30 split): never cite them as real performance.
- Track B still has leaks (no purge gap; full-series winsorization and HMM regimes).
- The DDP audit script checks files, not leakage; the Diebold-Mariano tables are empty.

Details and sources are in `DDP_KNOWLEDGE_BASE.md`.
