# Crude Oil Volatility and Risk Forecasting

## What this is
Research code that tests whether forecasts of crude oil volatility and tail risk (VaR/ES) beat standard benchmarks (naive, HAR, GARCH), out of sample and without look-ahead. Brent and WTI, 5, 22 and 66 trading days.

## Stack
Python 3.12, pandas, numpy, scipy, statsmodels, arch, scikit-learn, LightGBM, PyTorch (transformer experiment on Kaggle), pytest, ruff.

## Commands
- Install: `pip install -r requirements.txt -r requirements-dl.txt`
- Build data: `PYTHONPATH=src python -m oilfc.data.build [--offline]`
- Test: `python -m pytest -q`
- Lint: `python -m ruff check . && python -m ruff format --check .`
- Results summary: `python experiments/summarise_vol.py`
- Live forecasts: run daily by `.github/workflows/live.yml` (the bot pushes to main: `git pull --rebase` before you push). By hand: `python experiments/live_run.py`, commit `live/forecasts` at once, then `python experiments/live_score.py` (see `live/README.md`; forecast files are never edited or overwritten)
- API: `python -m uvicorn oilfc.api.app:app` (read-only, spec in `docs/api_spec.md`)

## Layout
`src/oilfc/{data,features,eval,models,live,api}` library code, `experiments/` scripts that write `results/tables/`, `tests/`, `configs/`, `docs/` (ADRs, task cards), `literature/`.

## Conventions
- Type hints on public functions, docstrings that state units and information timing.
- Every feature at time t uses only data public by t (see `configs/data.yaml` for release lags). New feature builders go into the `FEATURES` registry in `tests/test_leakage.py`.
- The clean holdout (from 2025-07-01) is never evaluated in development: use `oilfc.eval.holdout.development_only`.
- Generated tables in `results/` are never hand-edited. Documents quote `results/tables/SUMMARY.md`.
- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`). No paid API calls in tests or experiments.

## Working rules
1. Think before coding: state assumptions, ask when unclear, surface trade-offs.
2. Simplicity first: the minimum code that solves the request, nothing speculative.
3. Surgical changes: touch only what the request needs. Every changed line traces to the request.
4. Goal-driven: define how success is verified, then verify it.
