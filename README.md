# Multi-Horizon Crude Oil Price Forecasting with Transformers and Agentic AI

See ROADMAP.md for the research plan, timeline and evaluation protocol.

Setup (Python 3.12):
    python -m venv .venv && .venv/Scripts/activate
    pip install -r requirements.txt            # core
    pip install -r requirements-dl.txt         # + torch, transformers, TSFMs
    pip install -r requirements-agents.txt     # + agent stack (vllm: Linux/Colab only)
    # Moirai needs its own venv: pip install -r requirements-moirai.txt
    pytest tests -q

Protocol freeze: tag `protocol-frozen` (week 6). Test-split access is gated on it.

## Dev workflow

    make lint            # ruff check + format check (same as CI)
    make fmt             # auto-fix and format
    make test            # pytest
    pre-commit install   # once: run the checks automatically on every commit

`make` is not installed on Windows by default; run the commands inside the Makefile directly (e.g. `python -m ruff check .`).
Unfinished task cards live in `docs/tasks/`; their tests show as `XFAIL` until you implement them.

## Engineering plan

`docs/ENGINEERING_ROADMAP.md` (proposed): how AI-engineering concepts, a FastAPI backend and a React UI fit around the research, plus the ownership protocol.

`docs/FREE_RESOURCES.md`: the $0 policy. Every tool, dataset and service used, with its free limit and fallback.

## Prior work

The earlier DDP thesis (WTI at 1/7/90/180 days) is archived and linked from `prior_work/`:
- `prior_work/README.md`: map of the archive, canonical files and caveats
- `prior_work/DDP_KNOWLEDGE_BASE.md`: what was done, results, pitfalls, reusable pieces
- `prior_work/ddp/`: the full DDP archive itself (gitignored; `C:\Users\anand\Documents\DDP1` now links here)

This project starts fresh; the old work is reference only.
