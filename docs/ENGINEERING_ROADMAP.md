# Engineering roadmap: AI-engineering layer, React UI and ownership protocol

Status: **proposed, awaiting your review.** Nothing in this document has been built yet.

## Why this exists

You shared `AI_Engineer_Roadmap_Complete.md` and asked to (1) bring its concepts into this project (ignoring its portfolio-project suggestions), (2) add a frontend UI, and (3) get coding help while keeping full ownership and real understanding.

Your choices:
- Frontend: React + TypeScript on a FastAPI backend.
- Ownership: tiered by component (section 4).
- Priority: research first; engineering only where it also serves the research.
- Deployment: local Docker, plus one free public read-only demo.

This project replaces the generic portfolio projects in that roadmap. Each concept is learned by building the piece of *this* project that needs it.

## Free-tier policy

Everything here must be free: permanent free tier, no credit card, available on the open web. The full register (limits, fallbacks, what is not used) is [FREE_RESOURCES.md](FREE_RESOURCES.md). Consequences for this plan:
- LLMs: open-weight models on Kaggle/Colab for backtests; a free-tier API through LiteLLM (with failover) only for the live test. No Claude/OpenAI API in the project.
- Tracing: self-hosted Langfuse only (no LangSmith). Error tracking: local JSON logs (no Sentry).
- Public demo: a static React build with exported JSON on GitHub Pages or Cloudflare Pages. The live Agent console is shown as replayed recorded runs; the full app runs locally with Docker Compose.
- CI/CD: GitHub Actions and GHCR on a public repository.
- Kubernetes, MongoDB and Celery stay out of scope.

Where things stand (week 3 of 26, 5 Nov 2026): the data layer works (`src/oilfc/data/`, 7 tests pass); everything else is a stub; two commits, no remote, no CI.

## 1. Target architecture

```
 web/ (Vite + React + TypeScript)  --REST + SSE-->  src/oilfc_api/ (FastAPI)
   pages: Data, Benchmarks, Forecasts, Events,        routers / Pydantic schemas / deps
          Agent console, Live test                        |
   types generated from the API's OpenAPI spec            v
        +---------------- src/oilfc/ (research core: pure Python, no web imports) ----------------+
        | data/ (release-aligned panel)  eval/ (rolling origin, DM/CW/MCS/SPA)  models/  rag/  agents/ |
        +----------------------------------------------------------------------------------------------+
              |                       |                              |
        Postgres + pgvector      Redis (API cache,           mcp_server/ (as_of-enforced tools)
        runs, forecasts, metrics  job queue)                 <-- OilAgent (LangGraph) / any MCP client
        documents + embeddings, LLM call cache, agent traces, live-test log
              ^
        worker (scheduled: data refresh, backtests, daily live forecast)    Langfuse tracing (self-hosted), local JSON logs
```

Principles (your first ADRs):
1. The research core never imports the web stack. The API and UI read results; they never change how results are produced.
2. Leakage protection lives in infrastructure. Every data, retrieval and MCP tool takes `as_of` and filters server-side.
3. Postgres is the single source of truth for results. Parquet holds raw and processed data.
4. Every LLM call is cached by prompt hash (reproducible backtests, lower cost).
5. News text is untrusted input; prompt-injection guards sit on the retrieval-to-LLM path.

## 2. Concept map: AI-Engineer roadmap to this project

| Roadmap concept | Where it lives here | Research value |
|---|---|---|
| Pydantic, type hints, settings | `configs/*.yaml` via `pydantic-settings`; event schema; API schemas | Configs validated before experiments run |
| Decorators, context managers, logging | `@timed` on fits and LLM calls; `structlog` JSON logs | Efficiency table in the paper; debuggable runs |
| Async (asyncio, httpx) | Concurrent EIA/FRED/GDELT downloads | Faster data refresh |
| pytest, mocking | Leakage suite, mocked LLM clients, API tests | Leakage-free claims are tested |
| FastAPI, DI, middleware, SSE | `src/oilfc_api/` | Results dashboard; agent streaming |
| SQL / Postgres | Results store (runs, forecasts, metrics, tests) | Every table and figure comes from queries, never hand-typed |
| Redis | Response cache and job queue | Fast UI |
| Background jobs | Worker for data refresh, backtests, the daily live forecast | The live forward test runs itself |
| Tokens, cost, prompting, structured output | Event-extraction prompts with a Pydantic output schema; harness scored on 200 hand-labelled headlines | M1 quality is measured, not judged by eye |
| Multi-provider routing (LiteLLM) | Open-weight vLLM on free GPUs for backtests; free-tier Gemini/Groq (failover) for the live test | The cost plan in ROADMAP section 4.4 |
| Embeddings, chunking, vector DB, hybrid search, reranking | Point-in-time news RAG on pgvector + BM25 (reciprocal rank fusion), filtered by `as_of` | The Curator agent |
| RAG evaluation | Retrieval recall@k on labelled events; an "as-of violation = 0" test | Leakage-controlled retrieval |
| ReAct by hand, then LangGraph | `agents/react_min.py` first, then the `agents/graph.py` OilAgent | M2/M3 agent modes |
| MCP servers | `mcp_server/` with `as_of`-enforced tools | Leakage prevention in infrastructure |
| Memory | Memory of resolved past forecasts only | Reflection without look-ahead |
| HITL | `interrupt()` before a live-test forecast is published; approve in the UI | Audit trail for the live test |
| AI system design | `docs/architecture.md`; cost per forecast for M1-M4 | Agent cost section of the workshop paper |
| Tracing, prompt versioning, cost monitoring | Langfuse (self-hosted in compose, MIT licence); prompts versioned in git | Rationale-faithfulness audit |
| Eval in CI | GitHub Actions gates: ruff, pytest leakage suite, extraction-eval threshold, frozen mini-backtest regression | Protocol stays frozen |
| Docker (multi-stage, non-root), Compose | `docker/`, `compose.yaml` | Reproducibility package for reviewers |
| CI/CD, GHCR, staged deploy, health checks, rollback | Commit-SHA images on GHCR; static demo deployed to GitHub Pages or Cloudflare Pages, production gated by a manual approval on a GitHub environment | Public demo link |
| Structured logs | API and worker (JSON logs, no paid error tracker) | Live-test reliability |

Optional, only after submission: Kubernetes, MongoDB, Celery. Postgres JSONB plus a simple worker cover the same needs; explaining that choice is an ADR in itself.

## 3. The UI (React + TypeScript)

Stack: Vite, React, TypeScript, React Router; TanStack Query and Table; Recharts; Tailwind; Vitest + React Testing Library. TypeScript types are generated from FastAPI's OpenAPI spec (`openapi-typescript`), so the frontend-backend contract is checked at compile time.

| Page | Shows | API |
|---|---|---|
| Data | Panel series; an "as-of" view of what was known on date t, with release ages | `GET /series`, `GET /panel?as_of=` |
| Benchmarks | MSPE ratio vs RW by model and horizon; CW, DM, MCS, multi-horizon SPA; crisis filter | `GET /runs`, `GET /benchmarks`, `GET /tests` |
| Forecasts | Latest fan charts per horizon; anchor vs model; regime gate and shrinkage | `GET /forecasts/latest` |
| Events | Timeline of extracted events over the price, linked to source documents | `GET /events` |
| Agent console | One OilAgent run, streamed: retrieved docs, tool calls, adjustment, critic verdict | `POST /agent/runs`, SSE stream |
| Live test | Hash-committed forecasts vs outcomes; HITL approve/reject | `GET /live`, `POST /live/{id}/decision` |

Plus `GET /health`, which checks the DB, Redis and vector store, not just the process.

## 4. Ownership protocol

| Tier | Components | Who writes | My role |
|---|---|---|---|
| A: research core | models, `eval/`, AR-MoE, OilAgent logic, prompts | **You** | Specs, failing tests, review, explanations |
| B: learning-critical | FastAPI routers, DB schema and queries, RAG retrieval, hand-built ReAct loop, MCP tools, React pages | **You**, from a task card | Task card, failing tests, review; unblock when stuck |
| C: boilerplate | CI YAML, Dockerfiles, compose, ruff/pre-commit config, scaffolding | **Me**, drafted | Line-by-line comments; you read it, change one thing and explain it back |

Learning artifacts that you write and I review:
- `docs/architecture.md`, kept current by you.
- `docs/adr/NNNN-*.md`: one per decision (context, options, decision, consequences). This rehearses the "why X over Y" interview probe.
- A weekly personal log, three lines (built, broke, learned), kept outside the repository.
- Task cards in `docs/tasks/` (goal, concepts, acceptance tests, hints, stretch). Each ships with a test marked `xfail` until you implement it, so CI stays green.
- Explain-back gate: each milestone ends with a short quiz from me; it is done only when you pass it.

## 5. Milestones (about 8-10 hours a week of engineering)

| Weeks | Milestone | Deliverables | Runs alongside |
|---|---|---|---|
| 3-5 | E0 Foundations | GitHub repo and Actions CI (ruff, pytest); pre-commit; `pydantic-settings`; `structlog`; async `httpx` downloads; `.env.example`; docs skeleton | Phase 1 data, EDA |
| 5-9 | E1 Store, API, UI shell | Compose (Postgres + pgvector, Redis); results schema and writer; FastAPI read endpoints and `/health`; OpenAPI to TS types; React shell with Data and Benchmarks pages | Phase 2 baselines; protocol freeze in week 6 |
| 9-13 | E2 LLM and RAG | Corpus ingest, chunk, embed, hybrid `as_of` retrieval, rerank; extraction prompts and harness; LiteLLM routing; cost logging; LLM cache; injection guard; Events page | Phase 3 and 4b (M1) |
| 12-17 | E3 Agents | MCP server; hand-built ReAct loop, then LangGraph OilAgent; resolved-forecast memory; HITL interrupt; Langfuse tracing; SSE Agent console; Live page; worker runs the daily live forecast | 4b M2/M3; live test from week 12 |
| 17-21 | E4 LLMOps hardening | CI eval gates; prompt versioning; multi-stage non-root images; pip-audit; Forecasts page; full Benchmarks page | Final evaluation |
| 21-26 | E5 Ship and document | GHCR commit-SHA images; static read-only demo on GitHub Pages/Cloudflare Pages (replayed agent runs, exported results) with manual approval for production; README with architecture, decisions, what broke, rollback | Writing and submission |

If research slips, E4 and E5 shrink to Docker plus README. E0-E3 stay, because they are research infrastructure.

## 6. Repo additions

```
src/oilfc_api/   main.py, deps.py, routers/, schemas/
src/oilfc/store/ db.py, models.py, writer.py, migrations (alembic)
src/oilfc/rag/   ingest.py, chunk.py, embed.py, retrieve.py (hybrid + as_of), guard.py
src/oilfc/agents/ react_min.py, graph.py, schemas.py, prompts/, memory.py, llm.py (LiteLLM + cache)
mcp_server/      server.py (as_of-enforced tools)
web/             Vite React TS app
docker/          api, worker, web Dockerfiles
compose.yaml  .env.example  .github/workflows/ci.yml  .pre-commit-config.yaml  pyproject.toml
docs/            architecture.md, ENGINEERING_ROADMAP.md, adr/, tasks/
```

## 7. Proposed first step (E0 scaffolding), not yet done

Me (tier C, commented line by line): `pyproject.toml` with ruff and pytest config (replacing `pytest.ini`), `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (ruff + pytest), `.env.example`, ruff and pre-commit added to requirements.

You (tier B, task cards with `xfail` tests):
- E0-1: concurrent downloads with `httpx.AsyncClient` and `asyncio.gather`, with retries and a timeout.
- E0-2: a `pydantic-settings` model loading `configs/*.yaml`; a malformed config must fail with a clear error.
- E0-3: `structlog` plus a `@timed` decorator used in `build.py`.
- ADR 0001, "research core has no web deps": title and prompts from me, the content from you.

Not done by me: creating the GitHub repo and pushing (you do it; I will walk you through), ADR content, any research-core code.

Verification: `ruff check` and `ruff format --check` pass; `pytest -q` shows the 7 current tests passing plus the new task tests as xfail; the YAML files parse; once pushed, the Actions run is green, and walking me through what each CI step did is your first explain-back.
