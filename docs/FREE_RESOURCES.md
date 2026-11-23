# Free resources register ($0 policy)

**Rule:** everything this project needs must have a *permanent* free tier, require **no credit card**, and be available on the open web. No paid APIs, no paid data, no paywalled-only papers, no trials or credits that expire.

Free-tier limits change often. The "limit" column is what secondary sources reported on 2026-11-05; **confirm in the provider's own dashboard before relying on a number**, and write the date you checked next to it.

Adding a new tool or dataset? It must pass this checklist, then get a row here:
1. Permanent free tier (not a trial). 2. No card needed. 3. Limit written down. 4. A fallback if the limit changes. 5. Nothing in the research results depends on a service that could vanish (cache outputs, record versions).

## Compute

| Need | Free choice | Card? | Limit (verify) | Fallback / how we cope |
|---|---|---|---|---|
| GPU for training | Kaggle Notebooks (P100 or T4) | No | About 30 GPU-hours per week reported; the number floats, check your account settings. Sessions up to about 12 h | Colab free GPU (no guaranteed quota); small models, yearly refits, checkpoints every epoch, 5 seeds only for final runs |
| CPU work (data, statistics, tests) | Your laptop | No | none | none needed |
| Local services | Docker Desktop (personal use) or Podman | No | none | Podman if the Docker licence ever blocks you |

## LLMs

| Need | Free choice | Card? | Limit (verify) | Fallback / how we cope |
|---|---|---|---|---|
| Historical backtests (thousands of calls) | Open-weight models (Qwen, Llama; weights from Hugging Face) served with vLLM on Kaggle/Colab | No (free HF account for gated models) | GPU hours (above) | Smaller model (3-4B); cache every call by prompt hash; run overnight in chunks |
| Live forward test (about 60 days, a few calls a day) | Free-tier API via LiteLLM: Google Gemini API free tier and/or Groq free tier; OpenRouter `:free` models as a third option | No | Gemini: limits shown in AI Studio, reportedly reduced in 2026. Groq: reports disagree (for example 30 requests/min; 8k tokens/min can be the binding limit). OpenRouter free: about 20 requests/min and 50/day reported | Two providers behind LiteLLM with failover; the open-weight model on a free GPU as last resort. Log model id and date on every call |
| Embeddings | `sentence-transformers` models run locally | No | none | none needed |
| Reranking | Local cross-encoder from `sentence-transformers` | No | none | none needed |

Cautions: free API tiers may use your prompts to improve their models. Only public news and prices go through them, never anything private. Free-tier model versions change silently, so the live test records the exact model id and caches responses; headline claims rest on the open-weight vintages and the logged live test.

## Data (all free, all on the web)

| Data | Source | Notes |
|---|---|---|
| Brent, WTI spot | EIA (public XLS, no key) and FRED | Already downloading in `src/oilfc/data/` |
| Futures contracts 1-4 | EIA | The full curve needs paid data and is out of scope |
| Macro and market series | FRED (CSV needs no key); ALFRED real-time vintages need a free API key | |
| Weekly petroleum status, STEO text | EIA | |
| Positioning | CFTC Commitments of Traders | |
| Geopolitical risk, policy uncertainty | GPR (Caldara and Iacoviello) and EPU (Baker, Bloom, Davis) from the authors' sites | |
| News events | GDELT 2.0 (free); EIA and OPEC press releases | |
| Real-time oil-market data used by Baumeister and Kilian | Authors' web pages | Check availability in Phase 0 and note it in `literature/lit_matrix.csv` |

## Literature

Search and read with: arXiv, SSRN, RePEc/IDEAS, Semantic Scholar, OpenAlex, Unpaywall and CORE (legal free copies), the authors' own pages, and Google Scholar's "all versions" link. A paywalled journal paper is acceptable to cite only if you can read a free version (preprint or author copy). `lit_matrix.csv` has a `free_access_url` column; fill it so every paper in the review can be re-read at no cost.

## Software (all open source)

PyTorch, neuralforecast, statsforecast, statsmodels, arch, scikit-learn, LightGBM, Optuna, MLflow, Hydra; Chronos, TimesFM, Moirai and TiRex weights on Hugging Face; LangGraph, LiteLLM, the MCP SDK, vLLM; PostgreSQL with pgvector, Redis; Langfuse (self-hosted, MIT licence); FastAPI, React, Vite, TanStack, Recharts, Tailwind.

## Engineering services

| Need | Free choice | Notes |
|---|---|---|
| Git hosting and CI | GitHub public repository with Actions | Free for public repos; a private repo has a monthly minute cap. Keep the repo public once you are ready to publish |
| Container images | GitHub Container Registry (public images) | |
| Public demo | Static React build with pre-exported JSON on GitHub Pages or Cloudflare Pages | No server, no card. Roughly: GitHub Pages about 1 GB per repo and 100 GB/month soft bandwidth. The live Agent console is shown as replayed recorded runs; the full live app runs locally with Docker Compose |
| Tracing | Langfuse self-hosted in Docker Compose | Replaces LangSmith (cloud free tier has limits) |
| Error tracking | Local structured JSON logs | Sentry is dropped from the plan |

## Publishing

- Preprints: arXiv and SSRN (free). Code and data DOI: Zenodo (free).
- Journal: publish through the subscription route, which has no article-processing charge. Do not choose gold open access unless a fee waiver applies. Check the journal's current policy before submitting.
- Workshops: submitting is free; travel or registration is your decision later.

## Explicitly not used

Bloomberg and Datastream; Claude, OpenAI or other paid LLM APIs inside the project; LangSmith; Cohere Rerank; Sentry; paid cloud hosting (Render paid, Cloud Run, AWS); paid courses and books. (Your own Claude Code session is a separate tool that helps you write code; no project code calls it.)
