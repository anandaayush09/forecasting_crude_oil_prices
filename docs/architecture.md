# Architecture

> **You own this document.** The diagram below is my proposal. Keep it true: when the code changes, change this.
> The three sections after it are empty on purpose. Fill them in your own words as you build; that is how you learn the system.

## Diagram (proposed)

```
 web/ (React + TS)  --REST + SSE-->  src/oilfc_api/ (FastAPI)
                                            |
                                            v
        src/oilfc/  research core (pure Python, no web imports)
        data/  eval/  models/  rag/  agents/
              |                 |                    |
        Postgres+pgvector     Redis            mcp_server/ (as_of-enforced tools)
        (results, documents)  (cache, queue)
```

Today only `src/oilfc/data/` exists. Everything else is a stub.

## Data flow (fill in)

Trace one number from the internet to a chart. Questions to answer:
- Where does a price enter the system, and in what format?
- Where is the release lag applied, and why must it happen before any model sees the data?
- What file or table holds it after each step?

## Request flow (fill in, once the API exists)

What happens, step by step, when the UI asks for `GET /benchmarks?freq=daily&h=22`?
Questions: which layer validates input, which reads the database, where is the cache, what happens on a cache miss?

## Failure modes (fill in)

For each box in the diagram: what breaks, how would you notice, what should the system do?
Start with: EIA is down during the daily refresh. A news article is dated after the forecast origin. The LLM returns invalid JSON.
