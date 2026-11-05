# E0-1. Concurrent, retrying downloads

**Module you create:** `src/oilfc/data/async_fetch.py`  **Tests:** `tests/test_async_fetch.py`

## Why this matters

`build.py` downloads series one after another with blocking `requests`. A real refresh later fetches dozens of sources (EIA, FRED, GDELT). Network calls spend almost all their time waiting, which is exactly what `asyncio` is for. The same pattern is how an agent calls several tools at once.

## Read first

- httpx: *Async Support* (python-httpx.org/async).
- SuperFastPython: *Python Asyncio: The Complete Guide* (the sections on `async`/`await`, `gather`, and semaphores).
- httpx docs: *Transports* and `MockTransport` (how the tests avoid the real network).

## Interface (the tests pin this down)

```python
async def fetch_text(client, url, *, params=None, retries=3, backoff=0.5) -> str
async def fetch_many(client, urls: dict[str, str], *, concurrency=5, retries=3, backoff=0.5) -> dict[str, str]
```

- `client` is an `httpx.AsyncClient`. Never create it inside these functions; the caller owns it (this is what makes testing possible).
- `retries` is the number of **extra** attempts after the first. Retry on HTTP 5xx and on transport errors. Do **not** retry on 4xx: a 404 will not fix itself.
- Wait `backoff * 2**attempt` seconds between attempts. After the last attempt, raise the original `httpx.HTTPStatusError`.
- `fetch_many` returns `{key: body}` for every key in `urls`, runs requests concurrently, and never has more than `concurrency` requests in flight.

## Hints (not answers)

- How do you let N coroutines run while limiting how many run *at once*? Look up `asyncio.Semaphore`.
- `await asyncio.sleep(...)` yields control; `time.sleep(...)` blocks everything. Which one belongs in the backoff?
- `response.raise_for_status()` raises for both 4xx and 5xx. How will you decide whether to retry?

## Done when

- All 7 tests pass and you removed the `xfail` marker.
- Self-check, in your own words: (1) why is five `asyncio.sleep(0.2)` calls under `gather` faster than five in a loop? (2) what would happen to a server if you had no concurrency limit? (3) why does the function take the client as an argument instead of creating it?

## Stretch (after the tests pass)

Change `build.py` to download the FRED series with `fetch_many` while keeping `parse_fred_csv` synchronous and pure. Keep `--offline` working. Measure the time before and after, and write the numbers in `docs/progress_log.md`.
