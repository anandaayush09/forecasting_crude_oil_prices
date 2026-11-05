# E0-3. Structured logging and a `@timed` decorator

**Module you create:** `src/oilfc/logging_utils.py`  **Tests:** `tests/test_logging.py`

## Why this matters

`print()` output cannot be searched, filtered or aggregated. In production (and in your live forward test) you debug from logs alone; there is no debugger attached. Structured logs are one JSON object per line, so a tool can answer "which LLM calls took longer than 5 seconds?". The `@timed` decorator will measure fit times for the efficiency table in the paper and latency of LLM calls later.

## Read first

- structlog docs: *Getting Started* and *Testing* (`structlog.testing.capture_logs`).
- Corey Schafer's *Python Decorators* video (or any decorator tutorial), and `functools.wraps`.

## Interface

```python
from oilfc.logging_utils import configure_logging, timed

configure_logging(level="INFO", json=True)   # call once at program start

@timed
def fit(...): ...                            # works for plain AND async functions
```

Behaviour the tests pin down:
- `@timed` emits exactly one log event named `"timed"` with fields `func` (function name), `duration_ms` (float), and `status` (`"ok"` or `"error"`).
- On an exception it logs `status="error"` plus `error_type` (e.g. `"ValueError"`), then **re-raises** the same exception.
- The decorated function keeps its `__name__` and docstring.
- `configure_logging(json=True)` prints one JSON object per line with `event`, `level` and `timestamp`; log calls below `level` are dropped.

## Hints (not answers)

- A decorator that must support both `def` and `async def` needs to detect which it got (`inspect.iscoroutinefunction`) and return a matching wrapper.
- Which clock should you use to measure durations: `time.time()` or `time.perf_counter()`? Why?
- Put the logging in a `finally`-style structure so the error path is also timed.

## Done when

- All 6 tests pass and you removed the `xfail` marker.
- Run `PYTHONPATH=src python -m oilfc.data.build --offline` after decorating `build()` with `@timed`, and show me the JSON line it prints.
- Self-check: (1) why JSON instead of f-strings? (2) why must `@timed` re-raise? (3) what goes wrong if the wrapper loses the function's name?

## Stretch

Bind a `run_id` to every log line of one pipeline run with `structlog.contextvars`, so you can filter a whole run in the logs.
