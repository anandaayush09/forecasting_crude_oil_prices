# Task cards

Each card is a small piece of the project that **you** write. I provide the goal, the interface, the tests and hints; I do not provide the implementation.

## How to work a card

1. Read the card, then the "Read first" links. Do not skip the reading; it is the point.
2. Create the module named in the card. Write the smallest version that passes the *first* test, then the next, and so on.
3. Run only that card's tests: `pytest tests/test_<name>.py -q`. They start as `xfail` (expected failure). As you implement them they show up as `XPASS`.
4. When **all** tests of a card pass, delete the `pytestmark = pytest.mark.xfail(...)` line at the top of the test file. From then on a failure is a real failure and CI will catch it.
5. Run `ruff check .` and `ruff format .`, then commit with a prefix: `feat:`, `fix:`, `test:`, `docs:`.
6. Add three lines to `docs/progress_log.md` (built, broke, learned).
7. Ask me to review. I will read your code, point out problems, and then ask you a few questions (the explain-back). The card is done when you can answer them.

If you are stuck for more than 30 minutes, paste the error and what you tried. I will give a hint, not the answer.

## Cards

| Card | What you build | Concepts |
|---|---|---|
| [E0-1](E0-1_async_downloads.md) | Concurrent, retrying downloader | `asyncio`, `httpx`, retries, backpressure |
| [E0-2](E0-2_settings.md) | Validated settings from the YAML configs | Pydantic models, `pydantic-settings`, failing fast |
| [E0-3](E0-3_structured_logging.md) | JSON logging and a `@timed` decorator | decorators, `structlog`, observability |

Also in E0: write `docs/adr/0001-research-core-has-no-web-dependencies.md` yourself.
