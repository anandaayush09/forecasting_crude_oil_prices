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

| [V1](V1_volatility_targets.md) | Forward and past realised-volatility targets | rolling windows, causality, annualisation |
| [V2](V2_losses.md) | QLIKE, log-vol MSE, normal CRPS | proper scoring rules, robust volatility losses |
| [V3](V3_dm_test.md) | Diebold-Mariano test (HLN form) | hypothesis tests, HAC variance, overlapping forecasts |
| [V4](V4_protocol_review.md) | Review and freeze `configs/protocol.yaml` | pre-registration, leakage by design |
| [V5](V5_rolling_splits.md) | Expanding-origin splits with a gap | walk-forward evaluation, purging |
| [V6](V6_vol_baselines.md) | Volatility ladder: naive, EWMA, HAR, OVX, HAR + OVX | log-linear forecasting, reproducing a result |

Suggested order: V1, V2, V3, V5, V6 (V4 in parallel this week). V6's last test needs V1, V2, V3 and V5.

Also in E0: write `docs/adr/0001-research-core-has-no-web-dependencies.md` yourself. For the volatility pivot, write `docs/adr/0002-volatility-as-main-target.md`.
