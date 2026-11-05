"""E0-3: structured logging and @timed. See docs/tasks/E0-3_structured_logging.md.

Marked xfail until you implement src/oilfc/logging_utils.py. When all tests pass, delete the pytestmark line.
"""

import asyncio
import json
import time

import pytest
import structlog
from structlog.testing import capture_logs

pytestmark = pytest.mark.xfail(reason="E0-3: implement oilfc.logging_utils", strict=False)


@pytest.fixture(autouse=True)
def _reset_structlog():
    yield
    structlog.reset_defaults()


def test_timed_logs_one_event_for_a_sync_function():
    from oilfc.logging_utils import timed

    @timed
    def work(x):
        time.sleep(0.05)
        return x * 2

    with capture_logs() as logs:
        assert work(3) == 6
    assert len(logs) == 1
    e = logs[0]
    assert e["event"] == "timed"
    assert e["func"] == "work"
    assert e["status"] == "ok"
    assert e["duration_ms"] >= 40


def test_timed_keeps_the_function_name_and_docstring():
    from oilfc.logging_utils import timed

    @timed
    def documented():
        """I am documented."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "I am documented."


def test_timed_logs_the_error_and_reraises_it():
    from oilfc.logging_utils import timed

    @timed
    def boom():
        raise ValueError("nope")

    with capture_logs() as logs:
        with pytest.raises(ValueError, match="nope"):
            boom()
    assert len(logs) == 1
    assert logs[0]["status"] == "error"
    assert logs[0]["error_type"] == "ValueError"


def test_timed_supports_async_functions():
    from oilfc.logging_utils import timed

    @timed
    async def slow():
        await asyncio.sleep(0.05)
        return "done"

    with capture_logs() as logs:
        assert asyncio.run(slow()) == "done"
    assert len(logs) == 1
    assert logs[0]["func"] == "slow"
    assert logs[0]["duration_ms"] >= 40


def test_configure_logging_prints_one_json_object_per_line(capsys):
    from oilfc.logging_utils import configure_logging

    configure_logging(level="INFO", json=True)
    structlog.get_logger().info("hello", a=1)
    line = capsys.readouterr().out.strip().splitlines()[-1]
    data = json.loads(line)
    assert data["event"] == "hello"
    assert data["a"] == 1
    assert data["level"] == "info"
    assert "timestamp" in data


def test_configure_logging_drops_events_below_the_level(capsys):
    from oilfc.logging_utils import configure_logging

    configure_logging(level="WARNING", json=True)
    log = structlog.get_logger()
    log.info("quiet")
    log.warning("loud")
    out = capsys.readouterr().out
    assert "loud" in out
    assert "quiet" not in out
