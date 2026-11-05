"""E0-1: concurrent, retrying downloads. See docs/tasks/E0-1_async_downloads.md.

Marked xfail until you implement src/oilfc/data/async_fetch.py. When all tests pass, delete the pytestmark line.
No test touches the real network: httpx.MockTransport fakes the server.
"""

import asyncio
import time

import httpx
import pytest

pytestmark = pytest.mark.xfail(reason="E0-1: implement oilfc.data.async_fetch", strict=False)

URL = "https://example.test/data"


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _run(coro_fn):
    return asyncio.run(coro_fn())


def test_fetch_text_returns_body():
    from oilfc.data.async_fetch import fetch_text

    async def handler(request):
        return httpx.Response(200, text="hello")

    async def go():
        async with _client(handler) as c:
            return await fetch_text(c, URL)

    assert _run(go) == "hello"


def test_fetch_text_retries_server_errors_then_succeeds():
    from oilfc.data.async_fetch import fetch_text

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, text="ok")

    async def go():
        async with _client(handler) as c:
            return await fetch_text(c, URL, retries=3, backoff=0.0)

    assert _run(go) == "ok"
    assert calls["n"] == 3


def test_fetch_text_gives_up_after_retries_and_raises():
    from oilfc.data.async_fetch import fetch_text

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    async def go():
        async with _client(handler) as c:
            return await fetch_text(c, URL, retries=2, backoff=0.0)

    with pytest.raises(httpx.HTTPStatusError):
        _run(go)
    assert calls["n"] == 3  # the first attempt plus two retries


def test_fetch_text_does_not_retry_client_errors():
    from oilfc.data.async_fetch import fetch_text

    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    async def go():
        async with _client(handler) as c:
            return await fetch_text(c, URL, retries=3, backoff=0.0)

    with pytest.raises(httpx.HTTPStatusError):
        _run(go)
    assert calls["n"] == 1


def test_fetch_many_returns_results_keyed_like_the_input():
    from oilfc.data.async_fetch import fetch_many

    async def handler(request):
        return httpx.Response(200, text=request.url.path)

    urls = {"a": "https://example.test/a", "b": "https://example.test/b"}

    async def go():
        async with _client(handler) as c:
            return await fetch_many(c, urls, backoff=0.0)

    assert _run(go) == {"a": "/a", "b": "/b"}


def test_fetch_many_runs_requests_concurrently():
    from oilfc.data.async_fetch import fetch_many

    async def handler(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, text="x")

    urls = {str(i): f"https://example.test/{i}" for i in range(5)}

    async def go():
        async with _client(handler) as c:
            return await fetch_many(c, urls, concurrency=5, backoff=0.0)

    start = time.perf_counter()
    assert len(_run(go)) == 5
    elapsed = time.perf_counter() - start
    assert elapsed < 0.7  # sequential would take at least 1.0 s


def test_fetch_many_never_exceeds_the_concurrency_limit():
    from oilfc.data.async_fetch import fetch_many

    state = {"now": 0, "max": 0}

    async def handler(request):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.05)
        state["now"] -= 1
        return httpx.Response(200, text="x")

    urls = {str(i): f"https://example.test/{i}" for i in range(8)}

    async def go():
        async with _client(handler) as c:
            return await fetch_many(c, urls, concurrency=2, backoff=0.0)

    assert len(_run(go)) == 8
    assert state["max"] == 2
