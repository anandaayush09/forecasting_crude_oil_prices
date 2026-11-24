import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from oilfc.api.app import create_app
from oilfc.live.forecast import MODELS, write_forecast


def _rec(asset="brent", origin="2026-11-17", base=0.5):
    return {
        "asset": asset,
        "origin_date": origin,
        "ovx_date_used": "2026-11-16",
        "ovx_value_used": 49.6,
        "n_returns": 9000,
        "horizons": {
            h: {m: round(base + 0.01 * i, 8) for i, m in enumerate(MODELS)} for h in ("5", "22")
        },
        "code_version": "abc123",
        "spec_version": "1",
    }


@pytest.fixture()
def live(tmp_path):
    forecasts = tmp_path / "forecasts"
    write_forecast(_rec("brent", "2026-11-10", 0.4), forecasts, run_utc="2026-11-11T10:00:00Z")
    write_forecast(_rec("brent", "2026-11-17", 0.5), forecasts, run_utc="2026-11-24T10:00:00Z")
    write_forecast(_rec("wti", "2026-11-17", 0.45), forecasts, run_utc="2026-11-24T10:00:01Z")
    return tmp_path


@pytest.fixture()
def client(live):
    return TestClient(create_app(live))


def test_health_with_no_forecasts_yet(tmp_path):
    out = TestClient(create_app(tmp_path)).get("/health").json()
    assert out == {
        "status": "ok",
        "forecast_files": 0,
        "latest_origin": {"brent": None, "wti": None},
    }


def test_health_reports_counts_and_latest_origins(client):
    out = client.get("/health").json()
    assert out["forecast_files"] == 3
    assert out["latest_origin"] == {"brent": "2026-11-17", "wti": "2026-11-17"}


def test_listing_is_newest_first_and_can_filter_by_asset(client):
    rows = client.get("/v1/forecasts").json()
    assert [r["origin_date"] for r in rows if r["asset"] == "brent"] == ["2026-11-17", "2026-11-10"]
    assert set(rows[0]) == {"asset", "origin_date", "run_utc", "sha256", "code_version"}
    only = client.get("/v1/forecasts", params={"asset": "wti"}).json()
    assert [r["asset"] for r in only] == ["wti"]


def test_latest_returns_the_newest_verified_forecast(client):
    out = client.get("/v1/forecasts/latest/brent").json()
    assert out["verified"] is True
    assert out["content"]["origin_date"] == "2026-11-17"
    assert out["content"]["horizons"]["5"]["naive"] == pytest.approx(0.5)
    assert len(out["sha256"]) == 64


def test_one_forecast_by_date_and_missing_ones_are_404(client):
    ok = client.get("/v1/forecasts/brent/2026-11-10")
    assert ok.status_code == 200 and ok.json()["content"]["horizons"]["5"][
        "naive"
    ] == pytest.approx(0.4)
    missing = client.get("/v1/forecasts/wti/2026-11-10")
    assert missing.status_code == 404 and "detail" in missing.json()


def test_latest_with_no_forecasts_is_404(tmp_path):
    assert TestClient(create_app(tmp_path)).get("/v1/forecasts/latest/brent").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/v1/forecasts/gold/2026-11-17",
        "/v1/forecasts/brent/not-a-date",
        "/v1/forecasts/brent/2026-13-45",
        "/v1/forecasts/latest/oil",
        "/v1/forecasts?asset=gold",
        "/v1/scores?horizon=7",
    ],
)
def test_bad_input_is_422(client, path):
    assert client.get(path).status_code == 422


def test_path_traversal_attempts_never_read_files(client, live):
    secret = live / "secret.json"
    secret.write_text("{}")
    for path in ("/v1/forecasts/brent/..%2Fsecret", "/v1/forecasts/..%2F..%2Fx/2026-11-17"):
        assert client.get(path).status_code in (404, 422)


def test_a_tampered_forecast_is_refused_not_served(client, live):
    path = live / "forecasts" / "brent_2026-11-17.json"
    stored = json.loads(path.read_text())
    stored["content"]["horizons"]["5"]["naive"] = 9.99
    path.write_text(json.dumps(stored))
    out = client.get("/v1/forecasts/latest/brent")
    assert out.status_code == 500
    assert "hash" in out.json()["detail"]
    assert "9.99" not in out.text


def test_the_api_is_read_only(client):
    for method in ("post", "put", "patch", "delete"):
        assert getattr(client, method)("/v1/forecasts/brent/2026-11-17").status_code == 405
    assert client.post("/v1/forecasts").status_code == 405


def test_scores_are_empty_before_scoring_and_filterable_after(client, live):
    assert client.get("/v1/scores").json() == {"scored": 0, "rows": []}
    pd.DataFrame(
        [
            {"asset": "brent", "origin_date": "2026-11-10", "horizon": 5, "model": "naive", "forecast": 0.4,
             "realised": 0.5, "qlike": 0.1, "run_utc": "2026-11-11T10:00:00Z"},
            {"asset": "brent", "origin_date": "2026-11-17", "horizon": 22, "model": "naive", "forecast": 0.5,
             "realised": None, "qlike": None, "run_utc": "2026-11-24T10:00:00Z"},
            {"asset": "wti", "origin_date": "2026-11-10", "horizon": 5, "model": "gjr_t", "forecast": 0.3,
             "realised": 0.35, "qlike": 0.05, "run_utc": "2026-11-11T10:00:01Z"},
        ]
    ).to_csv(live / "scores.csv", index=False)  # fmt: skip
    everything = client.get("/v1/scores").json()
    assert everything["scored"] == 2 and len(everything["rows"]) == 3
    assert [r["qlike"] for r in everything["rows"]] == [0.1, None, 0.05]
    only = client.get("/v1/scores", params={"asset": "brent", "horizon": 22}).json()
    assert len(only["rows"]) == 1 and only["rows"][0]["realised"] is None
    done = client.get("/v1/scores", params={"scored_only": "true"}).json()
    assert len(done["rows"]) == 2 and all(r["qlike"] is not None for r in done["rows"])
