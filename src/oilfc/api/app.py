"""Read-only API over live/forecasts, their manifest and live/scores.csv. See docs/api_spec.md."""

import csv
import json
import os
from datetime import date
from enum import StrEnum
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from pydantic import BaseModel

from oilfc.live.forecast import MANIFEST, verify_forecast

ROOT = Path(__file__).resolve().parents[3]
DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


class Asset(StrEnum):
    brent = "brent"
    wti = "wti"


class ForecastSummary(BaseModel):
    asset: str
    origin_date: str
    run_utc: str
    sha256: str
    code_version: str


class ForecastRecord(BaseModel):
    content: dict
    sha256: str
    verified: bool


class ScoreRow(BaseModel):
    asset: str
    origin_date: str
    horizon: int
    model: str
    forecast: float
    realised: float | None
    qlike: float | None
    run_utc: str


class Scores(BaseModel):
    scored: int
    rows: list[ScoreRow]


def create_app(live_dir: Path | str | None = None) -> FastAPI:
    """Build the app. live_dir defaults to $OILFC_LIVE_DIR, then to the repository's live/ folder."""
    live = Path(live_dir or os.environ.get("OILFC_LIVE_DIR") or ROOT / "live")
    forecasts_dir = live / "forecasts"
    app = FastAPI(title="Crude oil volatility: live forward test", version="1")

    def manifest_rows(asset: Asset | None = None) -> list[dict]:
        path = forecasts_dir / MANIFEST
        if not path.exists():
            return []
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if asset is not None:
            rows = [r for r in rows if r["asset"] == asset.value]
        return sorted(rows, key=lambda r: (r["origin_date"], r["run_utc"]), reverse=True)

    def load(asset: Asset, origin: str) -> ForecastRecord:
        path = forecasts_dir / f"{asset.value}_{origin}.json"
        if not path.exists():
            raise HTTPException(404, f"no {asset.value} forecast for origin {origin}")
        if not verify_forecast(path):
            raise HTTPException(500, "the stored forecast failed its hash check and is not served")
        stored = json.loads(path.read_text(encoding="utf-8"))
        return ForecastRecord(content=stored["content"], sha256=stored["sha256"], verified=True)

    @app.get("/health")
    def health() -> dict:
        rows = manifest_rows()
        latest = {
            a.value: next((r["origin_date"] for r in rows if r["asset"] == a.value), None)
            for a in Asset
        }
        return {"status": "ok", "forecast_files": len(rows), "latest_origin": latest}

    @app.get("/v1/forecasts", response_model=list[ForecastSummary])
    def list_forecasts(asset: Asset | None = None):
        return manifest_rows(asset)

    @app.get("/v1/forecasts/latest/{asset}", response_model=ForecastRecord)
    def latest_forecast(asset: Asset):
        rows = manifest_rows(asset)
        if not rows:
            raise HTTPException(404, f"no {asset.value} forecasts yet")
        return load(asset, rows[0]["origin_date"])

    @app.get("/v1/forecasts/{asset}/{origin_date}", response_model=ForecastRecord)
    def one_forecast(asset: Asset, origin_date: str = PathParam(pattern=DATE_PATTERN)):
        try:
            date.fromisoformat(origin_date)
        except ValueError:
            raise HTTPException(422, "origin_date must be a real date, YYYY-MM-DD") from None
        return load(asset, origin_date)

    @app.get("/v1/scores", response_model=Scores)
    def scores(
        asset: Asset | None = None,
        horizon: int | None = Query(None, description="5, 22 or 66"),
        scored_only: bool = False,
    ):
        if horizon is not None and horizon not in (5, 22, 66):
            raise HTTPException(422, "horizon must be 5, 22 or 66")
        path = live / "scores.csv"
        if not path.exists():
            return Scores(scored=0, rows=[])
        df = pd.read_csv(path, dtype={"origin_date": str})
        if asset is not None:
            df = df[df["asset"] == asset.value]
        if horizon is not None:
            df = df[df["horizon"] == horizon]
        if scored_only:
            df = df[df["qlike"].notna()]
        rows = df.astype(object).where(df.notna(), None).to_dict("records")
        return Scores(scored=int(df["qlike"].notna().sum()), rows=rows)

    return app


app = create_app()
