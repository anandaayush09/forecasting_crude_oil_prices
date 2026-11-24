# API specification (read-only)

Status: draft written before the code. Implementation: `src/oilfc/api/app.py`. Tests: `tests/test_api.py`.

## Purpose and users

A small read-only HTTP layer over the frozen live forecasts (`live/forecasts/`), their manifest and the generated scores (`live/scores.csv`). The users are the author (scripts, dashboards) and, later, readers of the paper who want to check the forward test. It never fits a model, never writes a file and never calls an outside service.

## Non-goals

No write endpoints, no authentication (it is meant to run on localhost or behind a reverse proxy that adds it), no model fitting on request, no historical backtest results (those are the files in `results/`), no web page.

## Endpoints

| Method and path | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "forecast_files": int, "latest_origin": {"brent": "YYYY-MM-DD" or null, "wti": ...}}` |
| `GET /v1/forecasts?asset=brent` | list of manifest rows, newest origin first: `asset`, `origin_date`, `run_utc`, `sha256`, `code_version` |
| `GET /v1/forecasts/latest/{asset}` | the newest frozen forecast of that asset |
| `GET /v1/forecasts/{asset}/{origin_date}` | one frozen forecast |
| `GET /v1/scores?asset=&horizon=&scored_only=` | rows of `live/scores.csv`, optionally filtered |

`asset` is `brent` or `wti`. `origin_date` is `YYYY-MM-DD`. `horizon` is 5, 22 or 66.

A forecast response is `{"content": {...}, "sha256": "...", "verified": true}`. `content` is exactly the stored object: asset, origin date, the OVX date used, horizons mapped to models mapped to annualised volatility (decimal), code version, run time. The file hash and the manifest hash are checked on every read.

A score row has `asset`, `origin_date`, `horizon`, `model`, `forecast`, `realised` (null until the window has ended), `qlike` (null likewise), `run_utc`. `scored_only=true` drops rows with a null `qlike`.

## Errors

| Status | When |
|---|---|
| 404 | no forecast for that asset and date, or none exist yet for `latest` |
| 422 | `asset` not in the allowed set, `origin_date` not a real date, `horizon` not 5, 22 or 66 |
| 405 | any method other than GET |
| 500 | a stored forecast fails its hash check. The body says so and the content is **not** returned |

Every error body is `{"detail": "..."}`.

## Safety rules

- File names are never built from user input: the asset comes from a fixed set and the date is parsed as a date, then looked up in the manifest.
- The live folder comes from the environment variable `OILFC_LIVE_DIR` (default: `live/` in the repository).
- No data are cached between requests, so a new forecast file appears immediately.

## Running

```bash
.venv/Scripts/python.exe -m uvicorn oilfc.api.app:app --port 8000
```
