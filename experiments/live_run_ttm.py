"""Write the frozen foundation-model forecasts for the latest origin (ADR 0010, protocol v2).

    python experiments/live_run_ttm.py [--check]

Needs the data panel and the primary forecast of the same asset and origin (run `live_run.py` first).
Writes live/forecasts_ttm/{asset}_{origin}.json and a manifest row, never overwriting. Origins on or
before the protocol-v2 tag date are skipped. With --check nothing is loaded or written: it prints
`needed=true` when a forecast is due and sets the `needed` output when run in GitHub Actions.
Runs in its own environment (requirements-ttm.txt): torch and granite-tsfm cannot share the main one.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.live.records import verify_forecast, write_forecast  # noqa: E402
from oilfc.live.ttm import (  # noqa: E402
    FIRST_ORIGIN_AFTER,
    NotScoredOrigin,
    TTMPredictor,
    returns_of,
    ttm_forecasts,
)

PRIMARY = ROOT / "live" / "forecasts"
OUT = ROOT / "live" / "forecasts_ttm"
ASSETS = ("brent", "wti")


def code_version() -> str:
    """Git commit of the code, with '+dirty' if the code, config or experiments have uncommitted edits."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--", "src", "configs", "experiments")
    return git("rev-parse", "HEAD") + ("+dirty" if dirty else "")


def due(panel: pd.DataFrame) -> dict[str, dict]:
    """Assets with a forecast to write: {asset: stored primary record} for the latest origin."""
    out = {}
    for asset in ASSETS:
        origin = returns_of(panel, asset).index[-1]
        name = f"{asset}_{origin.date()}.json"
        if origin <= FIRST_ORIGIN_AFTER or (OUT / name).exists() or not (PRIMARY / name).exists():
            continue
        if not verify_forecast(PRIMARY / name):
            raise ValueError(f"{name} does not match its stored hash or the manifest")
        out[asset] = json.loads((PRIMARY / name).read_text(encoding="utf-8"))
    return out


def main():
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    todo = due(panel)
    print(f"data through {panel.index.max().date()}, forecasts due: {sorted(todo) or 'none'}")
    if "--check" in sys.argv:
        needed = "true" if todo else "false"
        print(f"needed={needed}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
                f.write(f"needed={needed}\n")
        return
    if not todo:
        return
    predictor, version = TTMPredictor(), code_version()
    for asset, head in todo.items():
        try:
            rec = ttm_forecasts(panel, asset, head, predictor, code_version=version)
        except NotScoredOrigin as e:
            print(f"{asset}: skipped ({e})")
            continue
        path = write_forecast(rec, OUT)
        print(f"wrote {path.relative_to(ROOT)}")
        for h, models in rec["horizons"].items():
            print(f"  {asset} {h}d: " + ", ".join(f"{m} {v:.3f}" for m, v in models.items()))


if __name__ == "__main__":
    main()
