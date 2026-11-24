"""Write the frozen live forecasts for the latest origin (one file per asset, never overwritten).

    .venv/Scripts/python.exe experiments/live_run.py [--no-refresh]

Refreshes the data panel (network) unless --no-refresh, then writes live/forecasts/{asset}_{origin}.json
and a row in live/forecasts/MANIFEST.csv for Brent and WTI. If a forecast for that origin already
exists it is left alone. Commit the new files right away: the git commit time is the evidence that the
forecast was fixed before its target window ended. See live/README.md.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.live.forecast import HORIZONS, live_forecasts, write_forecast  # noqa: E402

OUT = ROOT / "live" / "forecasts"


def code_version() -> str:
    """Git commit of the code, with '+dirty' if the code, config or experiments have uncommitted edits."""

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--", "src", "configs", "experiments")
    return git("rev-parse", "HEAD") + ("+dirty" if dirty else "")


def main():
    if "--no-refresh" not in sys.argv:
        from oilfc.data.build import build

        build(offline=False)
    panel = pd.read_parquet(ROOT / "data" / "processed" / "panel_daily.parquet")
    panel.index = pd.to_datetime(panel.index)
    version = code_version()
    print(f"data through {panel.index.max().date()}, code {version[:12]}")
    for asset in ("brent", "wti"):
        rec = live_forecasts(panel, asset, HORIZONS, code_version=version)
        try:
            path = write_forecast(rec, OUT)
            print(f"wrote {path.relative_to(ROOT)}")
        except FileExistsError:
            print(
                f"{asset}: forecast for origin {rec['origin_date']} already exists, left unchanged"
            )
            continue
        for h, models in rec["horizons"].items():
            print(f"  {asset} {h}d: " + ", ".join(f"{m} {v:.3f}" for m, v in models.items()))


if __name__ == "__main__":
    main()
