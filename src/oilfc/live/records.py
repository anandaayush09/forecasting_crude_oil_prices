"""Write-once forecast records: JSON with a SHA-256 of the content and a manifest row.

Shared by the primary live track (`oilfc.live.forecast`) and the foundation-model track
(`oilfc.live.ttm`). Needs only the standard library and pandas, so the foundation-model environment
(which cannot hold the full requirements) can write and check records in the same format.
"""

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

MANIFEST = "MANIFEST.csv"


def _digest(content: dict) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_forecast(rec: dict, out_dir: Path, run_utc: str | None = None) -> Path:
    """Write the record once as {asset}_{origin}.json with its SHA-256 and add a manifest row.

    Raises FileExistsError if a forecast for this asset and origin already exists: frozen forecasts are
    never overwritten.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_utc = run_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = {**rec, "run_utc": run_utc}
    sha = _digest(content)
    path = out_dir / f"{rec['asset']}_{rec['origin_date']}.json"
    with open(path, "x", encoding="utf-8") as f:  # "x": fail if the file exists
        f.write(json.dumps({"content": content, "sha256": sha}, indent=1, sort_keys=True) + "\n")
    manifest = out_dir / MANIFEST
    new = not manifest.exists()
    with open(manifest, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(["asset", "origin_date", "run_utc", "sha256", "code_version"])
        w.writerow([rec["asset"], rec["origin_date"], run_utc, sha, rec.get("code_version") or ""])
    return path


def verify_forecast(path: Path) -> bool:
    """True if the file's content matches its stored hash and, when listed, the manifest hash."""
    path = Path(path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    sha = _digest(stored["content"])
    if sha != stored["sha256"]:
        return False
    manifest = path.parent / MANIFEST
    if manifest.exists():
        m = pd.read_csv(manifest, dtype=str)
        c = stored["content"]
        hit = m[(m["asset"] == c["asset"]) & (m["origin_date"] == c["origin_date"])]
        return len(hit) == 1 and hit["sha256"].iloc[0] == sha
    return True
