"""Fetch the daily futures OHLC files used by experiments/vol_range.py (ADR 0006).

    .venv/Scripts/python.exe experiments/fetch_ohlc.py

Writes data/raw/ohlc_CLF.csv (WTI, CL=F) and data/raw/ohlc_BZF.csv (Brent, BZ=F). The files are not
tracked; later downloads add newer days, which the development experiments trim away.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.data.yahoo import fetch_ohlc  # noqa: E402


def main():
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for symbol, name in (("CL=F", "ohlc_CLF.csv"), ("BZ=F", "ohlc_BZF.csv")):
        frame = fetch_ohlc(symbol)
        frame.to_csv(out / name)
        print(
            f"{symbol}: {len(frame)} rows, {frame.index[0].date()} to {frame.index[-1].date()} -> {out / name}"
        )


if __name__ == "__main__":
    main()
