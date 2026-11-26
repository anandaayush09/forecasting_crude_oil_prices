"""Pre-registered tests for the tuned architectures (docs/adr/0007-deep-learning-architectures-tuned.md).

    python experiments/dl_arch_report.py [folder]

Reads the per-origin predictions written by vol_dl_arch.py (default results/dl_arch/). The six
architectures are the Holm family (36 tests over six cells) and the SPA set per cell; the ridge control
on the 12-channel window is shown outside the family. Writes results/tables/dl_arch.csv.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from dl_addendum_report import report  # noqa: E402

from oilfc.models.dl.encoders import ARCHITECTURES  # noqa: E402


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dl_arch"
    report(folder, "dl_arch", list(ARCHITECTURES), ["ridge_window"], "dl_arch.csv")


if __name__ == "__main__":
    main()
