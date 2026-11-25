"""Build kaggle/oilfc-residual.zip: the code, data and script the Kaggle notebook needs.

    .venv/Scripts/python.exe kaggle/build_upload.py

Upload the zip as a Kaggle dataset named `oilfc-residual` (Datasets > New Dataset > upload the file),
then open notebooks/kaggle_transformer_residual.ipynb or notebooks/kaggle_dl_addendum.ipynb in a Kaggle
notebook with that dataset attached (a new version of the same dataset works for both).
"""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "kaggle" / "oilfc-residual.zip"
FILES = [
    ROOT / "experiments" / "vol_residual_seq.py",
    ROOT / "experiments" / "vol_dl_addendum.py",
    ROOT / "data" / "processed" / "panel_daily.parquet",
]

README = """oilfc-residual: code and data for notebooks/kaggle_transformer_residual.ipynb.
src/oilfc/        research code (pure Python; torch, pandas, scikit-learn, scipy needed; `arch` optional)
experiments/      vol_residual_seq.py and vol_dl_addendum.py (the experiments)
data/processed/   panel_daily.parquet (Brent, WTI, OVX, VIX, ... with release-date alignment)
"""


def main():
    missing = [f for f in FILES if not f.exists()]
    assert not missing, f"missing: {missing} (run the data build first)"
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "src" / "oilfc").rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
        for p in FILES:
            z.write(p, p.relative_to(ROOT).as_posix())
        z.writestr("README.txt", README)
    names = zipfile.ZipFile(OUT).namelist()
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main()
