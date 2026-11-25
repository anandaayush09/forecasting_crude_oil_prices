"""Pre-registered tests for the deep-learning addendum (docs/adr/0004-deep-learning-addendum.md).

    python experiments/dl_addendum_report.py [folder]

Reads the per-origin predictions written by vol_dl_addendum.py (default results/kaggle_dl_addendum/),
then for the six registered variants in every available cell: QLIKE gain over the benchmark, one-sided
Diebold-Mariano p (Harvey correction), Holm adjustment over all variant-cell tests together, and
Hansen's SPA per cell over the six variants (block length h, 1,000 repetitions, seed 0). A variant
helps in a cell only if Holm p < 0.05 and the cell's SPA p < 0.05. The shuffled-OVX placebo is shown
next to the table and is not part of the test family. Writes results/tables/dl_addendum.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.spa import spa_pvalue  # noqa: E402
from oilfc.eval.stat_tests import dm_hln  # noqa: E402

VARIANTS = [
    "ridge_window",
    "mlp_seq",
    "gru_seq",
    "transformer_seq",
    "transformer_eqw",
    "transformer_gate",
]
PLACEBO = "placebo_shuffled_ovx"


def one_sided_p(loss_benchmark, loss_model, h):
    """One-sided DM p-value that the model has the lower expected loss (dm_hln is two-sided)."""
    stat, p = dm_hln(loss_benchmark, loss_model, h)
    return p / 2 if stat > 0 else 1 - p / 2


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "kaggle_dl_addendum"
    rows, spa = [], {}
    for f in sorted(folder.glob("dl_addendum_predictions_*.csv")):
        m = re.match(r"dl_addendum_predictions_(brent|wti)_(\d+)d\.csv$", f.name)
        if not m:
            continue  # skips smoke files
        asset, h = m.group(1), int(m.group(2))
        df = pd.read_csv(f, index_col=0)
        q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
        for v in VARIANTS + ([PLACEBO] if PLACEBO in df.columns else []):
            rows.append({
                "asset": asset, "horizon": h, "variant": v, "n": len(df),
                "qlike": q[v].mean(), "qlike_benchmark": q["benchmark"].mean(),
                "gain_pct": 100 * (1 - q[v].mean() / q["benchmark"].mean()),
                "dm_p": one_sided_p(q["benchmark"], q[v], h),
            })  # fmt: skip
        spa[(asset, h)] = spa_pvalue(
            q["benchmark"], pd.DataFrame({v: q[v] for v in VARIANTS}), block_size=h
        )
    if not rows:
        sys.exit(f"no prediction files in {folder}")
    t = pd.DataFrame(rows)
    fam = t["variant"].isin(VARIANTS)
    t["dm_p_holm"] = np.nan
    t.loc[fam, "dm_p_holm"] = holm_adjust(t.loc[fam, "dm_p"].to_numpy())
    t["spa_p_cell"] = [spa[(a, h)] for a, h in zip(t["asset"], t["horizon"], strict=True)]
    t["helps"] = fam & (t["dm_p_holm"] < 0.05) & (t["spa_p_cell"] < 0.05)
    t.to_csv(ROOT / "results" / "tables" / "dl_addendum.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"cells: {sorted(spa)}; tests in the Holm family: {int(fam.sum())}")
    print(t.round(4).to_string(index=False))
    print("variants that help (Holm p < 0.05 and SPA p < 0.05):", int(t["helps"].sum()))


if __name__ == "__main__":
    main()
