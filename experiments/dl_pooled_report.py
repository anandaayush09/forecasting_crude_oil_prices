"""Pre-registered tests for Stage B (docs/adr/0004-deep-learning-addendum.md, amendment of 2026-11-27).

    python experiments/dl_pooled_report.py [folder]

Reads the per-origin predictions written by vol_dl_pooled.py (default results/dl_pooled/) and, for each
cell, the Stage A per-asset predictions (results/kaggle_dl_addendum/). Each pooled variant is tested
one-sided against the Stage A per-asset version of the same architecture (pooled MLP against `mlp_seq`,
pooled transformer against `transformer_seq`): 2 variants times 6 cells = 12 tests, Holm-adjusted
together, with Hansen's SPA per cell over the 2 variants (block length h, 1,000 repetitions, seed 0). A
variant helps in a cell only if its Holm p and the cell's SPA p are both below 0.05. The comparison of
each pooled variant with the headline benchmark is reported next to it and is outside the family.
Writes results/tables/dl_pooled.csv.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from dl_addendum_report import one_sided_p  # noqa: E402

from oilfc.eval.losses import qlike  # noqa: E402
from oilfc.eval.multiplicity import holm_adjust  # noqa: E402
from oilfc.eval.spa import spa_pvalue  # noqa: E402

PAIRS = {"pooled_mlp": "mlp_seq", "pooled_transformer": "transformer_seq"}
STAGE_A = ROOT / "results" / "kaggle_dl_addendum"


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "dl_pooled"
    rows, spa = [], {}
    for f in sorted(folder.glob("dl_pooled_predictions_*.csv")):
        m = re.match(r"dl_pooled_predictions_(brent|wti)_(\d+)d\.csv$", f.name)
        if not m:
            continue  # skips smoke files
        asset, h = m.group(1), int(m.group(2))
        pooled = pd.read_csv(f, index_col=0)
        stage_a = pd.read_csv(STAGE_A / f"dl_addendum_predictions_{asset}_{h}d.csv", index_col=0)
        df = pooled.join(stage_a[list(PAIRS.values())], how="inner")
        if len(df) != len(pooled) or len(df) != len(stage_a):
            sys.exit(f"{asset} {h}d: origins differ ({len(pooled)} pooled, {len(stage_a)} Stage A)")
        if not np.allclose(df["benchmark"], stage_a["benchmark"], rtol=1e-9):
            sys.exit(f"{asset} {h}d: the benchmark differs from Stage A")
        q = {c: qlike(df["y"], df[c]) for c in df.columns if c != "y"}
        diffs = {}
        for v, ref in PAIRS.items():
            p = one_sided_p(q[ref], q[v], h)
            rows.append({
                "asset": asset, "horizon": h, "variant": v, "n": len(df),
                "qlike": q[v].mean(), "qlike_per_asset": q[ref].mean(),
                "gain_vs_per_asset_pct": 100 * (1 - q[v].mean() / q[ref].mean()),
                "dm_p": p if p == p else 1.0,
                "gain_vs_headline_pct": 100 * (1 - q[v].mean() / q["benchmark"].mean()),
                "dm_p_vs_headline": one_sided_p(q["benchmark"], q[v], h),
            })  # fmt: skip
            diffs[v] = q[v] - q[ref]
        spa[(asset, h)] = spa_pvalue(np.zeros(len(df)), pd.DataFrame(diffs), block_size=h)
    if not rows:
        sys.exit(f"no prediction files in {folder}")
    t = pd.DataFrame(rows)
    t["dm_p_holm"] = holm_adjust(t["dm_p"].to_numpy())
    t["spa_p_cell"] = [spa[(a, h)] for a, h in zip(t["asset"], t["horizon"], strict=True)]
    t["helps"] = (t["dm_p_holm"] < 0.05) & (t["spa_p_cell"] < 0.05)
    t.to_csv(ROOT / "results" / "tables" / "dl_pooled.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(f"cells: {sorted(spa)}; tests in the Holm family: {len(t)}")
    print(t.round(4).to_string(index=False))
    print("variants that help (Holm p < 0.05 and SPA p < 0.05):", int(t["helps"].sum()))


if __name__ == "__main__":
    main()
