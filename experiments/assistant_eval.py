"""Evaluate the research assistant on the gold questions (ADR 0011).

    python experiments/assistant_eval.py --split dev     # tuning allowed
    python experiments/assistant_eval.py --split test    # run once, after the settings are fixed

Ranking quality: recall at 1, 3 and 5 and mean reciprocal rank over the answerable questions of the
split, with the refusal threshold switched off (a hit is a passage from any listed file). Refusal: the
share of the unanswerable questions refused, and the share of the answerable ones still answered, at the
threshold. Writes results/tables/assistant_retrieval_<split>.csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.assistant.search import MIN_SCORE, search  # noqa: E402


def evaluate(split: str) -> pd.DataFrame:
    gold = yaml.safe_load((ROOT / "docs" / "assistant_gold.yaml").read_text(encoding="utf-8"))
    rows = []
    for g in (q for q in gold["questions"] if q["split"] == split):
        hits = search(g["q"], k=5, min_score=0.0)
        rank = next((i for i, h in enumerate(hits, 1) if h.passage.path in g["expected"]), None)
        gated = search(g["q"], k=1, min_score=MIN_SCORE)
        rows.append({"id": g["id"], "kind": "answerable", "rank": rank, "answered": bool(gated),
                     "top_file": hits[0].passage.path, "top_score": hits[0].score})  # fmt: skip
    for g in (q for q in gold["unanswerable"] if q["split"] == split):
        hits = search(g["q"], k=1, min_score=0.0)
        rows.append({"id": g["id"], "kind": "unanswerable", "rank": None,
                     "answered": bool(search(g["q"], k=1, min_score=MIN_SCORE)),
                     "top_file": hits[0].passage.path, "top_score": hits[0].score})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    split = ap.parse_args().split
    d = evaluate(split)
    a, u = d[d.kind == "answerable"], d[d.kind == "unanswerable"]
    out = {
        "split": split, "n_answerable": len(a), "min_score": MIN_SCORE,
        "recall_at_1": (a["rank"] <= 1).mean(), "recall_at_3": (a["rank"] <= 3).mean(),
        "recall_at_5": (a["rank"] <= 5).mean(), "mrr": (1 / a["rank"]).fillna(0).mean(),
        "answered_share_of_answerable": a["answered"].mean(), "n_unanswerable": len(u),
        "refused_share_of_unanswerable": 1 - u["answered"].mean(),
    }  # fmt: skip
    pd.set_option("display.width", 200)
    print(d.round(3).to_string(index=False))
    print(
        pd.Series(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}
        ).to_string()
    )
    table = ROOT / "results" / "tables" / f"assistant_retrieval_{split}.csv"
    pd.DataFrame([out]).to_csv(table, index=False)
    print("wrote", table)


if __name__ == "__main__":
    main()
