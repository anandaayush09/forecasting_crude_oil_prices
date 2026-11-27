"""Ask the project's own documents a question (ADR 0011). Local, offline, read-only.

    python experiments/ask.py "Which TTM model revision is pinned?" [-k 3]

Prints the best passages with their file, heading and score, or says that the documents hold no answer.
The tool quotes the documents. It can be out of date, so check the cited file.
"""

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oilfc.assistant import search  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=3, help="number of passages (default 3)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # documents hold non-ASCII symbols
    hits = search(args.question, args.k)
    if not hits:
        print("The documents do not contain an answer to this question.")
        return
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h.passage.path}  >  {h.passage.heading}  (score {h.score:.2f})")
        print(textwrap.indent(textwrap.fill(h.passage.text, 100), "    "))
        print()


if __name__ == "__main__":
    main()
