"""Public documents of the repository, cut into passages (ADR 0011).

Only an allow-list of public files is read. Private notes, the PDFs of other authors, raw data and
every result table except the generated summary are never indexed.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 200
# patterns relative to the repository root
INCLUDE = (
    "README.md",
    "ROADMAP.md",
    "docs/**/*.md",
    "live/README.md",
    "literature/notes/*.md",
    "configs/*.yaml",
    "results/tables/SUMMARY.md",
)
LIT_MATRIX = "literature/lit_matrix.csv"
EXCLUDED_PARTS = {".project-local", "prior_work", "graphify-out", "data", ".venv", "node_modules"}


@dataclass(frozen=True)
class Passage:
    """One indexed passage: where it comes from and its text."""

    path: str  # relative to the repository root, with forward slashes
    heading: str
    text: str


def _is_allowed(rel: Path) -> bool:
    return not (set(rel.parts) & EXCLUDED_PARTS)


def split_markdown(text: str, path: str, max_words: int = MAX_WORDS) -> list[Passage]:
    """Cut markdown at headings, then split long sections into pieces of at most `max_words` words."""
    sections, heading, buf = [], Path(path).name, []
    for line in text.replace("\r\n", "\n").split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            sections.append((heading, buf))
            heading, buf = m.group(2), []
        else:
            buf.append(line)
    sections.append((heading, buf))
    out = []
    for head, lines in sections:
        words = " ".join(lines).split()
        for i in range(0, max(len(words), 1), max_words):
            piece = " ".join(words[i : i + max_words])
            if piece:
                out.append(Passage(path, head, piece))
    return out


def lit_matrix_passages(csv_path: Path, path: str) -> list[Passage]:
    """One passage per paper of the literature matrix: its citation, target, model family and notes."""
    out = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            keep = ("target", "frequency", "horizons", "model_family", "reported_gain", "notes")
            text = " ".join(f"{k}: {row[k]}" for k in keep if row.get(k) and row[k] != "n/a")
            out.append(Passage(path, row["citation"], f"{row['citation']}. {text}"))
    return out


def load_passages(root: Path) -> list[Passage]:
    """All passages of the allow-listed public files under `root`, in a fixed order."""
    root = Path(root)
    files = sorted({p for pat in INCLUDE for p in root.glob(pat) if p.is_file()})
    out = []
    for p in files:
        rel = p.relative_to(root)
        if _is_allowed(rel):
            out += split_markdown(p.read_text(encoding="utf-8"), rel.as_posix())
    matrix = root / LIT_MATRIX
    if matrix.is_file():
        out += lit_matrix_passages(matrix, LIT_MATRIX)
    return out
