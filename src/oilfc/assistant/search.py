"""TF-IDF retrieval over the passages (ADR 0011): deterministic, offline, no model download."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from oilfc.assistant.corpus import Passage, load_passages

ROOT = Path(__file__).resolve().parents[3]
# below this cosine similarity the question is refused (midpoint of the development scores)
MIN_SCORE = 0.10
NGRAMS = (1, 2)
HEADING_WEIGHT = 1  # the heading is repeated this many times before the text


@dataclass(frozen=True)
class Hit:
    """A retrieved passage with its cosine similarity to the question (0 to 1)."""

    passage: Passage
    score: float


class Index:
    """TF-IDF index of a list of passages. Build once, query many times."""

    def __init__(
        self,
        passages: list[Passage],
        ngrams: tuple[int, int] = NGRAMS,
        heading_weight: int = HEADING_WEIGHT,
    ):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True, stop_words="english", ngram_range=ngrams, lowercase=True
        )
        self.matrix = self.vectorizer.fit_transform(
            [f"{(p.heading + ' ') * heading_weight}{p.text}" for p in passages]
        )

    def search(self, question: str, k: int = 3, min_score: float = MIN_SCORE) -> list[Hit]:
        """Top `k` passages by cosine similarity; empty if even the best is below `min_score`.

        Ties are broken by passage order, so the ranking is the same on every run.
        """
        q = self.vectorizer.transform([question])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        if len(order) == 0 or scores[order[0]] < min_score:
            return []
        return [Hit(self.passages[i], float(scores[i])) for i in order]


_cache: dict[Path, Index] = {}


def search(question: str, k: int = 3, root: Path = ROOT, min_score: float = MIN_SCORE) -> list[Hit]:
    """Search the public documents under `root`. An empty list means the documents hold no answer."""
    root = Path(root)
    if root not in _cache:
        _cache[root] = Index(load_passages(root))
    return _cache[root].search(question, k, min_score)
