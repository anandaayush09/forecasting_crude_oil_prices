"""Research assistant (ADR 0011): corpus guards, determinism and refusal, on synthetic documents."""

from pathlib import Path

import yaml

from oilfc.assistant.corpus import load_passages, split_markdown
from oilfc.assistant.search import search

ROOT = Path(__file__).resolve().parents[1]


def _repo(tmp_path):
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / ".project-local").mkdir()
    (tmp_path / "prior_work").mkdir()
    (tmp_path / "README.md").write_text(
        "# Title\nIntro about volatility.\n## Method\nHAR regression on OVX.\n"
    )
    (tmp_path / "docs" / "adr" / "0001.md").write_text(
        "# ADR one\n## Rule\nHolm correction over thirty tests.\n"
    )
    (tmp_path / ".project-local" / "secret.md").write_text("# Secret\nprivate diary entry\n")
    (tmp_path / "prior_work" / "old.md").write_text("# Old\nthesis draft\n")
    (tmp_path / "docs" / "adr" / "notes.md").write_text("# Notes\nGARCH with t errors benchmark.\n")
    return tmp_path


def test_markdown_is_cut_at_headings_and_long_sections_are_split():
    ps = split_markdown(
        "# A\none two\n## B\n" + " ".join(["w"] * 450) + "\n", "x.md", max_words=200
    )
    assert [p.heading for p in ps] == ["A", "B", "B", "B"]
    assert all(len(p.text.split()) <= 200 for p in ps)
    assert all(p.path == "x.md" for p in ps)


def test_private_and_excluded_paths_never_enter_the_index(tmp_path):
    paths = {p.path for p in load_passages(_repo(tmp_path))}
    assert paths == {"README.md", "docs/adr/0001.md", "docs/adr/notes.md"}
    assert not any("secret" in p.text or "thesis" in p.text for p in load_passages(tmp_path))


def test_the_real_corpus_excludes_private_files_and_pdfs():
    paths = {p.path for p in load_passages(ROOT)}
    assert not any(
        x.startswith((".project-local", "prior_work", "graphify-out", "data/")) for x in paths
    )
    assert not any(x.endswith(".pdf") for x in paths)
    assert "README.md" in paths and "literature/lit_matrix.csv" in paths


def test_the_right_passage_is_found_and_the_ranking_is_deterministic(tmp_path):
    root = _repo(tmp_path)
    a = search("Holm correction tests", k=2, root=root)
    b = search("Holm correction tests", k=2, root=root)
    assert a[0].passage.path == "docs/adr/0001.md" and a == b
    assert 0 < a[0].score <= 1


def test_a_question_without_an_answer_is_refused(tmp_path):
    assert search("recipe for chocolate cake", root=_repo(tmp_path)) == []


def test_the_gold_questions_exist_and_point_to_real_files():
    gold = yaml.safe_load((ROOT / "docs" / "assistant_gold.yaml").read_text())["questions"]
    assert len(gold) == 20 and {g["split"] for g in gold} == {"dev", "test"}
    assert sum(g["split"] == "dev" for g in gold) == 10
    for g in gold:
        assert all((ROOT / f).is_file() for f in g["expected"]), g["id"]
