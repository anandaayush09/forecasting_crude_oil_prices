# 0011. Local research assistant (Part 2): scope, guards and evaluation, fixed before any retriever is built

- Status: accepted (scope set by the assistant at Aayush's request, 2026-11-28)
- Date: 2026-11-28

## Context

The project's record is spread over about 28,000 words in ten ADRs, a README, two roadmaps, task cards, a live-test note and a literature matrix. Finding where a number, a rule or a decision is written takes searching by hand. Part 2 is a small tool that answers such questions from those documents only, offline, with a citation to the file and heading. It is not an input to any forecast and does not touch the research code, the protocols or the live tracks.

## Scope (v1)

- **Corpus (public files only):** `README.md`, `ROADMAP.md`, `docs/**/*.md`, `live/README.md`, `literature/notes/*.md`, `literature/lit_matrix.csv` (one passage per paper), `configs/*.yaml` and `results/tables/SUMMARY.md`. Never indexed: `.project-local/`, `prior_work/`, `graphify-out/`, `data/`, the PDFs in `literature/`, and every other file under `results/`. The PDFs are other people's copyrighted papers, and `.project-local/` is private.
- **Method:** retrieval first. Passages are cut at markdown headings (at most 200 words each), indexed with TF-IDF from scikit-learn (sublinear term frequency, English stop words, unigrams and bigrams) and ranked by cosine similarity. No embedding model, no download, no network, no GPU, no cost. The result is deterministic.
- **Answer:** the top passages with their file, heading and score. There is no generated prose in v1: no local language model is installed, and an extractive answer cannot state something the documents do not say.
- **Refusal:** if the best score is below a threshold, the tool says the documents do not contain an answer and shows nothing.
- **Interface:** `python experiments/ask.py "question" [-k 3]`, and a library function `oilfc.assistant.search`.
- **Out of scope for v1:** an MCP server (the empty `mcp_server/` package stays as it is), embeddings, a language model for generation, PDF text, and any use of the assistant's output as a model input.

## Evaluation (fixed now, before the code exists)

`docs/assistant_gold.yaml` holds 20 questions, each with the files in which the answer is written (any of them counts as a hit). The questions are split into 10 development questions and 10 test questions. Settings may be chosen by looking at the development questions only (the chunk size, the n-gram range, the refusal threshold). The test questions are run once, after the settings are fixed, and the result is reported as it comes: recall at 1, 3 and 5, and mean reciprocal rank, in `results/tables/assistant_retrieval.csv`. Ten test questions is a small sample, so the numbers carry wide uncertainty and say whether the tool is usable, not how good it is.

## Guards

- Read-only: the tool never writes to the repository.
- A test checks that no private path can enter the index.
- A test checks that the same question always gives the same ranking.
- A test checks that a question with no answer in the documents is refused.

## Consequences

- The tool answers questions about what the documents say, which may be out of date. It cites the file so the reader can check.
- A null on retrieval quality is an acceptable outcome and is reported. Better retrieval (embeddings) and generation are possible follow-ups, each needing its own note.
- Departures from this note are recorded as dated amendments here.
