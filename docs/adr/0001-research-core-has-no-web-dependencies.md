# 0001. The research core has no web dependencies

- Status: **draft, to be written by you**
- Date: (fill in)

> I gave you the title and the questions. The content is yours. Interviewers ask "why this over the alternative?"; this is your rehearsal.
> Use the template in `0000-template.md`. Delete these prompt lines when you are done.

## Context (prompts)

- The project has a research part (data, models, evaluation) and a product part (API, UI, worker). What goes wrong if `src/oilfc/eval/` imports FastAPI?
- Who else needs to run the research code? (A reviewer reproducing the paper, a Colab notebook, the daily live job.)

## Options considered (prompts)

- One package where everything imports everything.
- Two layers: research core, and a thin web layer that only reads results.
- Two separate repositories.

## Decision (prompt)

Which option, and what rule enforces it? (Hint: can a test or a lint rule check "no module under `oilfc/` imports `oilfc_api`"?)

## Consequences (prompts)

- What becomes harder? (For example, the API cannot call a model directly; it has to read stored results.)
- What would make you revisit this?
