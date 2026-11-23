# V4. Review and freeze the protocol

**File you edit:** `configs/protocol.yaml`  **Tests:** `tests/test_protocol_draft.py` (already real tests; they pass today)

## Why this matters

The protocol is the contract with the reviewer: what is forecast, over which dates, scored by which loss, tested how. It is frozen *before* the final results exist so no choice can be tuned to the answer. Week 6 is the freeze week, and the draft was rewritten on 2026-11-23 for the volatility pivot. This card is you reading it critically and deciding.

## Read first

- `ROADMAP.md` sections 1, 7 and 8 (the revised research questions, benchmark ladder and protocol).
- The `volatility`, `risk` and `design` blocks of `configs/protocol.yaml`.

## What to do

For each item, decide, write one line why, and put it in ADR 0002 (or a short note in the progress log):

1. **Horizons** 5, 22 and 66 trading days: do they match how a user of the forecast would act (weekly, monthly, quarterly)?
2. **Gap = horizon.** Explain in one sentence what leakage this prevents. Would a larger gap change results?
3. **OVX test start 2012-01-01** (OVX begins 2007-05). Is 4.6 years of training enough for the first refit, and how will you check it?
4. **Clean holdout.** The draft keeps the last 12 months untouched. Is that long enough to contain a volatile and a calm period? What would make you break it open?
5. **Primary loss = QLIKE** and the stretch criterion (inside the 90% MCS, plus beating HAR + OVX at p < 0.05 at 1+ horizons). Are these ambitious but honest? Edit the numbers if you disagree, and say why.
6. **Brent primary, WTI robustness.** OVX is built from WTI options. Does this hurt the Brent result, and where will you show it?

## Tests

`pytest tests/test_protocol_draft.py -q` must keep passing after your edits (it checks horizon order, the anchor being a listed benchmark, the OVX start date, the gap rule and that QLIKE is first). If you change a rule on purpose, change the test in the same commit and explain why.

## Done when

- You edited the draft where you disagreed and answered the six items.
- You committed the file and created the tag: `git tag protocol-frozen` (only when you are satisfied).
- Self-check: (1) why freeze before the results? (2) what is the cost of freezing too early? (3) which single protocol choice would a hostile reviewer attack first?
