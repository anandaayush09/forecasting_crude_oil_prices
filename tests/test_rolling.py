"""V5: expanding-origin splits with a gap. See docs/tasks/V5_rolling_splits.md.

Marked xfail until you implement src/oilfc/eval/rolling.py. When all tests pass, delete the pytestmark line.

expanding_splits(index, test_start, h, refit_months) -> list[tuple[np.ndarray, np.ndarray]]
  index:         DatetimeIndex of forecast origins (one per trading day)
  test_start:    first date a test origin may have
  h:             forecast horizon in trading days (rows)
  refit_months:  length of each test block in calendar months
Each tuple is (train_positions, test_positions): integer positions into `index`.
  - train is every origin i with i + h < first test position (the target of origin i is fully
    known before the test block starts), starting at position 0 (expanding window)
  - test block k holds the origins with test_start + k*m months <= date < test_start + (k+1)*m months
"""

import numpy as np
import pandas as pd

INDEX = pd.bdate_range("2000-01-03", "2020-12-31")


def _splits(h=22, months=12, start="2010-01-01"):
    from oilfc.eval.rolling import expanding_splits

    return expanding_splits(INDEX, pd.Timestamp(start), h, months)


def test_no_training_target_overlaps_the_test_block():
    for h in (1, 5, 22, 66):
        for train, test in _splits(h=h):
            assert train[-1] + h < test[0]


def test_training_window_is_an_expanding_prefix():
    splits = _splits()
    sizes = [len(train) for train, _ in splits]
    assert sizes == sorted(set(sizes))  # strictly increasing
    for train, _ in splits:
        assert train[0] == 0
        assert (np.diff(train) == 1).all()


def test_test_blocks_cover_every_origin_from_test_start_once():
    splits = _splits()
    all_test = np.concatenate([test for _, test in splits])
    expected = np.flatnonzero(INDEX >= pd.Timestamp("2010-01-01"))
    assert (all_test == expected).all()  # ordered, disjoint, complete


def test_each_block_spans_the_requested_calendar_months():
    start = pd.Timestamp("2010-01-01")
    for k, (_, test) in enumerate(_splits(months=6)):
        lo = start + pd.DateOffset(months=6 * k)
        hi = start + pd.DateOffset(months=6 * (k + 1))
        dates = INDEX[test]
        assert (dates >= lo).all() and (dates < hi).all()


def test_a_longer_horizon_shortens_the_training_window_by_the_extra_gap():
    short = _splits(h=1)[0][0]
    long_ = _splits(h=22)[0][0]
    assert len(short) - len(long_) == 21


def test_first_test_origin_is_the_first_date_on_or_after_test_start():
    train, test = _splits(start="2010-01-02")[0]  # a Saturday
    assert INDEX[test[0]] == pd.Timestamp("2010-01-04")


def test_no_splits_when_test_start_is_after_the_data():
    assert _splits(start="2030-01-01") == []
