"""V4: sanity checks on the revised protocol draft. These are real tests (they pass today).
If you change configs/protocol.yaml during the freeze review, keep these true or change them on purpose."""

from pathlib import Path

import pandas as pd
import yaml

CFG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "configs" / "protocol.yaml").read_text()
)


def test_volatility_horizons_are_positive_and_strictly_increasing():
    h = CFG["volatility"]["horizons"]
    assert all(x > 0 for x in h) and h == sorted(set(h))


def test_the_anchor_is_one_of_the_benchmarks_and_the_benchmark_ladder_is_complete():
    v = CFG["volatility"]
    assert v["anchor"] in v["benchmarks"]
    assert {"naive_rv", "garch_t", "har", "ovx", "har_ovx"} <= set(v["benchmarks"])


def test_ovx_tests_start_after_ovx_exists_with_enough_history():
    # OVX starts 2007-05; require at least 4 years of training before the first OVX test origin
    assert pd.Timestamp(CFG["design"]["ovx_test_start"]) >= pd.Timestamp("2011-05-01")


def test_a_gap_equal_to_the_horizon_is_required_and_the_holdout_is_last():
    assert CFG["design"]["gap"] == "horizon"
    assert pd.Timestamp(CFG["clean_holdout_start"]) > pd.Timestamp(CFG["design"]["ovx_test_start"])


def test_primary_loss_and_risk_backtests_are_declared():
    assert CFG["volatility"]["losses"][0] == "qlike"
    assert {"kupiec", "christoffersen"} <= set(CFG["risk"]["backtests"])
    assert all(0 < q < 0.5 for q in CFG["risk"]["var_levels"])
