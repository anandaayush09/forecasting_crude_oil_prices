"""V4: sanity checks on the protocol (v1, draft for freeze). Real tests.
If you change configs/protocol.yaml, keep these true or change them on purpose in the same commit."""

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
    assert {"naive", "har_bc", "lhar_bc", "garch_t", "gjr_t"} <= set(v["benchmarks"])


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


def test_the_headline_is_strict_in_time_and_the_beaten_comparators_are_declared_benchmarks():
    v = CFG["volatility"]
    assert v["headline_model"] in v["benchmarks"] and v["headline_model"].endswith("lag1")
    assert v["ovx_timing"] == "previous_trading_day"
    assert set(v["claim_comparators"]) <= set(v["benchmarks"])
    assert v["headline_model"] not in v["claim_comparators"]


def test_the_development_window_ends_before_the_holdout_and_the_live_test_covers_the_claim():
    d = CFG["design"]
    assert pd.Timestamp(d["test_end"]) < pd.Timestamp(CFG["clean_holdout_start"])
    assert CFG["holdout_rule"] == "single_evaluation"
    live = CFG["live_test"]
    assert CFG["volatility"]["headline_model"] in live["models"]
    assert set(CFG["volatility"]["claim_comparators"]) <= set(live["models"])
    assert set(live["horizons"]) == set(CFG["volatility"]["horizons"])


def test_the_live_test_models_match_the_code_that_writes_the_forecasts():
    from oilfc.live.forecast import HORIZONS, MODELS

    assert set(CFG["live_test"]["models"]) == set(MODELS)
    assert tuple(CFG["live_test"]["horizons"]) == HORIZONS
