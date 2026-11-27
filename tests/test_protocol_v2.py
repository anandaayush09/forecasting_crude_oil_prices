"""Protocol v2 (draft, ADR 0010): it extends v1 without changing it."""

import hashlib
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
V1 = yaml.safe_load((CONFIGS / "protocol.yaml").read_text())
V2 = yaml.safe_load((CONFIGS / "protocol_v2.yaml").read_text())


def test_v1_is_unchanged_since_v2_was_written():
    text = (CONFIGS / "protocol.yaml").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(text).hexdigest() == V2["extends"]["sha256_lf"]


def test_v2_extends_v1_and_names_what_it_leaves_alone():
    assert V1["version"] == 1 and V2["version"] == 2
    assert V2["extends"]["file"] == "protocol.yaml"
    assert set(V2["unchanged"]) <= set(V1)


def test_the_foundation_track_is_secondary_and_inside_the_v1_task():
    f = V2["foundation_track"]
    assert f["status"] == "secondary"
    assert set(f["horizons"]) <= set(V1["volatility"]["horizons"])
    assert set(f["assets"]) == {V1["targets"]["primary"], V1["targets"]["robustness"]}
    assert V1["volatility"]["headline_model"] in f["combination"]


def test_the_model_is_pinned_to_a_commit_and_runs_zero_shot():
    m = V2["foundation_track"]["model"]
    assert len(m["revision"]) == 40 and set(m["revision"]) <= set("0123456789abcdef")
    assert m["mode"] == "zero_shot" and m["licence"] == "Apache-2.0"
    assert V2["foundation_track"]["input"]["context"] == 512
    assert V2["foundation_track"]["backtest"] == "not_evidence"
