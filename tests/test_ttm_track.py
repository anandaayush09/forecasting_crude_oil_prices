"""Foundation-model live track (ADR 0010): input series, records and gating, with a fake predictor."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from oilfc.features.ttm_input import ttm_input
from oilfc.live import ttm
from oilfc.live.records import verify_forecast, write_forecast
from oilfc.live.ttm import NotScoredOrigin, ttm_forecasts

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _panel(n=900, start="2024-01-01", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    r = rng.standard_t(6, n) * 0.015
    return pd.DataFrame({"brent_spot": 70 * np.exp(np.cumsum(r))}, index=idx)


class Fake:
    """Predicts a path whose value at step k is base + 0.01 * k (log volatility)."""

    def __init__(self, base=-1.5):
        self.base, self.seen = base, []

    def predict(self, context):
        self.seen.append(np.array(context))
        return self.base + 0.01 * np.arange(ttm.PREDICTION_LENGTH)

    def versions(self):
        return {"fake": "0"}


def _head(panel, hl=0.4):
    origin = str(panel.index[-1].date())
    content = {
        "asset": "brent",
        "origin_date": origin,
        "horizons": {str(h): {ttm.HEADLINE: hl} for h in ttm.HORIZONS},
    }
    return {"content": content, "sha256": "abc"}


def test_the_input_is_the_annualised_log_volatility_of_the_last_h_returns():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.02, 60))
    x = ttm_input(r, 5)
    assert x.iloc[:4].isna().all()
    assert x.iloc[10] == pytest.approx(0.5 * np.log(252 * np.mean(r.iloc[6:11] ** 2)))


def test_the_forecast_is_the_model_output_at_step_h_and_the_mean_is_in_log_volatility():
    p = _panel()
    fake = Fake()
    rec = ttm_forecasts(p, "brent", _head(p, 0.4), fake, code_version="t")
    for h in ttm.HORIZONS:
        row = rec["horizons"][str(h)]
        log_ttm = fake.base + 0.01 * (h - 1)
        assert row["ttm"] == pytest.approx(np.exp(log_ttm), rel=1e-7)
        assert row["ttm_headline_mean"] == pytest.approx(np.sqrt(np.exp(log_ttm) * 0.4), rel=1e-7)
        assert row["headline"] == 0.4
    assert all(len(c) == ttm.CONTEXT for c in fake.seen)
    assert rec["model"]["revision"] == ttm.REVISION and rec["model"]["mode"] == "zero_shot"
    assert set(rec["horizons"]["5"]) == {*ttm.MODELS, "headline"}


def test_the_context_is_the_last_512_values_ending_at_the_origin():
    p = _panel()
    fake = Fake()
    ttm_forecasts(p, "brent", _head(p), fake, horizons=(22,))
    r = ttm.returns_of(p, "brent")
    assert np.allclose(fake.seen[0], ttm_input(r, 22).iloc[-512:].to_numpy())


def test_nothing_after_the_origin_is_used():
    p = _panel()
    origin = p.index[-40]
    corrupted = p.copy()
    corrupted.loc[corrupted.index > origin, "brent_spot"] = 999.0
    a = ttm_forecasts(corrupted, "brent", _head(p.loc[:origin]), Fake(), origin=origin)
    b = ttm_forecasts(p.loc[:origin], "brent", _head(p.loc[:origin]), Fake())
    assert a == b


def test_an_origin_on_or_before_the_tag_date_is_not_part_of_the_claim():
    p = _panel(start="2023-06-01", n=700)
    assert p.index[-1] <= ttm.FIRST_ORIGIN_AFTER
    with pytest.raises(NotScoredOrigin, match="not after"):
        ttm_forecasts(p, "brent", _head(p), Fake())


def test_a_non_finite_input_or_a_short_history_writes_no_forecast():
    p = _panel()
    flat = p.copy()
    flat.iloc[-30:, 0] = flat.iloc[-30, 0]  # 30 zero returns: the 5-day window is -inf
    with pytest.raises(NotScoredOrigin, match="finite"):
        ttm_forecasts(flat, "brent", _head(flat), Fake())
    short = _panel(n=400, start="2026-01-01")
    with pytest.raises(NotScoredOrigin, match="512"):
        ttm_forecasts(short, "brent", _head(short), Fake())


def test_the_headline_record_must_match_the_asset_and_origin():
    p = _panel()
    wrong = _head(p)
    wrong["content"]["origin_date"] = "2020-01-01"
    with pytest.raises(ValueError, match="another asset or origin"):
        ttm_forecasts(p, "brent", wrong, Fake())


def test_a_record_is_written_once_and_verifies(tmp_path):
    p = _panel()
    rec = ttm_forecasts(p, "brent", _head(p), Fake(), code_version="t")
    path = write_forecast(rec, tmp_path, run_utc="2027-01-01T00:00:00Z")
    assert verify_forecast(path)
    with pytest.raises(FileExistsError):
        write_forecast(rec, tmp_path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["content"]["horizons"]["5"]["ttm"] = 9.9
    path.write_text(json.dumps(stored), encoding="utf-8")
    assert not verify_forecast(path)


def test_the_code_constants_match_protocol_v2():
    f = yaml.safe_load((ROOT / "configs" / "protocol_v2.yaml").read_text())["foundation_track"]
    assert f["model"]["revision"] == ttm.REVISION and f["model"]["branch"] == ttm.BRANCH
    assert f["model"]["repo"] == ttm.REPO and f["input"]["context"] == ttm.CONTEXT
    assert tuple(f["horizons"]) == ttm.HORIZONS and tuple(f["models"]) == ttm.MODELS


def _scored(n, gain=0.2, seed=0):
    rng = np.random.default_rng(seed)
    dates = [str(d.date()) for d in pd.bdate_range("2027-01-04", periods=n)]
    rows = []
    for asset in ("brent", "wti"):
        for hh in (5, 22):
            base = 0.4 + rng.normal(0, 0.05, n)
            mean = base - gain + rng.normal(0, 0.02, n)
            for i, d in enumerate(dates):
                rows += [
                    (asset, hh, d, "headline", base[i]),
                    (asset, hh, d, "ttm", base[i] + 0.05),
                    (asset, hh, d, "ttm_headline_mean", mean[i]),
                ]
    return pd.DataFrame(rows, columns=["asset", "horizon", "origin_date", "model", "qlike"])


def _live_score():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "live_score_ttm", ROOT / "experiments" / "live_score_ttm.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_claim_is_not_decided_before_60_origins_in_every_cell():
    m = _live_score()
    assert m.verdict(m.claim_table(_scored(30))).startswith("Not decided")


def test_the_claim_holds_only_with_a_gain_and_a_holm_p_below_005_in_all_four_cells():
    m = _live_score()
    assert m.verdict(m.claim_table(_scored(120, gain=0.2))).startswith("Claim holds")
    worse = _scored(120, gain=-0.2)  # the mean is worse than the headline everywhere
    assert m.verdict(m.claim_table(worse)).startswith("Claim does not hold")
