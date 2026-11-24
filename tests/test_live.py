import json

import numpy as np
import pandas as pd
import pytest

from oilfc.eval.losses import qlike
from oilfc.live.forecast import MODELS, live_forecasts, verify_forecast, write_forecast
from oilfc.live.score import realised_vol_after, score_forecasts


def _panel(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-05", periods=n)
    vol = np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.3 - 3.7)
    r = rng.standard_t(6, n) * vol
    price = 60 * np.exp(np.cumsum(r))
    price2 = 58 * np.exp(np.cumsum(r + rng.normal(0, 0.002, n)))
    ovx = 100 * vol * np.sqrt(252) * np.exp(rng.normal(0.1, 0.1, n))
    return pd.DataFrame({"brent_spot": price, "wti_spot": price2, "ovx": ovx}, index=idx)


def _fc(panel, **kw):
    return live_forecasts(panel, "brent", horizons=(5,), code_version="test", **kw)


def test_every_model_gives_a_positive_finite_volatility():
    rec = _fc(_panel())
    assert set(rec["horizons"]["5"]) == set(MODELS)
    for v in rec["horizons"]["5"].values():
        assert np.isfinite(v) and v > 0
    assert rec["origin_date"] == str(_panel().index[-1].date())


def test_nothing_after_the_origin_is_used():
    p = _panel()
    origin = p.index[-40]
    corrupted = p.copy()
    corrupted.loc[corrupted.index > origin, ["brent_spot", "ovx"]] = 999.0
    assert _fc(corrupted, origin=origin) == _fc(p.loc[:origin])


def test_strict_ovx_timing_uses_yesterdays_value_not_todays():
    p = _panel()
    base = _fc(p)["horizons"]["5"]["lhar_ovx_bc_lag1"]
    today = p.copy()
    today.iloc[-1, today.columns.get_loc("ovx")] *= 3
    assert _fc(today)["horizons"]["5"]["lhar_ovx_bc_lag1"] == pytest.approx(base)
    yesterday = p.copy()
    yesterday.iloc[-2, yesterday.columns.get_loc("ovx")] *= 3
    assert _fc(yesterday)["horizons"]["5"]["lhar_ovx_bc_lag1"] != pytest.approx(base)


def test_written_forecasts_are_hashed_never_overwritten_and_tamper_evident(tmp_path):
    rec = _fc(_panel())
    path = write_forecast(rec, tmp_path, run_utc="2026-11-24T10:00:00Z")
    assert verify_forecast(path)
    manifest = pd.read_csv(tmp_path / "MANIFEST.csv")
    stored = json.loads(path.read_text())
    assert manifest.loc[0, "sha256"] == stored["sha256"]
    assert manifest.loc[0, "origin_date"] == rec["origin_date"]
    with pytest.raises(FileExistsError):
        write_forecast(rec, tmp_path, run_utc="2026-11-25T10:00:00Z")
    stored["content"]["horizons"]["5"]["naive"] += 0.01
    path.write_text(json.dumps(stored))
    assert not verify_forecast(path)
    assert len(pd.read_csv(tmp_path / "MANIFEST.csv")) == 1


def test_realised_volatility_after_an_origin():
    idx = pd.bdate_range("2020-01-01", periods=10)
    r = pd.Series([0.0, 0.01, -0.02, 0.03, 0.01, -0.01, 0.02, 0.0, 0.01, 0.02], index=idx)
    got = realised_vol_after(r, idx[2], 3)
    assert got == pytest.approx(np.sqrt(252 * np.mean(r.iloc[3:6] ** 2)))
    assert np.isnan(realised_vol_after(r, idx[8], 3))  # only one later return


def test_scoring_matches_a_manual_qlike_and_leaves_open_forecasts_unscored(tmp_path):
    p = _panel()
    old = p.index[-100]
    write_forecast(_fc(p.loc[:old]), tmp_path, run_utc="2026-01-01T00:00:00Z")
    write_forecast(_fc(p), tmp_path, run_utc="2026-02-01T00:00:00Z")
    table = score_forecasts(tmp_path, p)
    done = table[table["origin_date"] == str(old.date())]
    open_ = table[table["origin_date"] == str(p.index[-1].date())]
    assert done["realised"].notna().all() and open_["realised"].isna().all()
    r = np.log(p["brent_spot"]).diff().dropna()
    realised = realised_vol_after(r, old, 5)
    row = done[done["model"] == "naive"].iloc[0]
    assert row["realised"] == pytest.approx(realised)
    assert row["qlike"] == pytest.approx(
        float(np.asarray(qlike(pd.Series([realised]), pd.Series([row["forecast"]])))[0])
    )
