"""The final holdout evaluation: freeze guards, single-write output, and same procedure as development."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oilfc.eval.holdout_final import (
    NotFrozenError,
    check_frozen,
    evaluate,
    holdout_forecasts,
    src_changes_since_freeze,
    write_once,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS_NO_GARCH = ("naive", "har_bc", "lhar_bc", "lhar_ovx_bc_lag1")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "protocol.yaml").write_text("version: 1\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_repo_without_the_freeze_tag_is_refused(repo):
    with pytest.raises(NotFrozenError, match="does not exist"):
        check_frozen(repo)


def test_a_frozen_clean_repo_passes_and_returns_head(repo):
    _git(repo, "tag", "-a", "protocol-frozen", "-m", "frozen")
    head = check_frozen(repo)
    assert len(head) == 40


def test_a_protocol_changed_after_the_tag_is_refused(repo):
    _git(repo, "tag", "-a", "protocol-frozen", "-m", "frozen")
    (repo / "configs" / "protocol.yaml").write_text("version: 2\n")
    _git(repo, "commit", "-aq", "-m", "change protocol")
    with pytest.raises(NotFrozenError, match="differs"):
        check_frozen(repo)


def test_uncommitted_edits_to_tracked_files_are_refused(repo):
    _git(repo, "tag", "-a", "protocol-frozen", "-m", "frozen")
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(NotFrozenError, match="uncommitted"):
        check_frozen(repo)


def test_a_tag_that_is_not_an_ancestor_of_head_is_refused(repo):
    _git(repo, "checkout", "-q", "-b", "other")
    (repo / "code.py").write_text("x = 3\n")
    _git(repo, "commit", "-aq", "-m", "other work")
    _git(repo, "tag", "-a", "protocol-frozen", "-m", "frozen")
    _git(repo, "checkout", "-q", "-")
    with pytest.raises(NotFrozenError, match="ancestor"):
        check_frozen(repo)


def test_the_result_file_is_written_once_and_never_overwritten(tmp_path):
    path = tmp_path / "out" / "holdout_v1.json"
    digest = write_once(path, {"a": 1})
    assert len(digest) == 64 and path.exists()
    with pytest.raises(FileExistsError):
        write_once(path, {"a": 2})
    assert '"a": 1' in path.read_text()


def _panel(n=2000, start="2008-01-01", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    vol = np.exp(np.cumsum(rng.normal(0, 0.03, n)) * 0.3 - 3.7)
    r = rng.standard_t(6, n) * vol
    price = 60 * np.exp(np.cumsum(r))
    ovx = 100 * vol * np.sqrt(252) * np.exp(rng.normal(0.1, 0.1, n))
    return pd.DataFrame({"brent_spot": price, "ovx": ovx}, index=idx)


def test_unknown_model_names_are_rejected():
    p = _panel(n=400)
    r = np.log(p["brent_spot"]).diff().dropna()
    with pytest.raises(ValueError, match="unknown"):
        holdout_forecasts(r, p["ovx"], 5, "2009-06-01", ("naive", "magic"))


@pytest.fixture(scope="module")
def development_run():
    """One development-procedure run on synthetic data, shared by the equivalence tests."""
    spec = importlib.util.spec_from_file_location(
        "vol_ovx_timing", ROOT / "experiments" / "vol_ovx_timing.py"
    )
    dev_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev_module)
    p = _panel()
    return p, dev_module.forecasts("brent", 5, p)


def test_the_holdout_procedure_equals_the_development_procedure(development_run):
    """Same data and start date: the holdout function reproduces vol_ovx_timing.forecasts exactly."""
    p, dev = development_run
    r = np.log(p["brent_spot"]).diff().dropna()
    new = holdout_forecasts(r, p["ovx"], 5, "2012-01-01")
    cols = ["y", "naive", "har_bc", "lhar_bc", "lhar_ovx_bc_lag1", "garch_t", "gjr_t"]
    common = dev.index.intersection(new.index)
    assert len(common) > 800
    pd.testing.assert_frame_equal(
        dev.loc[common, cols], new.loc[common, cols], check_exact=False, rtol=1e-9
    )


def test_the_refit_check_with_monthly_blocks_equals_the_development_procedure(development_run):
    """vol_refit.py keeps its own copy of the log-linear ladder: it must not drift from the headline."""
    p, dev = development_run
    spec = importlib.util.spec_from_file_location(
        "vol_refit", ROOT / "experiments" / "vol_refit.py"
    )
    refit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refit)
    r = np.log(p["brent_spot"]).diff().dropna()
    new = refit.forecasts(r, p["ovx"].reindex(r.index).shift(1), 5, 1, None)
    cols = ["y", "naive", "har_bc", "lhar_bc", "lhar_ovx_bc_lag1"]
    common = dev.index.intersection(new.index)
    assert len(common) > 800
    pd.testing.assert_frame_equal(
        dev.loc[common, cols], new.loc[common, cols], check_exact=False, rtol=1e-9
    )


def test_a_forecast_at_an_origin_does_not_depend_on_later_data():
    p = _panel(n=1800)
    r = np.log(p["brent_spot"]).diff().dropna()
    full = holdout_forecasts(r, p["ovx"], 5, "2013-01-01", MODELS_NO_GARCH)
    cut = r.index[-120]
    short = holdout_forecasts(r.loc[:cut], p["ovx"], 5, "2013-01-01", MODELS_NO_GARCH)
    # origins whose month block is complete in the shorter sample and whose targets exist in both
    last_full_block = short.index[short.index < short.index[-1] - pd.DateOffset(months=1)]
    pd.testing.assert_frame_equal(
        full.loc[last_full_block], short.loc[last_full_block], check_exact=False, rtol=1e-9
    )


def test_evaluate_reports_a_gain_and_a_one_sided_p_for_a_clearly_better_headline():
    rng = np.random.default_rng(0)
    y = np.exp(rng.normal(-1.5, 0.3, 500))
    df = pd.DataFrame({"y": y, "head": y * np.exp(rng.normal(0, 0.05, 500)),
                       "bad": y * np.exp(rng.normal(0.4, 0.3, 500))})  # fmt: skip
    row = evaluate(df, 1, "head", ["bad"])[0]
    assert row["gain_pct"] > 50 and row["dm_p"] < 0.001 and row["n"] == 500


def test_the_dry_run_checks_the_freeze_and_evaluates_nothing(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location(
        "holdout_final_script", ROOT / "experiments" / "holdout_final.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    panel_path = tmp_path / "panel.parquet"
    p = _panel(n=300, start="2025-01-01")
    p["wti_spot"] = p["brent_spot"]
    p.to_parquet(panel_path)

    def boom(*a, **k):
        raise AssertionError("the dry run must not compute forecasts")

    monkeypatch.setattr(script, "check_frozen", lambda root: "a" * 40)
    monkeypatch.setattr(
        script, "src_changes_since_freeze", lambda root: []
    )  # CI clones have no tags
    monkeypatch.setattr(script, "holdout_forecasts", boom)
    monkeypatch.setattr(script, "PANEL", panel_path)
    monkeypatch.setattr(script, "OUT", tmp_path / "holdout_v1.json")
    monkeypatch.setattr(sys, "argv", ["holdout_final.py"])
    script.main()
    out = capsys.readouterr().out
    assert "dry run" in out and "origins with a realised target" in out
    assert not (tmp_path / "holdout_v1.json").exists()


def test_a_second_run_is_refused_when_the_result_file_exists(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "holdout_final_script2", ROOT / "experiments" / "holdout_final.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    done = tmp_path / "holdout_v1.json"
    done.write_text("{}")
    monkeypatch.setattr(script, "check_frozen", lambda root: "a" * 40)
    monkeypatch.setattr(
        script, "src_changes_since_freeze", lambda root: []
    )  # CI clones have no tags
    monkeypatch.setattr(script, "OUT", done)
    monkeypatch.setattr(sys, "argv", ["holdout_final.py", "--confirm-single-run"])
    with pytest.raises(SystemExit, match="already evaluated"):
        script.main()


def test_source_files_changed_after_the_freeze_tag_are_listed(repo):
    (repo / "src").mkdir()
    (repo / "src" / "model.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add model")
    _git(repo, "tag", "-a", "protocol-frozen", "-m", "frozen")
    assert src_changes_since_freeze(repo) == []
    (repo / "src" / "model.py").write_text("x = 2\n")
    (repo / "src" / "new.py").write_text("y = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "change model after the freeze")
    assert src_changes_since_freeze(repo) == ["src/model.py", "src/new.py"]
