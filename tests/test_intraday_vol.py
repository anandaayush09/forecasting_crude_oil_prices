"""Realised-volatility inputs from the VOLARE archive (synthetic data only)."""

import zipfile

import numpy as np
import pandas as pd
import pytest

from oilfc.features.intraday_vol import intraday_vol_design, load_volare


def _volare(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    rv5 = np.exp(rng.normal(np.log(3e-4), 0.5, n))
    return pd.DataFrame({"rv5": rv5, "rv5_ss": rv5 * 1.02}, index=idx)


def _csv(rows):
    head = "date,symbol,asset_type,rv5,rv5_ss"
    return chr(10).join([head, *rows]) + chr(10)


def test_the_loader_reads_a_zip_keeps_one_symbol_and_drops_bad_rows(tmp_path):
    body = _csv(
        [
            "2020-01-01,CL,futures,0.0003,0.0004",
            "2020-01-02,CL,futures,0.0,0.0004",  # zero variance
            "2020-01-03,CL,futures,,0.0004",  # missing
            "2020-01-06,CL,futures,0.0005,0.0006",
            "2020-01-06,CL,futures,0.9,0.9",  # duplicate date, first one kept
            "2020-01-01,GC,futures,0.0009,0.0009",  # other symbol
        ]
    )
    z = tmp_path / "v.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("data.csv", body)
        f.writestr("README.txt", "x")
    d = load_volare(z, "CL")
    assert list(d.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-06")]
    assert d["rv5"].tolist() == [0.0003, 0.0005]
    csv = tmp_path / "v.csv"
    csv.write_text(body)
    pd.testing.assert_frame_equal(load_volare(csv, "CL"), d)


def test_the_design_uses_days_up_to_the_previous_day_only():
    v = _volare()
    base = intraday_vol_design(v, v.index)
    changed = v.copy()
    changed.iloc[100:, :] *= 50.0  # rows from position 100 on are corrupted
    new = intraday_vol_design(changed, v.index)
    pd.testing.assert_frame_equal(base.iloc[:101], new.iloc[:101])  # position 100 sees through 99
    assert not base.iloc[101:].equals(new.iloc[101:])


def test_the_design_is_the_annualised_log_volatility_of_the_window():
    v = _volare()
    d = intraday_vol_design(v, v.index)
    expected = 0.5 * np.log(252 * v["rv5"].iloc[40:45].mean())  # window of 5 ending at position 44
    assert d["log_irv_5"].iloc[45] == pytest.approx(expected)
    assert set(d.columns) == {f"log_{t}_{w}" for t in ("irv", "irvss") for w in (5, 22, 63)}


def test_a_window_needs_enough_days_and_missing_days_are_averaged_over():
    v = _volare()
    d = intraday_vol_design(v.drop(v.index[60:62]), v.index)  # 22-day window still 90% present
    assert d["log_irv_22"].iloc[70:].notna().all()
    gone = intraday_vol_design(v.drop(v.index[50:70]), v.index)
    assert gone["log_irv_22"].iloc[70:80].isna().any()
