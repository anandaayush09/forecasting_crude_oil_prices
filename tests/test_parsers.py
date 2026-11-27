import pandas as pd
import pytest

from oilfc.data.eia import clean_eia_frame
from oilfc.data.fred import parse_fred_csv


def test_parse_fred_csv_handles_missing_marker():
    text = "observation_date,VIXCLS\n2020-01-01,.\n2020-01-02,12.5\n2020-01-03,13.1\n"
    s = parse_fred_csv(text, "VIXCLS")
    assert s.name == "VIXCLS" and s.index.name == "date"
    assert s.isna().tolist() == [True, False, False]
    assert s.iloc[1] == 12.5


def test_clean_eia_frame_skips_header_rows():
    raw = pd.DataFrame(
        [
            ["Back to Contents", "Data 1: Europe Brent Spot Price"],
            ["Sourcekey", "RBRTE"],
            ["Date", "Europe Brent Spot Price FOB"],
            ["1987-05-21", 18.45],
            ["1987-05-20", 18.63],
        ]
    )
    s = clean_eia_frame(raw, "RBRTE")
    assert s.name == "RBRTE" and s.index.is_monotonic_increasing
    assert s.loc["1987-05-20"] == 18.63 and len(s) == 2


def test_parse_gpr_daily_uses_the_integer_day_and_drops_footer_rows():
    from oilfc.data.uncertainty import parse_gpr_daily

    df = pd.DataFrame({"DAY": [19850102, 19850101, None], "GPRD": [115.7, 230.0, None]})
    s = parse_gpr_daily(df)
    assert s.name == "gpr_daily" and s.index.name == "date"
    assert list(s.index) == [pd.Timestamp("1985-01-01"), pd.Timestamp("1985-01-02")]
    assert s.iloc[0] == 230.0


def test_parse_epu_daily_builds_dates_from_three_columns():
    from oilfc.data.uncertainty import parse_epu_daily

    text = "day,month,year,daily_policy_index\n2,1,1985,296.43\n1,1,1985,103.83\n"
    s = parse_epu_daily(text)
    assert s.name == "epu_daily"
    assert s.index[0] == pd.Timestamp("1985-01-01") and s.iloc[0] == 103.83


@pytest.mark.parametrize(
    "key, freq, section, expected",
    [
        ("WCESTUS1", "w", None, "https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls"),
        ("RNGWHHD", "d", "ng", "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"),
    ],
)
def test_eia_url_carries_the_frequency_and_the_catalogue_section(
    monkeypatch, key, freq, section, expected
):
    import oilfc.data.eia as eia

    seen = []

    def fake_get(url, timeout):
        seen.append(url)
        raise RuntimeError("stop before any download")

    monkeypatch.setattr(eia.requests, "get", fake_get)
    with pytest.raises(RuntimeError):
        eia.fetch_eia(key, freq) if section is None else eia.fetch_eia(key, freq, section=section)
    assert seen == [expected]
