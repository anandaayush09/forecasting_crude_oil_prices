"""Parsing of the Yahoo chart response (no network)."""

import pandas as pd

from oilfc.data.yahoo import parse_chart


def test_a_chart_payload_becomes_a_dated_ohlc_frame_with_missing_values_kept():
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": [1577881800, 1577968200],  # 2020-01-01 and 2020-01-02, 13:30 UTC
                    "indicators": {
                        "quote": [
                            {
                                "open": [61.0, None],
                                "high": [62.0, 63.0],
                                "low": [60.0, 61.5],
                                "close": [61.5, 62.5],
                            }
                        ]
                    },
                }
            ]
        }
    }
    frame = parse_chart(payload)
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert list(frame.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert pd.isna(frame.loc["2020-01-02", "open"]) and frame.loc["2020-01-01", "high"] == 62.0
