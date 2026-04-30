import pandas as pd

from src.ingest.macro.nse_fred import _safe_key
from src.ingest.macro.nse_fred import build_macro_feature_daily


def test_safe_key_normalizes_index_names() -> None:
    assert _safe_key("NIFTY OIL & GAS") == "nifty_oil_and_gas"
    assert _safe_key("NIFTY 10 YR BENCHMARK G-SEC") == "nifty_10_yr_benchmark_g_sec"


def test_build_macro_feature_daily_adds_returns_and_flags() -> None:
    levels = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "series_key": ["india_vix", "india_vix", "india_vix"],
            "series_name": ["INDIA VIX", "INDIA VIX", "INDIA VIX"],
            "close": [14.0, 16.0, 18.0],
            "open": [13.0, 15.0, 17.0],
            "high": [15.0, 17.0, 19.0],
            "low": [12.0, 14.0, 16.0],
            "source_family": ["nse_vix_history"] * 3,
            "source_url": ["u"] * 3,
            "source_note": ["n"] * 3,
        }
    )
    valuations = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "series_key": ["nifty_50", "nifty_50", "nifty_50"],
            "series_name": ["NIFTY 50", "NIFTY 50", "NIFTY 50"],
            "pe": [22.0, 23.0, 24.0],
            "pb": [3.1, 3.2, 3.3],
            "dy": [1.1, 1.1, 1.0],
            "source_url": ["u"] * 3,
            "source_note": ["n"] * 3,
        }
    )

    result = build_macro_feature_daily(levels, valuations)
    assert "india_vix_level" in result.columns
    assert "india_vix_return_1d" in result.columns
    assert "macro_vix_below_20" in result.columns
    assert "nifty_50_pe" in result.columns
    assert bool(result.loc[result["trade_date"] == pd.Timestamp("2024-01-03"), "macro_vix_below_20"].iloc[0])
