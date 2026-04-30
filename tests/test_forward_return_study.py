from __future__ import annotations

from datetime import date

import pandas as pd

from src.analysis.forward_return_study import add_bucket_columns
from src.analysis.forward_return_study import add_market_regime_features
from src.analysis.forward_return_study import build_forward_return_labels


def test_build_forward_return_labels_uses_next_available_trade_date() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-17"]),
            "close": [100.0, 110.0, 130.0],
        }
    )
    labeled = build_forward_return_labels(
        df,
        analysis_start_date=date(2024, 1, 1),
        analysis_end_date=date(2024, 1, 1),
        horizon_days=15,
        target_return=0.25,
        min_price=None,
    )
    row = labeled.iloc[0]
    assert row["forward_trade_date"] == pd.Timestamp("2024-01-17")
    assert round(float(row["forward_return"]), 4) == 0.3
    assert bool(row["winner_flag"]) is True


def test_market_regime_and_buckets_attach_values() -> None:
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["AAA", "BBB"],
            "close": [110.0, 90.0],
            "sma_50": [100.0, 100.0],
            "sma_200": [100.0, 100.0],
            "rsi_14_daily": [65.0, 55.0],
            "volume_vs_20d": [2.0, 1.0],
            "return_20d": [0.1, -0.1],
            "delivery_pct": [0.6, 0.4],
            "avg_traded_value_20d": [5e7, 5e6],
        }
    )
    enriched = add_market_regime_features(df)
    enriched = add_bucket_columns(enriched)
    row = enriched.iloc[0]
    assert round(float(row["breadth_above_50_dma"]), 4) == 0.5
    assert round(float(row["breadth_rsi_60"]), 4) == 0.5
    assert row["price_bucket"] in {"50-200", ">2000"}
    assert pd.notna(row["liquidity_bucket_20d_cr"])
