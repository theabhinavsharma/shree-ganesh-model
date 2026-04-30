from __future__ import annotations

import numpy as np
import pandas as pd

from src.report.checklist import build_trade_channel_snapshot


def test_build_trade_channel_snapshot_detects_buy_zone(tmp_path) -> None:
    trade_dates = pd.date_range("2025-01-01", periods=80, freq="B")
    base = np.linspace(100.0, 140.0, num=len(trade_dates))
    wave = np.sin(np.linspace(0, 8 * np.pi, num=len(trade_dates))) * 2.5
    low = base - 4.0 + wave
    high = base + 4.0 + wave
    close = base + wave
    close[-1] = low[-1] + 0.8
    history = pd.DataFrame(
        {
            "symbol": ["AAA"] * len(trade_dates),
            "trade_date": trade_dates,
            "high": high,
            "low": low,
            "close": close,
        }
    )
    history_path = tmp_path / "daily_facts.parquet"
    history.to_parquet(history_path, index=False)

    result = build_trade_channel_snapshot(daily_facts_path=history_path, symbols={"AAA"})

    assert len(result) == 1
    row = result.iloc[0]
    assert bool(row["channel_valid_flag"]) is True
    assert row["trade_action"] == "Buy"
    assert float(row["channel_sell_target"]) > float(row["channel_buy_price_high"])
    assert float(row["channel_stop_loss"]) < float(row["channel_buy_price_low"])
