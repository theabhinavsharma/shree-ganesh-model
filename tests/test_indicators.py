import pandas as pd

from src.features.indicators import add_daily_price_features


def test_indicator_columns_exist() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["ABC"] * 600,
            "trade_date": pd.date_range("2022-01-03", periods=600, freq="B"),
            "close": range(1, 601),
            "total_traded_qty": [100] * 600,
            "total_traded_value": [100000.0] * 600,
            "deliverable_qty": [50] * 600,
        }
    )
    result = add_daily_price_features(df)
    last = result.iloc[-1]
    assert pd.notna(last["sma_20"])
    assert pd.notna(last["ema_200"])
    assert pd.notna(last["volume_vs_20d"])
    assert pd.notna(last["delivery_qty_vs_20d"])
    assert pd.notna(last["avg_delivery_pct_5d"])
    assert pd.notna(last["delivery_pct_vs_5d"])
    assert pd.notna(last["delivery_pct_vs_20d"])
    assert pd.notna(last["delivery_above_5d_avg_flag"])
    assert pd.notna(last["rsi_14_weekly"])
    assert pd.notna(last["rsi_14_monthly"])
    assert pd.notna(last["volume_high_63d_flag"])
    assert pd.notna(last["delivery_pct_high_63d_flag"])
    assert pd.notna(last["avg_traded_value_20d"])
    assert pd.notna(last["traded_value_vs_20d"])
