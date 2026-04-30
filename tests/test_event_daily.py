import pandas as pd

from src.transform.event_daily import build_event_feature_daily


def test_build_event_feature_daily_rolls_counts_and_recency() -> None:
    calendar = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-20", "2024-02-10"]),
        }
    )
    announcements = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC"],
            "event_date": pd.to_datetime(["2024-01-05", "2024-01-18"]),
            "sequence_id": ["1", "2"],
            "is_results_event": [True, False],
            "is_order_win": [False, True],
            "is_approval": [False, False],
            "is_pledge_change": [False, False],
            "is_promoter_buying": [False, False],
        }
    )

    result = build_event_feature_daily(calendar, announcements)
    row_1 = result.loc[result["trade_date"] == pd.Timestamp("2024-01-10")].iloc[0]
    row_2 = result.loc[result["trade_date"] == pd.Timestamp("2024-01-20")].iloc[0]
    row_3 = result.loc[result["trade_date"] == pd.Timestamp("2024-02-10")].iloc[0]

    assert row_1["announcements_7d"] == 1
    assert row_1["results_events_30d"] == 1
    assert row_1["days_since_results_event"] == 0
    assert bool(row_2["recent_order_win_flag"])
    assert row_2["days_since_order_win"] == 0
    assert row_3["announcements_30d"] == 1


def test_build_event_feature_daily_includes_insider_bulk_and_oi_lagged_to_next_trade_day() -> None:
    calendar = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"]),
        }
    )
    announcements = pd.DataFrame(columns=["symbol", "event_date", "sequence_id"])
    insider = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "event_date": pd.to_datetime(["2024-01-10"]),
            "buy_value": [100000.0],
            "sell_value": [0.0],
            "net_value": [100000.0],
            "is_buy_transaction": [True],
            "is_sell_transaction": [False],
            "is_promoter_group_or_promoter": [True],
            "is_director_or_kmp": [False],
        }
    )
    bulk_block = pd.DataFrame(
        {
            "deal_type": ["bulk_deals", "block_deals"],
            "symbol": ["ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
            "traded_value": [50000.0, 25000.0],
            "quantity_traded": [1000.0, 200.0],
            "is_buy": [True, True],
            "is_sell": [False, False],
        }
    )
    oi = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC"],
            "trade_date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
            "mwpl": [1000.0, 1000.0],
            "ncl_open_interest": [300.0, 360.0],
            "ncl_futeq_oi": [200.0, 220.0],
            "oi_share_of_mwpl": [0.30, 0.36],
            "oi_change_1d": [pd.NA, 60.0],
            "oi_change_pct_1d": [pd.NA, 0.20],
            "futeq_oi_change_1d": [pd.NA, 20.0],
            "oi_share_of_mwpl_change_1d": [pd.NA, 0.06],
        }
    )

    result = build_event_feature_daily(
        calendar,
        announcements,
        insider_trades=insider,
        bulk_block_deals=bulk_block,
        derivatives_oi=oi,
    )
    row_10 = result.loc[result["trade_date"] == pd.Timestamp("2024-01-10")].iloc[0]
    row_11 = result.loc[result["trade_date"] == pd.Timestamp("2024-01-11")].iloc[0]
    row_12 = result.loc[result["trade_date"] == pd.Timestamp("2024-01-12")].iloc[0]

    assert pd.isna(row_10["insider_buy_value_30d"])
    assert row_11["insider_buy_value_30d"] == 100000.0
    assert bool(row_11["recent_bulk_buy_flag"])
    assert row_12["block_buy_value_30d"] == 25000.0
    assert row_11["oi_share_of_mwpl"] == 0.30
    assert row_12["oi_change_1d"] == 60.0
