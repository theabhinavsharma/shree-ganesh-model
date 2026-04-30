from __future__ import annotations

import pandas as pd

from src.report.weekly_portfolio_report import build_weekly_portfolio_table


def test_build_weekly_portfolio_table_allocations_sum_to_investable() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "company_name": ["Alpha", "Beta", "Gamma"],
            "close": [100.0, 200.0, 300.0],
            "sma_50": [95.0, 190.0, 270.0],
            "model_score": [1.2, 0.9, 0.3],
            "model_pass_count": [8, 7, 6],
            "avg_traded_value_20d": [2_500_000_000, 700_000_000, 150_000_000],
            "instrument_type": ["EQ", "EQ", "EQ"],
            "filter_above_200_dma": [True, True, False],
            "filter_above_50_dma": [True, True, True],
            "pe_ttm": [18.0, 12.0, 25.0],
            "promoter_pct": [65.0, 51.0, 45.0],
            "revenue_cagr_5y": [0.12, 0.15, 0.09],
            "pat_cagr_5y": [0.25, 0.22, 0.05],
        }
    )

    report = build_weekly_portfolio_table(universe, portfolio_size=3, cash_buffer_pct=10.0, target_date="2026-12-31")

    assert len(report) == 3
    assert round(float(report["allocation_pct"].sum()), 2) == 90.00
    assert list(report["rank"]) == [1, 2, 3]
    assert report.loc[0, "buy_price_range"] == "99.00 - 103.00"
    assert report.loc[0, "sell_target"] == 150.0


def test_build_weekly_portfolio_table_tolerates_nullable_boolean_reason_flags() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "company_name": ["Alpha"],
            "close": [100.0],
            "sma_50": [95.0],
            "model_score": [1.2],
            "model_pass_count": [8],
            "avg_traded_value_20d": [2_500_000_000],
            "instrument_type": ["EQ"],
            "filter_above_200_dma": [pd.NA],
            "filter_above_50_dma": [True],
            "recent_promoter_buy_flag": [pd.NA],
            "recent_order_win_flag": [pd.NA],
            "recent_approval_flag": [pd.NA],
        }
    )

    report = build_weekly_portfolio_table(universe, portfolio_size=1, cash_buffer_pct=10.0, target_date="2026-04-08")

    assert len(report) == 1
    assert report.loc[0, "reasons"] == "above 50 DMA"


def test_build_weekly_portfolio_table_prefers_checklist_buy_candidates_and_channel_levels() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "company_name": ["Alpha", "Beta"],
            "sector": ["IT", "IT"],
            "close": [100.0, 120.0],
            "sma_50": [95.0, 115.0],
            "model_score": [1.2, 1.1],
            "model_pass_count": [8, 7],
            "avg_traded_value_20d": [2_500_000_000, 2_000_000_000],
            "instrument_type": ["EQ", "EQ"],
            "strategy_checklist_pass": [True, True],
            "trade_action": ["Buy", "Hold"],
            "channel_buy_price_low": [98.0, 118.0],
            "channel_buy_price_high": [101.0, 121.0],
            "channel_sell_target": [112.0, 124.0],
            "channel_stop_loss": [94.0, 116.0],
            "market_cap_cr": [6500.0, 7000.0],
        }
    )

    report = build_weekly_portfolio_table(universe, portfolio_size=2, cash_buffer_pct=10.0, target_date="2026-04-08")

    assert len(report) == 1
    assert report.loc[0, "symbol"] == "AAA"
    assert report.loc[0, "buy_price_range"] == "98.00 - 101.00"
    assert report.loc[0, "sell_target"] == 112.0
    assert report.loc[0, "trade_action"] == "Buy"


def test_build_weekly_portfolio_table_priority_p0_ranks_by_reliable_filter_coverage() -> None:
    universe = pd.DataFrame(
        {
            "symbol": ["ETF1", "AAA", "BBB", "CCC"],
            "company_name": ["ETF", "Alpha", "Beta", "Gamma"],
            "sector": ["ETF", "Chemicals", "Shipping", "Power"],
            "instrument_type": ["ETF", "EQ", "EQ", "EQ"],
            "close": [100.0, 110.0, 120.0, 130.0],
            "sma_50": [99.0, 100.0, 110.0, 120.0],
            "model_score": [9.0, 2.0, 1.5, 1.0],
            "model_pass_count": [99, 8, 7, 6],
            "avg_traded_value_20d": [3_000_000_000, 2_500_000_000, 2_000_000_000, 1_500_000_000],
            "trade_action": ["Buy", None, "Buy", "Hold"],
            "filter_market_cap": [True, True, True, True],
            "filter_debt": [True, True, pd.NA, False],
            "filter_revenue_growth": [True, True, True, True],
            "filter_profit_cagr": [True, True, True, False],
            "filter_volume_expansion": [True, True, False, True],
            "filter_volume_high_3m": [True, True, True, False],
            "filter_delivery_above_5d_avg": [True, False, True, True],
            "filter_rsi_daily": [True, True, True, False],
            "filter_rsi_weekly": [True, True, True, False],
            "filter_rsi_monthly": [True, False, True, False],
            "filter_pe": [True, True, True, True],
            "filter_promoter_holding": [True, True, True, True],
            "filter_above_50_dma": [True, True, True, True],
            "filter_above_200_dma": [True, True, True, False],
        }
    )

    report = build_weekly_portfolio_table(
        universe,
        portfolio_size=3,
        cash_buffer_pct=10.0,
        target_date="2026-04-08",
        selection_mode="priority_p0",
    )

    assert list(report["symbol"]) == ["BBB", "AAA", "CCC"]
    assert list(report["selection_mode"].unique()) == ["priority_p0"]
    assert report.loc[0, "screen_pass_count"] == 12
    assert report.loc[1, "screen_pass_count"] == 12
    assert report.loc[0, "screen_pass_ratio"] > report.loc[1, "screen_pass_ratio"]
