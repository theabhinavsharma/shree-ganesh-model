import pandas as pd

from src.screen.build_universe import apply_screen_filters
from src.screen.build_universe import build_daily_screen_universe


def test_screen_marks_missing_inputs_instead_of_fake_filters() -> None:
    daily_facts = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["ABC"],
            "close": [100.0],
            "sma_50": [99.0],
            "sma_200": [98.0],
            "volume_vs_20d": [2.0],
            "rsi_14_daily": [60.0],
        }
    )
    stock_master = pd.DataFrame({"symbol": ["ABC"], "sector": ["IT"]})
    result = build_daily_screen_universe(
        daily_facts=daily_facts,
        stock_master=stock_master,
        fundamentals=pd.DataFrame(),
        shareholding=pd.DataFrame(),
        sector_state_daily=pd.DataFrame(),
        config={
            "universe": {
                "require_above_50_dma": True,
                "min_volume_vs_20d": 1.5,
                "min_rsi_14_daily": 55.0,
            }
        },
    )
    row = result.iloc[0]
    assert bool(row["filter_above_50_dma"]) is True
    assert pd.isna(row["filter_promoter_holding"])
    assert "promoter_pct" in row["missing_inputs"]


def test_screen_can_pass_drop_134_strategy_when_all_remaining_rules_are_met() -> None:
    daily_facts = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["ABC"],
            "close": [100.0],
            "sma_50": [90.0],
            "sma_200": [80.0],
            "volume_vs_20d": [2.0],
            "volume_high_63d_flag": [True],
            "delivery_pct_high_63d_flag": [True],
            "rsi_14_daily": [65.0],
            "rsi_14_weekly": [64.0],
            "rsi_14_monthly": [63.0],
        }
    )
    stock_master = pd.DataFrame({"symbol": ["ABC"], "sector": ["IT"]})
    fundamentals = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "effective_from_date": pd.to_datetime(["2023-12-15"]),
            "revenue_cagr_5y": [0.12],
            "pat_cagr_5y": [0.25],
            "ebitda_positive_last_5q_flag": [True],
            "eps_ttm": [4.0],
        }
    )
    shareholding = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "effective_from_date": pd.to_datetime(["2023-12-20"]),
            "promoter_pct": [55.0],
        }
    )
    result = build_daily_screen_universe(
        daily_facts=daily_facts,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=pd.DataFrame(),
        config={
            "universe": {
                "min_volume_vs_20d": 1.5,
                "require_volume_high_3m": True,
                "require_delivery_high_3m": True,
                "require_above_50_dma": True,
                "require_above_200_dma": True,
                "min_rsi_14_daily": 60.0,
                "min_rsi_14_weekly": 60.0,
                "min_rsi_14_monthly": 60.0,
                "max_pe_ttm": 30.0,
                "min_promoter_pct": 50.0,
                "min_revenue_cagr_5y": 0.10,
                "min_pat_cagr_5y": 0.20,
            }
        },
    )
    row = result.iloc[0]
    assert bool(row["filter_pe"]) is True
    assert bool(row["filter_rsi"]) is True
    assert bool(row["strategy_drop_134_pass"]) is True


def test_screen_can_pass_full_checklist_when_all_inputs_are_available() -> None:
    daily_facts = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["ABC"],
            "close": [100.0],
            "sma_50": [90.0],
            "sma_200": [80.0],
            "volume_vs_20d": [2.0],
            "volume_high_63d_flag": [True],
            "delivery_above_5d_avg_flag": [True],
            "rsi_14_daily": [65.0],
            "rsi_14_weekly": [64.0],
            "rsi_14_monthly": [63.0],
            "market_cap_cr": [8000.0],
        }
    )
    stock_master = pd.DataFrame({"symbol": ["ABC"], "sector": ["IT"]})
    fundamentals = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "effective_from_date": pd.to_datetime(["2023-12-15"]),
            "revenue_cagr_5y": [0.12],
            "pat_cagr_5y": [0.25],
            "ebitda_positive_last_5q_flag": [True],
            "eps_ttm": [4.0],
            "debt_equity_ratio": [0.0],
        }
    )
    shareholding = pd.DataFrame(
        {
            "symbol": ["ABC"],
            "effective_from_date": pd.to_datetime(["2023-12-20"]),
            "promoter_pct": [45.0],
            "fii_fpi_pct_qoq_change": [0.4],
            "dii_pct_qoq_change": [0.2],
        }
    )
    result = build_daily_screen_universe(
        daily_facts=daily_facts,
        stock_master=stock_master,
        fundamentals=fundamentals,
        shareholding=shareholding,
        sector_state_daily=pd.DataFrame(),
        config={
            "universe": {
                "min_market_cap": 5000.0,
                "min_volume_vs_20d": 1.5,
                "require_volume_high_3m": True,
                "require_delivery_above_5d_avg": True,
                "require_above_50_dma": True,
                "require_above_200_dma": True,
                "min_rsi_14_daily": 60.0,
                "min_rsi_14_weekly": 60.0,
                "min_rsi_14_monthly": 60.0,
                "max_pe_ttm": 30.0,
                "min_promoter_pct": 40.0,
                "min_revenue_cagr_5y": 0.10,
                "min_pat_cagr_5y": 0.20,
                "require_debt_free": True,
                "require_sector_fii_dii_buying_30d": True,
            }
        },
    )
    row = result.iloc[0]
    assert bool(row["filter_market_cap"]) is True
    assert bool(row["filter_debt"]) is True
    assert bool(row["filter_sector_institutional_buying"]) is True
    assert bool(row["strategy_checklist_pass"]) is True


def test_apply_screen_filters_recomputes_sector_proxy_when_proxy_column_already_exists() -> None:
    universe = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["ABC"],
            "sector": ["IT"],
            "close": [100.0],
            "sma_50": [90.0],
            "sma_200": [80.0],
            "volume_vs_20d": [2.0],
            "rsi_14_daily": [61.0],
            "rsi_14_weekly": [62.0],
            "rsi_14_monthly": [63.0],
            "fii_fpi_pct_qoq_change": [0.4],
            "dii_pct_qoq_change": [0.2],
            "sector_fii_dii_buying_proxy_flag": [pd.NA],
        }
    )

    result = apply_screen_filters(
        universe,
        config={"universe": {"require_sector_fii_dii_buying_30d": True}},
    )

    row = result.iloc[0]
    assert bool(row["sector_fii_dii_buying_proxy_flag"]) is True
    assert bool(row["filter_sector_institutional_buying"]) is True


def test_apply_screen_filters_preserves_existing_pe_ttm_when_eps_is_missing() -> None:
    universe = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["ABC"],
            "close": [100.0],
            "pe_ttm": [22.5],
        }
    )

    result = apply_screen_filters(
        universe,
        config={"universe": {"max_pe_ttm": 30.0}},
    )

    row = result.iloc[0]
    assert float(row["pe_ttm"]) == 22.5
    assert bool(row["filter_pe"]) is True
