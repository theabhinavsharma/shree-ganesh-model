from datetime import date

import pandas as pd

from src.ingest.fundamentals.nse import (
    _add_growth_fields,
    _build_financial_listing_url,
    _normalize_financial_row,
    select_preferred_statement_scope,
)


def test_financial_history_url_uses_symbol_and_date_range() -> None:
    url = _build_financial_listing_url("RELIANCE", date(2020, 1, 1), date(2025, 12, 31))
    assert "symbol=RELIANCE" in url
    assert "from_date=01-01-2020" in url
    assert "to_date=31-12-2025" in url


def test_normalize_financial_row_maps_supported_fields() -> None:
    listing_row = {
        "symbol": "ABC",
        "toDate": "31-Dec-2024",
        "filingDate": "20-Feb-2025 17:01",
        "broadCastDate": "20-Feb-2025 17:05",
    }
    detail = {
        "filingDate": "20-Feb-2025 17:01",
        "periodEndDT": "31-Dec-2024",
        "resultsData2": {
            "re_net_sale": "1000",
            "re_net_profit": "100",
            "re_dilut_eps_for_cont_dic_opr": "5",
            "re_pro_loss_bef_tax": "120",
            "re_int_new": "10",
            "re_depr_und_exp": "20",
        },
    }
    row = _normalize_financial_row(listing_row, detail)
    assert row["revenue"] == 1000.0
    assert row["pat"] == 100.0
    assert row["eps"] == 5.0
    assert row["ebit"] == 130.0
    assert row["ebitda"] == 150.0
    assert row["interest_coverage"] == 13.0


def test_normalize_financial_row_derives_pbt_from_pat_and_tax_when_missing() -> None:
    listing_row = {
        "symbol": "ABC",
        "toDate": "31-Dec-2024",
        "filingDate": "20-Feb-2025 17:01",
        "broadCastDate": "20-Feb-2025 17:05",
    }
    detail = {
        "filingDate": "20-Feb-2025 17:01",
        "periodEndDT": "31-Dec-2024",
        "resultsData2": {
            "re_net_profit": "100",
            "re_tax": "25",
            "re_int_new": "10",
            "re_depr_und_exp": "20",
        },
    }
    row = _normalize_financial_row(listing_row, detail)
    assert row["pat"] == 100.0
    assert row["ebit"] == 135.0
    assert row["ebitda"] == 155.0
    assert row["interest_coverage"] == 13.5


def test_select_preferred_statement_scope_prefers_more_complete_series() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["ABC", "ABC", "ABC", "ABC", "XYZ"],
            "statement_scope": ["Non-Consolidated", "Consolidated", "Consolidated", "Consolidated", "Consolidated"],
            "fiscal_period_end": pd.to_datetime(["2024-03-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-03-31"]),
            "effective_from_date": pd.to_datetime(["2024-05-01", "2024-05-01", "2024-08-01", "2024-11-01", "2024-05-01"]),
            "revenue": [100.0, 100.0, 120.0, 130.0, 50.0],
            "pat": [10.0, 10.0, 11.0, 12.0, 5.0],
        }
    )
    preferred = select_preferred_statement_scope(df)
    assert preferred[preferred["symbol"] == "ABC"]["statement_scope"].unique().tolist() == ["Consolidated"]
    assert preferred[preferred["symbol"] == "XYZ"]["statement_scope"].unique().tolist() == ["Consolidated"]


def test_add_growth_fields_builds_yoy_and_acceleration() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["ABC"] * 5,
            "statement_scope": ["Non-Consolidated"] * 5,
            "fiscal_period_end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"]),
            "effective_from_date": pd.to_datetime(["2023-05-01", "2023-08-01", "2023-11-01", "2024-02-01", "2024-05-01"]),
            "revenue": [100.0, 120.0, 130.0, 140.0, 150.0],
            "pat": [10.0, 11.0, 12.0, 13.0, 15.0],
            "eps": [1.0, 1.1, 1.2, 1.3, 1.5],
            "ebitda": [20.0, 24.0, 26.0, 28.0, 30.0],
        }
    )
    result = _add_growth_fields(df)
    row = result.loc[result["fiscal_period_end"] == pd.Timestamp("2024-03-31")].iloc[0]
    assert float(row["revenue_yoy"]) == 0.5
    assert float(row["pat_yoy"]) == 0.5
    assert bool(row["revenue_yoy_positive_flag"])
