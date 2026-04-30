import pandas as pd

from src.ingest.public_fallback.groww import _cagr_from_year_dict
from src.ingest.public_fallback.groww import _extract_financial_row
from src.ingest.public_fallback.groww import _extract_stock_row


def test_cagr_from_year_dict_requires_five_points() -> None:
    assert pd.isna(_cagr_from_year_dict({"2023": 100, "2024": 110, "2025": 121}))


def test_extract_stock_row_maps_core_fields() -> None:
    stock_data = {
        "header": {"displayName": "Example Ltd"},
        "stats": {
            "marketCap": 12345.6,
            "peRatio": 18.2,
            "debtToEquity": 0.0,
        },
        "shareHoldingPattern": {
            "Sep '25": {
                "promoters": {"individual": {"percent": 51.5}, "corporation": {"percent": 2.0}},
                "mutualFunds": {"percent": 8.0},
                "otherDomesticInstitutions": {"insurance": {"percent": 3.0}, "otherFirms": {"percent": 1.0}},
                "foreignInstitutions": {"percent": 11.5},
            }
        },
        "financialStatementV2": {
            "CONSOLIDATED": [
                {
                    "title": "Revenue",
                    "yearly": {"2021": 100, "2022": 120, "2023": 130, "2024": 150, "2025": 180},
                },
                {
                    "title": "Profit",
                    "yearly": {"2021": 10, "2022": 12, "2023": 15, "2024": 18, "2025": 22},
                },
            ]
        },
    }

    row = _extract_stock_row(
        symbol="ABC",
        search_id="example-ltd",
        stock_data=stock_data,
        stock_url="https://groww.in/stocks/example-ltd",
    )

    assert row["groww_company_name"] == "Example Ltd"
    assert row["groww_market_cap_cr"] == 12345.6
    assert row["groww_pe_ttm"] == 18.2
    assert row["groww_debt_free_proxy_flag"] is True
    assert row["groww_promoter_pct"] == 53.5
    assert row["groww_fii_fpi_pct"] == 11.5
    assert row["groww_mf_pct"] == 8.0
    assert row["groww_dii_pct"] == 12.0
    assert not pd.isna(row["groww_revenue_cagr_5y_proxy"])
    assert not pd.isna(row["groww_pat_cagr_5y_proxy"])


def test_extract_financial_row_maps_ebitda_positive_flag() -> None:
    financial_data = {
        "statements": [
            {
                "title": "Income Statement",
                "consolidatedQuarterly": {
                    "financial": [
                        {"title": "Revenue", "value": [100, 110, 120, 130, 140]},
                        {"title": "EBITDA", "value": [10, 11, 12, 13, 14]},
                    ]
                },
                "standaloneQuarterly": {"financial": []},
            }
        ]
    }

    row = _extract_financial_row(financial_data, financial_url="https://groww.in/stocks/example-ltd/company-financial")

    assert row["groww_ebitda_positive_last_5q_flag"] is True
