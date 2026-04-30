import pandas as pd

from src.transform.sector_flow_daily import forward_fill_sector_flow_daily


def test_sector_flow_respects_effective_date() -> None:
    sector_flow = pd.DataFrame(
        {
            "sector_name": ["IT"],
            "effective_from_date": pd.to_datetime(["2024-01-10"]),
            "fpi_change_pct": [5.0],
        }
    )
    calendar = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-09", "2024-01-10", "2024-01-11"])})
    result = forward_fill_sector_flow_daily(sector_flow, calendar)
    assert pd.isna(result.loc[result["trade_date"] == pd.Timestamp("2024-01-09"), "fpi_change_pct"]).all()
    assert (result.loc[result["trade_date"] == pd.Timestamp("2024-01-10"), "fpi_change_pct"] == 5.0).all()
