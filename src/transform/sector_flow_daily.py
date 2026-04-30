from __future__ import annotations

import pandas as pd


def forward_fill_sector_flow_daily(
    sector_flow: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    if sector_flow.empty:
        return pd.DataFrame(columns=["trade_date", "sector_name"])
    sector_flow = sector_flow[sector_flow["effective_from_date"].notna()].copy()
    if sector_flow.empty:
        return pd.DataFrame(columns=["trade_date", "sector_name"])
    daily_rows: list[pd.DataFrame] = []
    for sector_name, sector_df in sector_flow.groupby("sector_name"):
        merged = pd.merge_asof(
            calendar.sort_values("trade_date"),
            sector_df.sort_values("effective_from_date"),
            left_on="trade_date",
            right_on="effective_from_date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["sector_name"] = sector_name
        daily_rows.append(merged)
    return pd.concat(daily_rows, ignore_index=True)
