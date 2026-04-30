from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.indicators import add_daily_price_features
from src.transform.corporate_actions import apply_split_bonus_adjustments
from src.ingest.nse.fetch_bhavcopy import build_nse_bhavcopy_url, build_nse_delivery_url
from src.ingest.nse.normalize import normalize_trade_date_directory
from src.utils.io import read_parquet_if_exists
from src.utils.validation import assert_unique_key


def build_stock_daily_facts(
    raw_dir: Path,
    symbol_filter: set[str] | None = None,
    *,
    corporate_actions_path: Path | None = None,
    use_adjusted_prices: bool = False,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for trade_dir in sorted(path for path in raw_dir.glob("trade_date=*") if path.is_dir()):
        trade_date = trade_dir.name.split("=")[1]
        trade_date_value = pd.Timestamp(trade_date).date()
        rows.append(
            normalize_trade_date_directory(
                trade_dir,
                trade_date_value,
                market_source_url=build_nse_bhavcopy_url(trade_date_value),
                delivery_source_url=build_nse_delivery_url(trade_date_value),
            )
        )
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["symbol"].notna()].copy()
    if symbol_filter:
        wanted = {symbol.upper() for symbol in symbol_filter}
        combined = combined[combined["symbol"].astype(str).str.upper().isin(wanted)].copy()
    combined["trade_date"] = pd.to_datetime(combined["trade_date"])
    if "series" in combined.columns:
        combined = combined[combined["series"].fillna("").str.upper().eq("EQ") | combined["series"].isna()].copy()
    if use_adjusted_prices:
        corporate_actions = read_parquet_if_exists(corporate_actions_path) if corporate_actions_path else pd.DataFrame()
        combined = apply_split_bonus_adjustments(combined, corporate_actions)
    featured = add_daily_price_features(combined)
    assert_unique_key(featured, ["trade_date", "symbol"])
    return featured
