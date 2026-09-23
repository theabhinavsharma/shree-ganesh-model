from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.indicators import add_daily_price_features
from src.transform.corporate_actions import apply_split_bonus_adjustments
from src.ingest.nse.fetch_bhavcopy import build_nse_bhavcopy_url, build_nse_delivery_url
from src.ingest.nse.normalize import normalize_trade_date_directory
from src.utils.io import read_parquet_if_exists
from src.utils.validation import assert_unique_key

# Main-board equity incl. trade-to-trade surveillance series. SME (SM/ST/SZ), debt and
# G-sec series stay out. RE-APPLIED 2026-09-24: the 2026-08-28 BE/BZ fix (reports/
# be_series_gap_repair_20260828.md) was never committed and the working copy was
# overwritten on 2026-09-10, so refresh_prices dropped ~270 symbols/day again from
# 09-08 to 09-23 (0 BE/BZ rows in the panel). A fix that is not in git is not a fix.
EQUITY_SERIES = {"EQ", "BE", "BZ"}
T2T_SERIES = {"BE", "BZ"}          # compulsory delivery; bhavcopy reports DELIV as "-"


def _keep_equity_series(frame: pd.DataFrame) -> pd.DataFrame:
    if "series" not in frame.columns:
        return frame
    ser = frame["series"].fillna("").astype(str).str.upper()
    out = frame[ser.isin(EQUITY_SERIES) | frame["series"].isna()].copy()
    t2t = out["series"].fillna("").astype(str).str.upper().isin(T2T_SERIES)
    if "delivery_pct" in out.columns:
        out.loc[t2t & out["delivery_pct"].isna(), "delivery_pct"] = 1.0
    if "deliverable_qty" in out.columns and "total_traded_qty" in out.columns:
        out.loc[t2t & out["deliverable_qty"].isna(), "deliverable_qty"] = out.loc[t2t & out["deliverable_qty"].isna(), "total_traded_qty"]
    return out


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
    combined = _keep_equity_series(combined)
    if use_adjusted_prices:
        corporate_actions = read_parquet_if_exists(corporate_actions_path) if corporate_actions_path else pd.DataFrame()
        combined = apply_split_bonus_adjustments(combined, corporate_actions)
    featured = add_daily_price_features(combined)
    assert_unique_key(featured, ["trade_date", "symbol"])
    return featured
