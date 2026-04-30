"""Fetch the latest NSE bhavcopy(s) and append/refresh stock_daily_facts parquet.

Idempotent: re-running on a date that's already in the parquet will overwrite
the row(s) for that date. The post-close pipeline calls this BEFORE retraining.
"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingest.nse.fetch_bhavcopy import fetch_bhavcopy_range
from src.ingest.nse.models import BhavcopyFetchRequest
from src.transform.build_daily_facts import build_stock_daily_facts
from src.features.indicators import add_daily_price_features

RAW = Path("data/raw/nse_full_history_official")
PARQUET = Path("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
CA_PATH = Path("data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet")
LOOKBACK_DAYS = 5


def main() -> None:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    print(f"fetching bhavcopy {start} → {today}")
    fetch_bhavcopy_range(BhavcopyFetchRequest(
        start_date=start, end_date=today, output_dir=RAW, delay_seconds=1.0,
    ))

    # Re-derive daily facts for the most recent ~LOOKBACK_DAYS partitions, then
    # splice them into the existing parquet.
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    new_dirs = []
    for d in sorted((RAW).glob("trade_date=*")):
        td = pd.Timestamp(d.name.split("=", 1)[1]).date()
        if td >= start:
            link = tmp / d.name
            link.symlink_to(d.absolute())
            new_dirs.append(td)
    if not new_dirs:
        print("no new dates — exiting")
        return

    new_df = build_stock_daily_facts(
        tmp,
        corporate_actions_path=CA_PATH if CA_PATH.exists() else None,
        use_adjusted_prices=CA_PATH.exists(),
    )
    new_df["trade_date"] = pd.to_datetime(new_df["trade_date"])
    print(f"new rows: {len(new_df):,}  dates: {new_df['trade_date'].min().date()} → {new_df['trade_date'].max().date()}")

    old = pd.read_parquet(PARQUET)
    old["trade_date"] = pd.to_datetime(old["trade_date"])
    cutoff = pd.Timestamp(start)
    old = old[old["trade_date"] < cutoff].copy()

    cols = old.columns.intersection(new_df.columns)
    combined = pd.concat([old[cols], new_df[cols]], ignore_index=True)
    combined = combined.drop_duplicates(["symbol", "trade_date"], keep="last")
    combined = combined.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # Recompute rolling features so the latest rows have valid sma/rsi/return.
    roll_cols = [c for c in combined.columns if c.startswith(
        ("sma_", "rsi_", "return_", "volume_vs_", "traded_value_vs_", "avg_traded_value_")
    )]
    base = combined.drop(columns=roll_cols)
    featured = add_daily_price_features(base)

    featured.to_parquet(PARQUET, index=False)
    print(f"wrote {PARQUET}")
    print(f"max trade_date in parquet: {featured['trade_date'].max().date()}")


if __name__ == "__main__":
    main()
