"""Refresh the corporate-actions store (splits/bonuses) from NSE.

Extracted 2026-08-27 from the ad-hoc heredocs in monday_run_20260824.sh /
friday_run_20260828.sh so the weekly pipeline can refresh CAs as a proper step —
and BEFORE refresh_prices.py, so price adjustment always sees today's actions
(the late-arriving-CA bug class: CORDELIA/TDPOWERSYS/GOODLUCK/KIRLPNU 2026-08-27,
TRENT/LICI "113-day rot" 2026-08-18).

Fetch window: trailing 30 days (idempotent merge, dedup on symbol/ex_date/subject/series).
Exit 1 on fetch failure so run scripts can decide fatality.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
from src.ingest.corporate_actions.nse import (  # noqa: E402
    NseCorporateActionsFetchConfig,
    load_corporate_actions_from_nse,
)

STORE = ROOT / "data/corporate_actions_full_history/normalized/stock_corporate_actions.parquet"
WINDOW_DAYS = 30


def main() -> None:
    start = date.today() - timedelta(days=WINDOW_DAYS)
    new = load_corporate_actions_from_nse(NseCorporateActionsFetchConfig(
        output_dir=ROOT / "data/corporate_actions_full_history/_incremental",
        start_date=start,
        end_date=date.today(),
    ))
    new["ex_date"] = pd.to_datetime(new["ex_date"], errors="coerce")
    full = pd.read_parquet(STORE)
    full["ex_date"] = pd.to_datetime(full["ex_date"], errors="coerce")
    key = ["symbol", "ex_date", "subject", "series"]
    merged = (
        pd.concat([full, new], ignore_index=True)
        .drop_duplicates(subset=key, keep="last")
        .sort_values(["ex_date", "symbol"])
        .reset_index(drop=True)
    )
    merged.to_parquet(STORE, index=False)
    fresh = new[new["adjustment_factor"].notna()]
    print(f"store: {len(merged):,} rows · max ex_date {merged['ex_date'].max().date()} · "
          f"fetched {len(new)} rows ({len(fresh)} with factors) since {start}")
    if len(fresh):
        print(fresh[["symbol", "ex_date", "subject", "adjustment_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
