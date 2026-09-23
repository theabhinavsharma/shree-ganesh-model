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
    _parse_bonus_factor,
    _parse_split_factor,
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
    healed = _reparse_factorless(merged)
    merged.to_parquet(STORE, index=False)
    if healed:
        print(f"STORE SELF-HEAL: {healed} historical rows gained a split/bonus factor "
              f"from the current parser (refresh_prices.py re-adjusts their symbols next)")
    fresh = new[new["adjustment_factor"].notna()]
    print(f"store: {len(merged):,} rows · max ex_date {merged['ex_date'].max().date()} · "
          f"fetched {len(new)} rows ({len(fresh)} with factors) since {start}")
    if len(fresh):
        print(fresh[["symbol", "ex_date", "subject", "adjustment_factor"]].to_string(index=False))


def _reparse_factorless(store: pd.DataFrame) -> int:
    """Re-run the CURRENT subject parser over the whole store; return rows whose
    factor changed.

    2026-09-19: the 2026-08-27 regex fix only ever touched newly fetched rows, so
    59 historical splits (JSWSTEEL/KARURVYSYA/TTL/DBEIL class) stayed factor-less
    and 13 same-day bonus+split pairs carried a half factor (CUPID 2 vs raw 12.6x,
    SBC 2 vs 29.6x) for three weeks after the parser could read them. Verified
    against raw ex-date close ratios: 23/24 changed rows move CLOSER to the tape.
    The store is a pure function of (NSE subject text, current parser) — nothing
    is hand-edited (source_note is uniform) — so re-parsing everything is safe and
    idempotent. refresh_prices.py's LATE-CA SELF-HEAL re-adjusts any symbol whose
    per-ex_date product changed.
    """
    bonus = store["subject"].map(_parse_bonus_factor)
    split = store["subject"].map(_parse_split_factor)
    factor = (bonus.fillna(1.0) * split.fillna(1.0)).where(bonus.notna() | split.notna())
    changed = factor.fillna(-1.0) != store["adjustment_factor"].astype(float).fillna(-1.0)
    if not changed.any():
        return 0
    store["bonus_factor"] = bonus
    store["split_factor"] = split
    store["is_bonus"] = bonus.notna()
    store["is_split"] = split.notna()
    store["adjustment_factor"] = factor
    return int(changed.sum())

if __name__ == "__main__":
    main()
