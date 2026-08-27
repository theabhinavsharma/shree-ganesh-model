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


MAX_BACKFILL_DAYS = 90     # safety cap for adaptive backfill


def main() -> None:
    today = date.today()
    # Optional explicit backfill start: python3 refresh_prices.py --start 2026-07-07
    forced_start = None
    if "--start" in sys.argv:
        forced_start = date.fromisoformat(sys.argv[sys.argv.index("--start") + 1])
    # ADAPTIVE lookback (2026-07-24 fix): the fixed 5-day window silently left a
    # 9-session hole (Jul-7..17) when runs were >5 days apart. Start from the
    # day after the parquet's actual max date, capped at MAX_BACKFILL_DAYS.
    start = forced_start or (today - timedelta(days=LOOKBACK_DAYS))
    if forced_start is None and PARQUET.exists():
        try:
            _existing_max = pd.to_datetime(
                pd.read_parquet(PARQUET, columns=["trade_date"])["trade_date"]
            ).max().date()
            gap_start = _existing_max + timedelta(days=1)
            if gap_start < start:
                start = max(gap_start, today - timedelta(days=MAX_BACKFILL_DAYS))
                print(f"adaptive backfill: parquet max={_existing_max}, extending window")
        except Exception as e:
            print(f"adaptive lookback check failed ({e}) — using fixed {LOOKBACK_DAYS}d")
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

    # --- LATE-CA SELF-HEAL (2026-08-27): a corporate action that reaches the store
    # AFTER its ex-date rows were built leaves the symbol's history mis-adjusted
    # (CORDELIA/TDPOWERSYS/GOODLUCK/KIRLPNU; same class as the 113-day rot). Detect:
    # stored price_adjustment_factor_to_present vs the step function expected from
    # the CURRENT store; heal: restore raw values and re-apply the production adjuster.
    if CA_PATH.exists() and "price_adjustment_factor_to_present" in combined.columns:
        import numpy as np
        from src.transform.corporate_actions import (
            PRICE_COLUMNS, QTY_COLUMNS, apply_split_bonus_adjustments,
        )
        ca = pd.read_parquet(CA_PATH)
        ca["ex_date"] = pd.to_datetime(ca["ex_date"], errors="coerce")
        eff = (
            ca[ca["adjustment_factor"].notna() & ca["adjustment_factor"].gt(0) & ca["ex_date"].notna()]
            .groupby(["symbol", "ex_date"])["adjustment_factor"].prod().reset_index()
        )
        # scan symbols that have store factors OR carry a non-identity stored factor
        # (catches symbols whose bogus factor was later nulled but rows stay divided)
        adjusted_syms = set(
            combined.loc[combined["price_adjustment_factor_to_present"].fillna(1.0) != 1.0, "symbol"].unique()
        )
        scan_syms = set(eff["symbol"].unique()) | adjusted_syms
        stale_syms = []
        for sym, g in combined[combined["symbol"].isin(scan_syms)].groupby("symbol"):
            sa = eff[eff["symbol"] == sym].sort_values("ex_date")
            td = g["trade_date"].to_numpy(dtype="datetime64[ns]")
            share = np.ones(len(g))
            if len(sa):
                exs = sa["ex_date"].to_numpy(dtype="datetime64[ns]")
                fs = sa["adjustment_factor"].astype(float).to_numpy()
                suffix = np.cumprod(fs[::-1])[::-1]
                idx = np.searchsorted(exs, td, side="right")
                share[idx < len(exs)] = suffix[idx[idx < len(exs)]]
            stored = g["price_adjustment_factor_to_present"].fillna(1.0).to_numpy(dtype=float)
            if (~np.isclose(stored, 1.0 / share, rtol=1e-6)).any():
                stale_syms.append(sym)
        if stale_syms:
            print(f"LATE-CA SELF-HEAL: re-adjusting {len(stale_syms)} symbols: {stale_syms}")
            mask = combined["symbol"].isin(stale_syms)
            part = combined[mask].copy()
            for col in list(PRICE_COLUMNS) + list(QTY_COLUMNS):
                raw = f"raw_{col}"
                if col in part.columns and raw in part.columns:
                    part[col] = part[raw].fillna(part[col])
            drop = [c for c in part.columns if c.startswith("raw_")] + [
                "share_adjustment_factor_to_present",
                "price_adjustment_factor_to_present",
                "future_split_bonus_action_count",
            ]
            part = apply_split_bonus_adjustments(
                part.drop(columns=[c for c in drop if c in part.columns]),
                ca[ca["symbol"].isin(stale_syms)],
            )
            combined = pd.concat([combined[~mask], part], ignore_index=True)
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
