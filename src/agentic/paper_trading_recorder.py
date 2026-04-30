"""Paper-trading recorder: every run, score yesterday's brief picks against
today's actual close. Builds an append-only ledger that lets us calibrate live
slippage and live hit-rate vs OOS-claimed precision.

Logic:
  1. On each run, look up the most recent brief (reports/daily_brief_*.md or
     tmp/from_scratch_7d_run/v3_live_top100.csv from yesterday's run).
  2. For each pick, compute the realized 1d-forward return using today's close.
  3. After 7 trading days, mark the pick "closed" with realized 7TD high+close+low
     and write a final outcome row.
  4. Produce a rolling P&L summary: hits, mean realized return, mean drawdown,
     edge vs OOS-claimed precision (so we know if we're degrading).
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
LEDGER = ROOT / "data/derived/paper_trading_ledger.parquet"
LIVE = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"


def record_picks_today() -> pd.DataFrame:
    """Snapshot today's top-20 picks into the ledger (status=open)."""
    if not LIVE.exists():
        print(f"no live picks at {LIVE}")
        return pd.DataFrame()
    live = pd.read_csv(LIVE).sort_values("score_ens", ascending=False).head(20)
    today = pd.Timestamp(live["trade_date"].iloc[0]) if "trade_date" in live.columns else pd.Timestamp(date.today())
    rows = []
    for _, r in live.iterrows():
        rows.append({
            "snapshot_date": today,
            "symbol": r["symbol"],
            "sector": r.get("sector", "OTHER"),
            "entry_close": r["close"],
            "pwin_ens": r["score_ens"],
            "pwin_cal": r["score_calibrated"],
            "status": "open",
            "fwd_d1_close": np.nan,
            "fwd_d3_close": np.nan,
            "fwd_d7_high": np.nan,
            "fwd_d7_close": np.nan,
            "fwd_d7_low": np.nan,
            "realized_return_7d_pct": np.nan,
            "realized_max_high_pct": np.nan,
            "realized_min_low_pct": np.nan,
            "outcome": np.nan,  # 'win' (>=+5% high), 'loss' (<-5% low), 'flat'
        })
    return pd.DataFrame(rows)


def update_open_positions(ledger: pd.DataFrame) -> pd.DataFrame:
    """For each open position, fill fwd return columns using latest price data."""
    if ledger.empty:
        return ledger
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "high", "low"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])

    open_mask = ledger["status"] == "open"
    for idx in ledger[open_mask].index:
        sym = ledger.at[idx, "symbol"]
        snap_date = ledger.at[idx, "snapshot_date"]
        future = px[(px["symbol"] == sym) & (px["trade_date"] > snap_date)].head(7)
        if len(future) == 0:
            continue
        entry = ledger.at[idx, "entry_close"]
        ledger.at[idx, "fwd_d1_close"] = future["close"].iloc[0] if len(future) >= 1 else np.nan
        ledger.at[idx, "fwd_d3_close"] = future["close"].iloc[2] if len(future) >= 3 else np.nan
        if len(future) >= 7:
            ledger.at[idx, "fwd_d7_close"] = future["close"].iloc[6]
            ledger.at[idx, "fwd_d7_high"] = future["high"].max()
            ledger.at[idx, "fwd_d7_low"] = future["low"].min()
            ret_c7 = future["close"].iloc[6] / entry - 1
            ret_max = future["high"].max() / entry - 1
            ret_min = future["low"].min() / entry - 1
            ledger.at[idx, "realized_return_7d_pct"] = ret_c7
            ledger.at[idx, "realized_max_high_pct"] = ret_max
            ledger.at[idx, "realized_min_low_pct"] = ret_min
            if ret_max >= 0.05:
                ledger.at[idx, "outcome"] = "win"
            elif ret_min <= -0.05:
                ledger.at[idx, "outcome"] = "loss"
            else:
                ledger.at[idx, "outcome"] = "flat"
            ledger.at[idx, "status"] = "closed"
    return ledger


def summarize(ledger: pd.DataFrame) -> None:
    closed = ledger[ledger["status"] == "closed"]
    print(f"\n=== paper-trading ledger summary ===")
    print(f"total snapshots: {len(ledger)}, closed: {len(closed)}, open: {len(ledger) - len(closed)}")
    if len(closed) == 0:
        return
    print(f"  hit rate (>= +5% high in 7d): {(closed['outcome']=='win').mean():.1%}")
    print(f"  mean realized C2C 7TD ret  : {closed['realized_return_7d_pct'].mean()*100:+.2f}%")
    print(f"  median realized C2C 7TD ret: {closed['realized_return_7d_pct'].median()*100:+.2f}%")
    print(f"  mean drawdown (worst-low)  : {closed['realized_min_low_pct'].mean()*100:+.2f}%")
    by_score = closed.copy()
    by_score["band"] = pd.cut(by_score["pwin_ens"],
                              bins=[0, 0.6, 0.7, 0.8, 0.9, 1.01],
                              labels=["<60", "60-70", "70-80", "80-90", "90+"])
    agg = by_score.groupby("band", observed=False).agg(
        n=("symbol", "size"),
        hit_rate=("outcome", lambda s: (s == "win").mean()),
        mean_ret=("realized_return_7d_pct", "mean"),
        mean_dd=("realized_min_low_pct", "mean"),
    ).round(3)
    print("\n  per-Pwin-band realized:")
    print(agg.to_string())


def main() -> None:
    ledger = pd.read_parquet(LEDGER) if LEDGER.exists() else pd.DataFrame()
    today = record_picks_today()
    if len(today):
        # dedupe on (snapshot_date, symbol)
        ledger = pd.concat([ledger, today], ignore_index=True)
        ledger = ledger.drop_duplicates(["snapshot_date", "symbol"], keep="last")
    ledger = update_open_positions(ledger)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(LEDGER, index=False)
    summarize(ledger)
    print(f"\nwrote {LEDGER}")


if __name__ == "__main__":
    main()
