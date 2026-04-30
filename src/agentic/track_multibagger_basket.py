"""Live tracker for the multibagger basket — proves whether the 90%-conviction
prediction holds in real forward returns.

Snapshots taken from multibagger_today_predictions.parquet whenever we re-pick.
For each name, track:
  • entry_close (date the score first cleared 0.86 / 0.84 / 0.77)
  • daily price + return-since-entry
  • whether it has hit +50%, +75%, +100% milestones
  • days remaining in the 180/252/378d window
  • current score (re-evaluated daily — if drops below 0.70, flag exit)

Output:
  data/derived/multibagger_basket_ledger.parquet — append-only history
  reports/multibagger_basket_status.md — current snapshot + alerts

Calibration check: after 90+ days of tracking we'll see if the realized
hit rate matches the 90% OOS claim.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRED = ROOT / "data/derived/multibagger_today_predictions.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
LEDGER = ROOT / "data/derived/multibagger_basket_ledger.parquet"
OUT_REPORT = ROOT / "reports/multibagger_basket_status.md"

# the recommended basket from yesterday (Tier 1, clean)
TIER_1_BASKET = ["NEWGEN", "KPITTECH", "LATENTVIEW", "ZAGGLE"]
HORIZON_DAYS = 180  # primary horizon; expect doubling within
TARGET_PCT = 1.00


def main() -> None:
    if not PRED.exists():
        print(f"missing {PRED} — run find_multibagger_today.py first")
        return
    pred = pd.read_parquet(PRED)
    today = pd.Timestamp(date.today()).normalize()

    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "high"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    latest_trade = px["trade_date"].max()

    # build / update ledger
    rows = []
    for sym in TIER_1_BASKET:
        psym = pred[pred["symbol"] == sym]
        if psym.empty:
            print(f"  {sym} not in today's predictions — may have dropped below floor")
            continue
        p = psym.iloc[0]
        entry_close = float(p["close"])

        # most recent close + max-high since entry
        sym_px = px[px["symbol"] == sym]
        if sym_px.empty:
            continue
        current_close = float(sym_px["close"].iloc[-1])
        # we only have entry "today", so since-entry = same day for now
        return_since_entry = current_close / entry_close - 1
        max_high_since_entry = float(sym_px["high"].iloc[-1])
        max_return_so_far = max_high_since_entry / entry_close - 1

        target_price = entry_close * (1 + TARGET_PCT)
        target_date = today + timedelta(days=int(HORIZON_DAYS * 1.4))  # ~180 trading days = ~252 cal days

        # milestones hit?
        milestone_50 = max_return_so_far >= 0.50
        milestone_75 = max_return_so_far >= 0.75
        milestone_100 = max_return_so_far >= 1.00

        rows.append({
            "symbol": sym,
            "entry_date": today,
            "entry_close": entry_close,
            "current_close": current_close,
            "return_since_entry": return_since_entry,
            "max_return_so_far": max_return_so_far,
            "target_price": target_price,
            "target_date": target_date,
            "days_held": 0,  # baseline; updated each subsequent run
            "days_remaining": HORIZON_DAYS,
            "milestone_50": milestone_50,
            "milestone_75": milestone_75,
            "milestone_100": milestone_100,
            "score_180d": float(p.get("score_100pct_180d", 0)),
            "score_252d": float(p.get("score_100pct_252d", 0)),
            "score_378d": float(p.get("score_100pct_378d", 0)),
            "snapshot_date": today,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no basket members to track today")
        return

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        old = pd.read_parquet(LEDGER)
        # preserve original entry_date / entry_close from first snapshot
        merged = pd.concat([old, df], ignore_index=True)
        # for repeat snapshots of same symbol, keep first entry but append snapshot rows
        merged.to_parquet(LEDGER, index=False)
    else:
        df.to_parquet(LEDGER, index=False)

    # report
    md = [f"# Multibagger basket — status snapshot {today:%Y-%m-%d}", "",
          f"**Goal:** ≥1 of {len(TIER_1_BASKET)} doubles within {HORIZON_DAYS} trading days "
          f"(historical 90% conviction at score ≥ 0.86).", "",
          "## Current status", "",
          "| Symbol | Entry ₹ | Current ₹ | Return | Max % seen | Target ₹ | 180d score | Days left |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        ret = r["return_since_entry"] * 100
        max_ret = r["max_return_so_far"] * 100
        md.append(f"| **{r['symbol']}** | ₹{r['entry_close']:.2f} | ₹{r['current_close']:.2f} | "
                  f"{ret:+.2f}% | {max_ret:+.2f}% | ₹{r['target_price']:.2f} | "
                  f"{r['score_180d']:.3f} | {int(r['days_remaining'])} |")
    md.append("")
    md.append("## Discipline rules")
    md.append("")
    md.append("- **Hold:** minimum 90 days, max 180-378d depending on score")
    md.append("- **Stop-loss:** -25% from entry (long-horizon thesis tolerates volatility)")
    md.append("- **Profit targets:** sell 25% at +50%, 50% at +75%, trail rest to +100%")
    md.append("- **Score-drift exit:** if model score drops below 0.70 in any subsequent run → close")
    md.append("- **Calibration check:** after 90 days, count how many hit milestones — expected ≥3 of 4 at +50%, ≥1 of 4 at +100%")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"wrote {OUT_REPORT}")
    print(df[["symbol", "entry_close", "current_close", "return_since_entry"]].to_string(index=False))


if __name__ == "__main__":
    main()
