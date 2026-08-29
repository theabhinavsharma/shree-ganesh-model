"""PAPER SLEEVE — 100td-hold momentum/ignition sleeve, paper-tracked weekly.

Shipped 2026-08-29 after the pre-registered walk-forward A/B
(logs/backtest_sleeve_walkforward.log):
  UNION (hc top-10 P(+20%/30d) ∪ fresh-IPO ignition) × HOLD-100td:
    disc<=2022 +4.59%/trade net · conf>=2023 +9.57%/trade net · 75% years positive
  Trailing stops (-15%/-18% from peak) were KILLED by the same test — they halve-to-
  quarter returns because median troughs run -16..-22% before positions pay
  (returns are back-loaded; a trail sells the bottom). Exit is TIME, not price.

PAPER ONLY: no capital until the forward record earns it. Selection:
  • hc top-10 by score_20pct_30d_cal, investable (ADV>=5cr, close>50)
  • C6 ignition events in the last 5 sessions (listed<=252td, ret1d>=8%, vol>=3x)
  • dedup: skip symbols with an open position or an entry in the last 60td
Entry: first open after signal date (pending-fill on next run). Exit: close of the
100th trading day. State: logs/paper_sleeve_positions.jsonl (full position log),
summary appended to logs/paper_sleeve.jsonl.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
POS = ROOT / "logs/paper_sleeve_positions.jsonl"
SUM = ROOT / "logs/paper_sleeve.jsonl"
HOLD_TD = 100


def load_positions() -> list[dict]:
    if not POS.exists():
        return []
    return [json.loads(l) for l in POS.read_text().splitlines() if l.strip()]


def main() -> None:
    px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
                         columns=["symbol", "trade_date", "open", "high", "low", "close",
                                  "return_1d", "volume_vs_20d", "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    today = px["trade_date"].max()
    sf = {s: g.reset_index(drop=True) for s, g in px.groupby("symbol")}

    positions = load_positions()
    open_syms = {p["symbol"] for p in positions if p["status"] in ("OPEN", "PENDING_FILL")}
    recent = {p["symbol"] for p in positions
              if (today - pd.Timestamp(p["signal_date"])).days <= 90}

    # ---- 1. fill pending entries at first open after signal ----
    for p in positions:
        if p["status"] != "PENDING_FILL":
            continue
        g = sf.get(p["symbol"])
        fut = g[g["trade_date"] > pd.Timestamp(p["signal_date"])] if g is not None else pd.DataFrame()
        if len(fut) and pd.notna(fut.iloc[0]["open"]):
            p["entry_price"] = float(fut.iloc[0]["open"])
            p["entry_date"] = str(fut.iloc[0]["trade_date"].date())
            p["status"] = "OPEN"

    # ---- 2. mark-to-market / close at 100td ----
    n_closed = 0
    for p in positions:
        if p["status"] != "OPEN":
            continue
        g = sf[p["symbol"]]
        held = g[g["trade_date"] >= pd.Timestamp(p["entry_date"])]
        p["days_held"] = int(len(held))
        ep = p["entry_price"]
        p["peak_ret_pct"] = round((held["high"].max() / ep - 1) * 100, 2)
        p["trough_pct"] = round((held["low"].min() / ep - 1) * 100, 2)
        p["mtm_ret_pct"] = round((float(held.iloc[-1]["close"]) / ep - 1) * 100, 2)
        p["touched_25"] = bool(p["peak_ret_pct"] >= 25)
        p["touched_2x"] = bool(p["peak_ret_pct"] >= 100)
        if len(held) >= HOLD_TD:
            exit_close = float(held.iloc[HOLD_TD - 1]["close"])
            p["status"] = "CLOSED"
            p["exit_date"] = str(held.iloc[HOLD_TD - 1]["trade_date"].date())
            p["exit_ret_pct"] = round((exit_close / ep - 1) * 100 - 0.30, 2)
            n_closed += 1

    # ---- 3. new signals this week ----
    new = []
    hc = pd.read_parquet(ROOT / "data/derived/high_conviction_predictions.parquet")
    snap = px[px["trade_date"] == today].set_index("symbol")
    inv = set(snap[(snap["avg_traded_value_20d"] / 1e7 >= 5) & (snap["close"] > 50)].index)
    if "score_20pct_30d_cal" in hc.columns:
        top = hc[hc["symbol"].isin(inv)].nlargest(10, "score_20pct_30d_cal")
        for _, r in top.iterrows():
            new.append((r["symbol"], "HC10", float(r["score_20pct_30d_cal"])))
    week = px[px["trade_date"] > today - pd.Timedelta(days=7)]
    age = px.groupby("symbol").cumcount()
    px["_age"] = age
    ign = week.merge(px[["symbol", "trade_date", "_age"]], on=["symbol", "trade_date"])
    ign = ign[(ign["_age"] <= 252) & (ign["return_1d"] >= 0.08) & (ign["volume_vs_20d"] >= 3)
              & ign["symbol"].isin(inv)]
    for s in ign["symbol"].unique():
        new.append((s, "C6_IPO_IGNITION", None))

    n_new = 0
    for sym, src, score in new:
        if sym in open_syms or sym in recent:
            continue
        positions.append(dict(symbol=sym, source=src, score=score, status="PENDING_FILL",
                              signal_date=str(today.date()), entry_price=None, entry_date=None))
        open_syms.add(sym); recent.add(sym); n_new += 1

    POS.write_text("\n".join(json.dumps(p) for p in positions) + "\n")
    closed = [p for p in positions if p["status"] == "CLOSED"]
    openp = [p for p in positions if p["status"] == "OPEN"]
    summary = dict(run_date=str(date.today()), data_through=str(today.date()),
                   new_signals=n_new, closed_this_run=n_closed,
                   open=len(openp), closed_total=len(closed),
                   closed_mean_ret=round(sum(p["exit_ret_pct"] for p in closed) / len(closed), 2) if closed else None,
                   closed_touch25=round(sum(p["touched_25"] for p in closed) / len(closed) * 100, 1) if closed else None,
                   open_mtm_mean=round(sum(p["mtm_ret_pct"] for p in openp) / len(openp), 2) if openp else None)
    with SUM.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print("PAPER SLEEVE:", json.dumps(summary))
    for p in sorted(openp, key=lambda x: -(x.get("mtm_ret_pct") or 0))[:15]:
        print(f"  OPEN {p['symbol']:12s} {p['source']:16s} entry {p['entry_date']} @ {p['entry_price']:.2f}  mtm {p['mtm_ret_pct']:+.1f}%  peak {p['peak_ret_pct']:+.1f}%  held {p['days_held']}td")


if __name__ == "__main__":
    main()
