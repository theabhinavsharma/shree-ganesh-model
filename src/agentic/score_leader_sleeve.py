"""LEADER SLEEVE forward scorer — replays every immutable screen day by day.

Contract (the registered strategy cell, sim_leader_sleeve.py 2026-09-08):
  entry  = first session OPEN after the screen's data_through (next-open)
  hold   = 126 trading days, exit at that day's CLOSE (time exit, no stops —
           trailing stops were killed: they sell the -14..-19% median trough)
  weight = equal across the screen's names; cost 0.5% round trip
Prices are the adjusted panel (open/high/low/close adjusted to present), so a later
corporate action rescales a whole path and leaves returns unchanged.

Per name: entry, last, return (gross + net), peak, trough, days held, OPEN/CLOSED.
Per cohort: EW return, top-4-by-RTW return, winners/n, P(touched 2x), P(touched +50%),
worst trough. One line per (screen, scored_through) appended to
logs/leader_sleeve/outcomes.jsonl — re-running on the same session is a no-op.

PAPER ONLY until outcomes.jsonl holds >= 13 weekly cohorts (screens' paper_rule).
Run: daily from sgm_daily.sh after the 15d report. Pandas only.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
DIR = ROOT / "logs/leader_sleeve"
OUT = DIR / "outcomes.jsonl"
HOLD_TD, COST, TOP_K, PAPER_MIN_COHORTS = 126, 0.5, 4, 13


def _r(v, n=2):
    return None if v is None or pd.isna(v) else round(float(v), n)


def score_screen(sc: dict, px: pd.DataFrame) -> dict:
    thru = pd.Timestamp(sc["data_through"])
    rows = []
    for n in sc["names"]:
        g = px[(px["symbol"] == n["symbol"]) & (px["trade_date"] > thru)].head(HOLD_TD)
        base = dict(symbol=n["symbol"], rank=n["rank"], rtw=n["rtw"], industry=n["industry"])
        if g.empty or pd.isna(g["open"].iloc[0]):
            rows.append(dict(base, status="PENDING", days=0))
            continue
        e = float(g["open"].iloc[0]); last = float(g["close"].iloc[-1])
        gross = (last / e - 1) * 100
        rows.append(dict(base, entry_date=str(g["trade_date"].iloc[0].date()), entry=_r(e),
                         last_date=str(g["trade_date"].iloc[-1].date()), last=_r(last),
                         ret_gross=_r(gross), ret_net=_r(gross - COST),
                         peak=_r((g["high"].max() / e - 1) * 100), trough=_r((g["low"].min() / e - 1) * 100),
                         days=len(g), status="CLOSED" if len(g) >= HOLD_TD else "OPEN"))
    live = [r for r in rows if r["status"] != "PENDING"]
    top = [r for r in sorted(live, key=lambda r: r["rank"])[:TOP_K]]
    mean = lambda xs: _r(sum(xs) / len(xs)) if xs else None
    return dict(
        ts=datetime.now().isoformat(timespec="seconds"), screen_id=sc["screen_id"],
        data_through=sc["data_through"], scored_through=max((r["last_date"] for r in live), default=None),
        n=len(rows), n_entered=len(live), days_held=max((r["days"] for r in live), default=0),
        status="CLOSED" if live and all(r["status"] == "CLOSED" for r in live) else ("PENDING" if not live else "OPEN"),
        ew_gross=mean([r["ret_gross"] for r in live]), ew_net=mean([r["ret_net"] for r in live]),
        top4_gross=mean([r["ret_gross"] for r in top]), top4_net=mean([r["ret_net"] for r in top]),
        winners=sum(r["ret_net"] > 0 for r in live),
        p_2x=_r(sum(r["peak"] >= 100 for r in live) / len(live), 3) if live else None,
        p_50=_r(sum(r["peak"] >= 50 for r in live) / len(live), 3) if live else None,
        worst_trough=min((r["trough"] for r in live), default=None),
        names=rows)


def logged_keys() -> set:
    if not OUT.exists():
        return set()
    return {(d["screen_id"], d["scored_through"]) for d in
            (json.loads(l) for l in OUT.read_text().splitlines() if l.strip())}


def print_cohort(c: dict) -> None:
    print(f"\n── cohort {c['screen_id']} (data {c['data_through']}, next-open entry) · scored through "
          f"{c['scored_through']} · day {c['days_held']}/{HOLD_TD} · {c['status']}")
    print(f"{'SYM':<12}{'RTW':>4} {'entry':>10} {'last':>10} {'gross':>8} {'net':>8} {'peak':>8} {'trough':>8} {'days':>5}  status")
    for r in c["names"]:
        if r["status"] == "PENDING":
            print(f"{r['symbol']:<12}{r['rtw']:>4}  (no session since data_through — entry pending)")
            continue
        print(f"{r['symbol']:<12}{r['rtw']:>4} {r['entry']:>10.2f} {r['last']:>10.2f} {r['ret_gross']:>+7.1f}% "
              f"{r['ret_net']:>+7.1f}% {r['peak']:>+7.1f}% {r['trough']:>+7.1f}% {r['days']:>5}  {r['status']}")
    if c["n_entered"]:
        print(f"EW-{c['n_entered']} {c['ew_gross']:+.2f}% gross / {c['ew_net']:+.2f}% net · top-{TOP_K} RTW "
              f"{c['top4_gross']:+.2f}% / {c['top4_net']:+.2f}% · winners {c['winners']}/{c['n_entered']} · "
              f"P(2x) {c['p_2x']:.0%} · P(+50%) {c['p_50']:.0%} · worst trough {c['worst_trough']:+.1f}%")


def main() -> None:
    screens = [json.loads(p.read_text()) for p in sorted(DIR.glob("screen_*.json"))]
    if not screens:
        raise SystemExit("no logs/leader_sleeve/screen_*.json — run screen_theme_leaders.py first")
    syms = sorted({n["symbol"] for s in screens for n in s["names"]})
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "open", "high", "low", "close"],
                         filters=[("symbol", "in", syms),
                                  ("trade_date", ">", pd.Timestamp(min(s["data_through"] for s in screens)))])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.sort_values(["symbol", "trade_date"])
    seen, appended = logged_keys(), 0
    with OUT.open("a") as fh:
        for sc in screens:
            c = score_screen(sc, px)
            print_cohort(c)
            if c["scored_through"] and (c["screen_id"], c["scored_through"]) not in seen:
                fh.write(json.dumps(c) + "\n"); appended += 1
    n_coh = len(screens)
    print(f"\ncohorts on record {n_coh} · sizing {'PAPER' if n_coh < PAPER_MIN_COHORTS else 'ELIGIBLE FOR REVIEW'}"
          f" (rule: >= {PAPER_MIN_COHORTS} weekly cohorts) · appended {appended} line(s) to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
