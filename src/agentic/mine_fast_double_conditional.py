"""CONDITIONAL FAST-DOUBLE MINER — pre-registered 2026-08-31.

The unconditional thrust cell died over 10 years (2.5-3.5% P(2x/100td), no lift).
This run tests whether CONDITIONING the thrust on the newly-acquired event corpus
creates a genuine fast-double pocket. Cells registered before results seen:

Event = first day with ret1d>=+8% & vol>=3x per symbol per 60td, investable
        (ADV>=5cr, close>50). Family B: softer thrust +5%/1.5x for the same cells.
Overlays:
  NEAR-PRINT-POST : a results print (calendar, 2005-2026) within the prior 5 days
  NEAR-PRINT-PRE  : a print scheduled within the NEXT 10td (known from history of
                    that symbol's print cadence? NO — we use actual print dates,
                    so PRE means the print occurred within the next 10 days; this
                    is label-leaky as a live rule but measures whether anticipation
                    thrusts pay — the live version uses board-meeting intimations)
  ORDER-WIN       : an order-win filing (2016+) within ±5 days of the thrust
  SMALL-YOUNG     : ADV<=25cr AND listed <=3y at thrust
  ACCUMULATION    : dlv10>=1.1 AND vol_asym>=1.2 at thrust
  PROMOTER-BUY    : promoter PIT acquisition in prior 90d (2019+ partial window)
Conjunctions: POST×SMALL-YOUNG, POST×ACCUM, ORDER×SMALL-YOUNG, POST×SMALL-YOUNG×ACCUM.

Outcomes: P(2x/60td), P(2x/100td), P(+25%/30td), weekly-cohort mean end-100td, med trough.
Discovery <=2022 · confirm >=2023 · bar: conf lift >=1.5x on P(2x/100td) with n>=60.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
print("loading…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d",
             "volume_vs_20d", "delivery_pct", "avg_delivery_pct_20d", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol", sort=False)
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["age"] = g.cumcount()
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["dlv10"] = g["dlv"].transform(lambda s: s.rolling(10, min_periods=3).mean())
up = np.where(px["return_1d"] > 0, px["volume_vs_20d"], np.nan)
dn = np.where(px["return_1d"] < 0, px["volume_vs_20d"], np.nan)
px["_u"], px["_d"] = up, dn
px["up15"] = g["_u"].transform(lambda s: s.rolling(15, min_periods=3).mean())
px["dn15"] = g["_d"].transform(lambda s: s.rolling(15, min_periods=3).mean())
px["vasym"] = px["up15"] / px["dn15"]

print("forward outcomes…", flush=True)
def fwd(col, n, how):
    return g[col].transform(lambda s: getattr(s.shift(-1)[::-1].rolling(n, min_periods=min(20, n)), how)()[::-1])
px["hi60"] = fwd("high", 60, "max")
px["hi100"] = fwd("high", 100, "max")
px["hi30"] = fwd("high", 30, "max")
px["lo100"] = fwd("low", 100, "min")
px["cl100"] = g["close"].transform(lambda s: s.shift(-100))

print("event joins…", flush=True)
cal = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")
cal["d"] = pd.to_datetime(cal["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
cal.loc[cal["d"].isna(), "d"] = pd.to_datetime(cal.loc[cal["d"].isna(), "filing_date"],
                                               format="%d-%b-%Y %H:%M", errors="coerce")
cal2 = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
cal2["d"] = pd.to_datetime(cal2["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
prints = pd.concat([cal[["symbol", "d"]], cal2[["symbol", "d"]]]).dropna()
prints["d"] = prints["d"].dt.normalize()
print_map = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in prints.groupby("symbol")}

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "sort_date", "desc"])
ah["d"] = pd.to_datetime(ah["sort_date"], errors="coerce").dt.normalize()
ORDER_RE = re.compile(r"bagging|receipt of order|award.{0,12}(order|contract)|work order|letter of (intent|award)|purchase order|order (received|book|win)", re.I)
orders = ah[ah["desc"].fillna("").str.contains(ORDER_RE)].dropna(subset=["d"])
order_map = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in orders.groupby("symbol")}

pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet")
pit["d"] = pd.to_datetime(pit["date"], errors="coerce", dayfirst=True).dt.normalize()
symc = [c for c in pit.columns if "symbol" in c.lower()][0]
pcat = [c for c in pit.columns if "person" in c.lower() or "category" in c.lower()]
pit["dpct"] = pd.to_numeric(pit.get("afterAcqSharesPer"), errors="coerce") - pd.to_numeric(pit.get("befAcqSharesPer"), errors="coerce")
pb = pit[(pit["dpct"] > 0) & (pit[pcat[0]].astype(str).str.contains("promoter", case=False) if pcat else True)]
pb_map = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in pb.dropna(subset=["d"]).groupby(symc)}

def near(mp, sym, t, lo_days, hi_days):
    a = mp.get(sym)
    if a is None:
        return False
    lo = np.searchsorted(a, t + np.timedelta64(lo_days, "D"), side="left")
    hi = np.searchsorted(a, t + np.timedelta64(hi_days, "D"), side="right")
    return hi > lo

# thrust events, one per symbol per 60td
for tag, rq, vq in [("HARD", 0.08, 3.0), ("SOFT", 0.05, 1.5)]:
    ev = px[(px["return_1d"] >= rq) & (px["volume_vs_20d"] >= vq)
            & (px["adv"] >= 5) & (px["close"] > 50) & px["hi100"].notna()
            & (px["trade_date"] >= "2016-06-01")].copy()
    keep, last = [], {}
    for i, r in ev.sort_values("trade_date").iterrows():
        if r["symbol"] not in last or (r["trade_date"] - last[r["symbol"]]).days > 90:
            keep.append(i); last[r["symbol"]] = r["trade_date"]
    ev = ev.loc[keep].copy()
    t64 = ev["trade_date"].values.astype("datetime64[ns]")
    ev["post_print"] = [near(print_map, s, t, -5, 0) for s, t in zip(ev["symbol"], t64)]
    ev["pre_print"] = [near(print_map, s, t, 1, 14) for s, t in zip(ev["symbol"], t64)]
    ev["order_win"] = [near(order_map, s, t, -5, 5) for s, t in zip(ev["symbol"], t64)]
    ev["prom_buy"] = [near(pb_map, s, t, -90, 0) for s, t in zip(ev["symbol"], t64)]
    ev["small_young"] = (ev["adv"] <= 25) & (ev["age"] <= 756)
    ev["accum"] = (ev["dlv10"] >= 1.1) & (ev["vasym"] >= 1.2)
    ev["t2x60"] = ev["hi60"] / ev["close"] - 1 >= 1.0
    ev["t2x100"] = ev["hi100"] / ev["close"] - 1 >= 1.0
    ev["t25"] = ev["hi30"] / ev["close"] - 1 >= 0.25
    ev["end100"] = (ev["cl100"] / ev["close"] - 1) * 100
    ev["trough"] = (ev["lo100"] / ev["close"] - 1) * 100
    ev["year"] = ev["trade_date"].dt.year
    ev["week"] = ev["trade_date"].dt.to_period("W")

    CELLS = {
        "thrust alone (baseline)": pd.Series(True, index=ev.index),
        "× post-print (print in prior 5d)": ev["post_print"],
        "× pre-print (print lands next 1-14d)": ev["pre_print"],
        "× order-win ±5d": ev["order_win"],
        "× small-young": ev["small_young"],
        "× accumulation": ev["accum"],
        "× promoter-buy 90d (2019+)": ev["prom_buy"],
        "× post-print × small-young": ev["post_print"] & ev["small_young"],
        "× post-print × accum": ev["post_print"] & ev["accum"],
        "× order-win × small-young": ev["order_win"] & ev["small_young"],
        "× post-print × small-young × accum": ev["post_print"] & ev["small_young"] & ev["accum"],
    }
    print(f"\n════ {tag} thrust (ret>={rq*100:.0f}% vol>={vq}x) · events {len(ev):,} ════")
    print(f"{'CELL':<38s}| era      |    n | P(2x/60) | P(2x/100) | P(+25/30) | wk-end100 | medTr | yrs+")
    for name, m in CELLS.items():
        for era, em in [("disc<=22", ev["year"] <= 2022), ("conf>=23", ev["year"] >= 2023)]:
            s = ev[m & em]
            if len(s) < 40:
                print(f"{name:<38s}| {era} | {len(s):>4d} | too few"); continue
            wk = s.groupby("week")["end100"].mean()
            yr = s.groupby("year")["end100"].mean()
            print(f"{name:<38s}| {era} | {len(s):>4,} |   {s['t2x60'].mean()*100:5.1f}% |    {s['t2x100'].mean()*100:5.1f}% |    {s['t25'].mean()*100:5.1f}% |   {wk.mean():+6.2f}% | {s['trough'].median():+5.1f}% | {(yr>0).mean()*100:3.0f}%", flush=True)
print("\nCONDITIONAL MINER COMPLETE", flush=True)
