"""Does the month you buy a big-order stock matter? (2026-10-04, EXPLORATION for Abhinav's question; no rule changed)

Trades: logs/leader_sleeve/big_order_hold/reference_trades.parquet (the registered test's trade list: cleaned order wins
>= 15% of trailing revenue, non-alarming, profitable, liquid, one per company per 30 days), bought at the open of the first
session after the filing, 2019-01 .. the last month whose hold has finished. 0.5% round-trip cost.
For each calendar month of purchase: trades, average and median return, share up, the typical stock (median of every stock
priced on the entry day, same dates), excess = average trade - average typical stock, and in how many years that month's
buys made money / beat the typical stock.
Luck check: shuffle the buy-month labels within each year 5,000 times; how often is the spread between the best and worst
month's average excess at least as wide as the real one? (A high share = the month pattern looks like chance.)
SGM_FULL=1 (user: "sampling kyu kara, full karo"): nothing was sampled before; this widens the data instead —
  A  the same >= 15% trade list from 2016 (2016-18 revenue known for only 17-50% of orders: thin years)
  B  EVERY cleaned order win of any size (no revenue or profit needed, so 2016+ is complete): order_amounts cat 'order',
     amount confidence high/medium, clarification / news-verification / amount-disagree / unit-guessed filings dropped, no
     bad filing in the prior 90 days, then-traded price >= Rs 20 and >= Rs 1 cr traded a day, one trade per company per 30 days
"""
import os
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402

COST = 0.005
cal = sp.load(None)["cal"]
PX = rp.load_panel(["open", "close"]); Ow = rp.wide(PX, "open", cal); Cw = rp.wide(PX, "close", cal).ffill(limit=300); del PX
On, Cn = Ow.to_numpy(), Cw.to_numpy(); col = {s: i for i, s in enumerate(Cw.columns)}
FULL = os.environ.get("SGM_FULL") == "1"
T = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
T["buy"] = [cal[i] for i in T["i0"]]; T = T[T.buy >= ("2016-01-01" if FULL else "2019-01-01")]
if FULL:   # B: every cleaned order win of any size
    O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
    O = O[(O["cat_current"] == "order") & O["amount_confidence"].isin(["high", "medium"])].copy()
    O["ts"] = pd.to_datetime(O["ts"]); O["act"] = pd.to_datetime(O["d_actionable"])
    O = O[~O["headline"].fillna("").str.contains(r"^(Clarification|News Verification|Reply to Clarification)|sought clarification|news item|media report", case=False, regex=True)]
    O = O[~(O["headline_fulltext_agree"].astype(str) == "False")]; O = O[~O["amount_flags"].fillna("").str.contains("unit_inferred")]
    O["i0"] = [cal.searchsorted(a) if a <= cal[-1] else -1 for a in O["act"]]; O = O[O["i0"] > 0].sort_values("ts")
    L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "direction"])
    Lb = L[L["direction"] < 0].groupby("symbol")["filed_at"].apply(lambda s: np.sort(s.values)).to_dict()
    O = O[[not (len(a := Lb.get(s, np.array([], dtype="datetime64[ns]"))) and ((a < np.datetime64(t)) & (a >= np.datetime64(t - pd.Timedelta(days=90)))).any())
           for s, t in zip(O["symbol"], O["ts"])]]
    PR = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
    PR["raw_close"] = rp.raw_price(PR, "close"); PR["adv"] = PR["avg_traded_value_20d"] / 1e7
    prev = PR.set_index(["symbol", "trade_date"])[["raw_close", "adv"]]; del PR
    pp = prev.reindex(pd.MultiIndex.from_arrays([O["symbol"].values, [cal[i - 1] for i in O["i0"]]]))
    O = O[(pp["raw_close"].values >= 20) & (pp["adv"].values >= 1)]
    keep, last = [], {}
    for r in O.itertuples():
        if r.symbol in last and (r.ts - last[r.symbol]).days < 30:
            continue
        last[r.symbol] = r.ts; keep.append(r.Index)
    TB = O.loc[keep, ["symbol", "ts", "i0"]].copy(); TB["buy"] = [cal[i] for i in TB["i0"]]; TB["ratio"] = 1.0; TB = TB[TB.buy >= "2016-01-01"]
    print(f"B: every cleaned order win -> {len(TB)} trades 2016+ (A: >= 15% list -> {len(T)} trades 2016+)")
typ_cache = {}


def typical(i0, h):
    if (i0, h) not in typ_cache:
        e, x = On[i0], Cn[i0 + h - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
        typ_cache[(i0, h)] = float(np.median(x[ok] / e[ok] - 1))
    return typ_cache[(i0, h)]


rng = np.random.default_rng(7)
out = {}
RUNS = ([("A >= 15%", T, 0.15, 252, "12m"), ("A >= 15%", T, 0.15, 504, "24m"), ("B every order", TB, 0.0, 252, "12m"), ("B every order", TB, 0.0, 504, "24m")] if FULL
        else [("", T, 0.15, 252, "12m"), ("", T, 0.15, 504, "24m"), ("", T, 0.50, 252, "12m")])
for tag, TT, cut, h, hs in RUNS:
    rows = []
    for r in TT[TT.ratio >= cut].itertuples():
        if r.symbol not in col or r.i0 + h > len(cal):
            continue
        e, x = On[r.i0, col[r.symbol]], Cn[r.i0 + h - 1, col[r.symbol]]
        if np.isfinite(e) and e > 0 and np.isfinite(x):
            rows.append(dict(year=r.buy.year, month=r.buy.month, ret=x / e - 1 - COST, typ=typical(r.i0, h)))
    R = pd.DataFrame(rows); R["exc"] = R.ret - R.typ
    yrs = f"{R.year.min()}-{R.year.max()}"
    print(f"\n=== {tag + ' · ' if tag else ''}orders >= {cut:.0%} of revenue · {hs} hold · bought {yrs} · {len(R)} trades ===")
    print(" month  trades   avg  median   up  typical  excess   years up / with buys   years beating typical")
    ym = R.groupby(["month", "year"]).agg(ret=("ret", "mean"), exc=("exc", "mean"))
    res = {}
    for m, g in R.groupby("month"):
        y = ym.loc[m]
        print(f"  {pd.Timestamp(2000, m, 1):%b}  {len(g):5d}  {g.ret.mean():+5.0%}  {g.ret.median():+5.0%}  {(g.ret > 0).mean():4.0%}  {g.typ.mean():+6.0%}  {g.exc.mean():+6.0%}"
              f"        {int((y.ret > 0).sum())} / {len(y)}                  {int((y.exc > 0).sum())} / {len(y)}")
        res[int(m)] = dict(trades=len(g), avg=round(float(g.ret.mean()), 4), median=round(float(g.ret.median()), 4), excess=round(float(g.exc.mean()), 4),
                           years_up=int((y.ret > 0).sum()), years_beat=int((y.exc > 0).sum()), years=len(y))
    mm = R.groupby("month").exc.mean(); spread = mm.max() - mm.min()
    sh = []
    for _ in range(5000):
        lab = R.groupby("year").month.transform(lambda s: rng.permutation(s.values))
        mm_ = R.exc.groupby(lab).mean(); sh.append(mm_.max() - mm_.min())
    p = float((np.array(sh) >= spread).mean())
    print(f" luck check: best-minus-worst month excess {spread:+.0%}; shuffled months give a spread at least that wide {p:.0%} of the time")
    out[f"{tag}>={cut:.2f} {hs}"] = dict(by_month=res, spread=round(float(spread), 4), p_shuffle=round(p, 3), trades=len(R), years=yrs)
    if hs == "12m" and (FULL or cut == 0.15):
        print(f"\n average 12m return by buy month (rows) and year (columns){' · ' + tag if tag else ', >= 15% orders'}; blank = no buys")
        piv = R.pivot_table(index="month", columns="year", values="ret", aggfunc="mean")
        cnt = R.pivot_table(index="month", columns="year", values="ret", aggfunc="size")
        print("        " + "".join(f"{y:>12d}" for y in piv.columns))
        for m in piv.index:
            print(f"  {pd.Timestamp(2000, m, 1):%b}  " + "".join(f"{'':>12s}" if pd.isna(piv.loc[m, y]) else f"{piv.loc[m, y]:+6.0%} ({int(cnt.loc[m, y]):2d})" for y in piv.columns))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-big-order-buy-month" + ("-full" if FULL else "") + "-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_big_order_month.py", results=out)) + "\n")
