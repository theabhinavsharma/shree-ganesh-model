"""Does the month you buy a big-order stock matter? (2026-10-04, EXPLORATION for Abhinav's question; no rule changed)

Trades: logs/leader_sleeve/big_order_hold/reference_trades.parquet (the registered test's trade list: cleaned order wins
>= 15% of trailing revenue, non-alarming, profitable, liquid, one per company per 30 days), bought at the open of the first
session after the filing, 2019-01 .. the last month whose hold has finished. 0.5% round-trip cost.
For each calendar month of purchase: trades, average and median return, share up, the typical stock (median of every stock
priced on the entry day, same dates), excess = average trade - average typical stock, and in how many years that month's
buys made money / beat the typical stock.
Luck check: shuffle the buy-month labels within each year 5,000 times; how often is the spread between the best and worst
month's average excess at least as wide as the real one? (A high share = the month pattern looks like chance.)
"""
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
T = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
T["buy"] = [cal[i] for i in T["i0"]]; T = T[T.buy >= "2019-01-01"]
typ_cache = {}


def typical(i0, h):
    if (i0, h) not in typ_cache:
        e, x = On[i0], Cn[i0 + h - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
        typ_cache[(i0, h)] = float(np.median(x[ok] / e[ok] - 1))
    return typ_cache[(i0, h)]


rng = np.random.default_rng(7)
out = {}
for cut, h, hs in [(0.15, 252, "12m"), (0.15, 504, "24m"), (0.50, 252, "12m")]:
    rows = []
    for r in T[T.ratio >= cut].itertuples():
        if r.symbol not in col or r.i0 + h > len(cal):
            continue
        e, x = On[r.i0, col[r.symbol]], Cn[r.i0 + h - 1, col[r.symbol]]
        if np.isfinite(e) and e > 0 and np.isfinite(x):
            rows.append(dict(year=r.buy.year, month=r.buy.month, ret=x / e - 1 - COST, typ=typical(r.i0, h)))
    R = pd.DataFrame(rows); R["exc"] = R.ret - R.typ
    yrs = f"{R.year.min()}-{R.year.max()}"
    print(f"\n=== orders >= {cut:.0%} of revenue · {hs} hold · bought {yrs} · {len(R)} trades ===")
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
    out[f">={cut:.2f} {hs}"] = dict(by_month=res, spread=round(float(spread), 4), p_shuffle=round(p, 3), trades=len(R), years=yrs)
    if cut == 0.15 and hs == "12m":
        print("\n average 12m return by buy month (rows) and year (columns), >= 15% orders; blank = no buys")
        piv = R.pivot_table(index="month", columns="year", values="ret", aggfunc="mean")
        cnt = R.pivot_table(index="month", columns="year", values="ret", aggfunc="size")
        print("        " + "".join(f"{y:>12d}" for y in piv.columns))
        for m in piv.index:
            print(f"  {pd.Timestamp(2000, m, 1):%b}  " + "".join(f"{'':>12s}" if pd.isna(piv.loc[m, y]) else f"{piv.loc[m, y]:+6.0%} ({int(cnt.loc[m, y]):2d})" for y in piv.columns))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-big-order-buy-month-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_big_order_month.py", results=out)) + "\n")
