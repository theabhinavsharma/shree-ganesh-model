"""Leverage flags vs outcome at 6 / 9 / 12 / 18 / 24 months (2026-10-04, EXPLORATION for Abhinav; no rule changed).

Same flags as EXP-2026-10-04-v3-leverage (test_v3_leverage.py): weak interest cover (trailing 4 quarters filed before the
decision date: (profit before tax + finance cost) / finance cost < 1.5x, or a loss before interest while paying interest)
and a credit event (rating_down / default_insolvency filing in the prior 180 days). Tickers mapped to today's for lookups.
Universes: V3 picks (phase 0, weekly from 2019; decision = the list date, entry = next open) and the big-order trades
(logs/leader_sleeve/big_order_hold/reference_trades.parquet, orders >= 15% of revenue, 2019+; >= 50% shown separately;
decision = the session before entry). For each hold: average and median return (entry open -> close at the hold's last
session, 0.5% cost for big orders as in their test, none for V3 as in its stats), share ending <= -30%, share touching
+50% at any point within the hold. A trade enters a hold's row only if that hold has ended.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic"))
import nse_symbols  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402
import sim_screen_rank_exit as sre  # noqa: E402
import test_v3_leverage as tl  # noqa: E402
import v3_rule  # noqa: E402

HOLDS = {"6m": 126, "9m": 189, "12m": 252, "18m": 378, "24m": 504}
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
wk = [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp("2019-01-01")]
V3 = v3_rule.picks(X["F"], wk, imap, P, ctx)
T = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
T = T[[cal[i] >= pd.Timestamp("2019-01-01") for i in T["i0"]]]
syms = {nse_symbols.now(s) for v in V3.values() for s in v} | {nse_symbols.now(s) for s in T["symbol"]}
IC = tl.interest_cover(syms)
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket"])
L = L[L["bucket"].isin(["rating_down", "default_insolvency"])]
CR = {s: np.sort(pd.to_datetime(g["filed_at"]).values) for s, g in L.groupby("symbol")}


def ic_at(s, d):
    a = IC.get(nse_symbols.now(s))
    if a is None:
        return "na"
    i = np.searchsorted(a[0], np.datetime64(pd.Timestamp(d)), side="left") - 1
    return a[1][i][0] if i >= 0 else "na"


def cr_at(s, d):
    a = [x for x in (CR.get(s), CR.get(nse_symbols.now(s))) if x is not None]
    if not a:
        return False
    a = np.concatenate(a); d = np.datetime64(pd.Timestamp(d))
    return bool(((a <= d) & (a > d - np.timedelta64(tl.CR_DAYS, "D"))).any())


PX = rp.load_panel(["open", "high", "close"]); O, Hh, C = (rp.wide(PX, c, cal).to_numpy() for c in ("open", "high", "close"))
cols = {s: i for i, s in enumerate(rp.wide(PX, "close", cal).columns)}; del PX
Cf = pd.DataFrame(C).ffill(limit=300).to_numpy()
rows = []
for d, names in V3.items():
    i0 = cal.get_loc(d) + 1
    for s in names:
        rows.append(dict(u="V3 picks", s=s, i0=i0, ic=ic_at(s, d), cr=cr_at(s, d), cost=0.0))
for r in T.itertuples():
    d = cal[r.i0 - 1]
    rows.append(dict(u="big orders >= 15%", s=r.symbol, i0=int(r.i0), ic=ic_at(r.symbol, d), cr=cr_at(r.symbol, d), cost=0.005))
    if r.ratio >= 0.5:
        rows.append(dict(u="big orders >= 50%", s=r.symbol, i0=int(r.i0), ic=ic_at(r.symbol, d), cr=cr_at(r.symbol, d), cost=0.005))
R = pd.DataFrame(rows)
R["group"] = np.select([R["cr"], R["ic"] == "weak", R["ic"] == "ok", R["ic"] == "none", R["ic"] == "na"],
                       ["credit event", "weak interest cover", "interest covered >= 1.5x", "no interest (debt-free)", "not computable"], "other")
out = {}
for h, n in HOLDS.items():
    ret, hit = [], []
    for r in R.itertuples():
        c = cols.get(r.s)
        if c is None or r.i0 + n > len(cal) or not np.isfinite(O[r.i0, c]) or O[r.i0, c] <= 0:
            ret.append(np.nan); hit.append(np.nan); continue
        e = O[r.i0, c]; x = Cf[r.i0 + n - 1, c]
        ret.append(x / e - 1 - r.cost if np.isfinite(x) else np.nan)
        hh = Hh[r.i0:r.i0 + n, c]; hit.append(float(np.nanmax(hh) >= 1.5 * e) if np.isfinite(hh).any() else np.nan)
    R[f"r{h}"], R[f"h{h}"] = ret, hit
order = ["weak interest cover", "credit event", "interest covered >= 1.5x", "no interest (debt-free)", "not computable"]
for u in ["V3 picks", "big orders >= 15%", "big orders >= 50%"]:
    print(f"\n=== {u} · avg (median) return · ended <= -30% · touched +50% · [n] ===")
    print(" group                        " + "".join(f"{h:>34s}" for h in HOLDS))
    for g in order:
        x = R[(R.u == u) & (R.group == g)]
        cells = []
        for h in HOLDS:
            v = x[f"r{h}"].dropna()
            cells.append(f"{v.mean():+.0%} ({v.median():+.0%}) {(v <= -0.3).mean():3.0%} {x.loc[v.index, f'h{h}'].mean():3.0%} [{len(v)}]" if len(v) else "-")
            out[f"{u}|{g}|{h}"] = dict(n=int(len(v)), avg=round(float(v.mean()), 4) if len(v) else None, median=round(float(v.median()), 4) if len(v) else None,
                                       lost30=round(float((v <= -0.3).mean()), 3) if len(v) else None)
        print(f" {g:28s} " + "".join(f"{c:>34s}" for c in cells))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-leverage-holds-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_leverage_holds.py", results=out)) + "\n")
