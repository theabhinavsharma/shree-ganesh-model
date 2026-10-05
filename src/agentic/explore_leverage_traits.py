"""What separates +50% hitters from the rest on leverage, pledges and P&L? (2026-10-04, EXPLORATION for Abhinav: "is there a
pattern in leverage or some filter that shows up in 50% hitters vs not hitters?"; no rule changed). Companion to
explore_hitter_traits.py (68 anatomy-row traits inside the G1 pool, 2019-22, same model rank), which has no leverage or
pledge traits.

Universes: V3 picks (phase 0, weekly 2019+, decision = list date, entry = next open) at 6 and 12 months; big-order trades
(>= 15% of revenue, 2019+, decision = the session before entry) at 12 and 24 months.
Traits at the decision date (point in time; today's ticker for lookups via nse_symbols):
  interest cover (trailing 4 quarters, as test_v3_leverage), interest / sales (trailing 4 quarters), sales growth (trailing
  4 quarters vs the 4 before), profit margin (trailing PAT / sales), credit event in 180 days, pledges (NSE Reg 31): promoter
  shares newly pledged in the prior 90 days as % of capital, released in 90 days, any invocation in 180 days.
Outcome: touched +50% within the hold (hit), ended <= -30% (loser). Per trait: fixed buckets; hit / loser rate [n]; the
spread (best minus worst bucket hit rate, buckets with n >= 30) ranks the traits. Univariate: traits overlap.
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

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
V3 = v3_rule.picks(X["F"], [d for d in sp.weekly_grid(cal, 0) if d >= pd.Timestamp("2019-01-01")], imap, P, ctx)
T = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
T = T[[cal[i] >= pd.Timestamp("2019-01-01") for i in T["i0"]]]
need = {nse_symbols.now(s) for v in V3.values() for s in v} | {nse_symbols.now(s) for s in T["symbol"]}

Q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "filing_dt", "basis", "source", "net_sales", "pat", "finance_cost"])
Q["quarter_end"] = pd.to_datetime(Q["quarter_end"]); Q["filing_dt"] = pd.to_datetime(Q["filing_dt"], errors="coerce")
fx = np.where(Q["source"] == "xbrl", 1e-5, 1.0)
for c in ("net_sales", "pat", "finance_cost"):
    Q[c] = Q[c] * fx
Q = Q[Q["symbol"].isin(need)].dropna(subset=["filing_dt"]).sort_values("filing_dt")
PL = {}
for s, g in Q.groupby("symbol"):
    ts, vals = [], []
    for t in g["filing_dt"].unique():
        h = g[g["filing_dt"] <= t]; v = {}
        for b in ("con", "sa"):
            x = h[h["basis"] == b].drop_duplicates("quarter_end", keep="first").sort_values("quarter_end").tail(8)
            l4 = x.tail(4)
            if len(l4) == 4 and (l4["quarter_end"].iloc[-1] - l4["quarter_end"].iloc[0]).days in range(260, 285) and l4["net_sales"].notna().all():
                s4 = l4["net_sales"].sum()
                v = dict(margin=l4["pat"].sum() / s4 if s4 > 0 and l4["pat"].notna().all() else np.nan,
                         int_sales=l4["finance_cost"].sum() / s4 if s4 > 0 and l4["finance_cost"].notna().all() else np.nan)
                if len(x) == 8 and (x["quarter_end"].iloc[-1] - x["quarter_end"].iloc[0]).days in range(620, 650) and x["net_sales"].head(4).notna().all():
                    p4 = x["net_sales"].head(4).sum(); v["growth"] = s4 / p4 - 1 if p4 > 0 else np.nan
                break
        ts.append(t); vals.append(v)
    PL[s] = (np.array(ts, dtype="datetime64[ns]"), vals)
IC = tl.interest_cover(need)
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket"])
L = L[L["bucket"].isin(["rating_down", "default_insolvency"])]
CR = {s: np.sort(pd.to_datetime(g["filed_at"]).values) for s, g in L.groupby("symbol")}
E = pd.read_parquet(ROOT / "data/derived/pledge_events.parquet")
E = E[E["encumbrance"].str.contains("pledge", case=False, na=False) | (E["event"] == "Invocation")].copy()
E["pct"] = E["pct_of_capital"].clip(0, 100)
ES = {s: g.sort_values("broadcast_dt") for s, g in E.groupby("symbol_now")}


def asof(a, d):
    if a is None:
        return None
    i = np.searchsorted(a[0], np.datetime64(pd.Timestamp(d)), side="left") - 1
    return a[1][i] if i >= 0 else None


def traits(s, d):
    n = nse_symbols.now(s); d = pd.Timestamp(d)
    pl = asof(PL.get(n), d) or {}
    ic = asof(IC.get(n), d)
    crs = [x for x in (CR.get(s), CR.get(n)) if x is not None]
    cr = bool(len(crs) and ((np.concatenate(crs) <= np.datetime64(d)) & (np.concatenate(crs) > np.datetime64(d - pd.Timedelta(days=180)))).any())
    e = ES.get(n); new = rel = 0.0; inv = False
    if e is not None:
        w = e[(e["broadcast_dt"] < d) & (e["broadcast_dt"] >= d - pd.Timedelta(days=90))]
        new = float(w.loc[w["event"] == "Creation", "pct"].sum()); rel = float(w.loc[w["event"] == "Release", "pct"].sum())
        inv = bool(((e["event"] == "Invocation") & (e["broadcast_dt"] < d) & (e["broadcast_dt"] >= d - pd.Timedelta(days=180))).any())
    icv = np.nan
    if ic and ic[0] in ("ok", "weak"):
        icv = ic[1]
    elif ic and ic[0] == "none":
        icv = np.inf
    return dict(ic=icv, int_sales=pl.get("int_sales", np.nan), growth=pl.get("growth", np.nan), margin=pl.get("margin", np.nan),
                credit=cr, pledge_new=new, pledge_rel=rel, invocation=inv)


PX = rp.load_panel(["open", "high", "close"]); Ow = rp.wide(PX, "open", cal); cols = {s: i for i, s in enumerate(Ow.columns)}
O, Hh, C = Ow.to_numpy(), rp.wide(PX, "high", cal).to_numpy(), rp.wide(PX, "close", cal).ffill(limit=300).to_numpy(); del PX
rows = [dict(u="V3", s=s, d=d, i0=cal.get_loc(d) + 1) for d, names in V3.items() for s in names]
rows += [dict(u="BIG", s=r.symbol, d=cal[r.i0 - 1], i0=int(r.i0)) for r in T.itertuples()]
R = pd.DataFrame(rows)
R = pd.concat([R, pd.DataFrame([traits(s, d) for s, d in zip(R["s"], R["d"])])], axis=1)
for n in (126, 252, 504):
    hit, end = [], []
    for r in R.itertuples():
        c = cols.get(r.s)
        if c is None or r.i0 + n > len(cal) or not np.isfinite(O[r.i0, c]) or O[r.i0, c] <= 0:
            hit.append(np.nan); end.append(np.nan); continue
        e = O[r.i0, c]; hit.append(float(np.nanmax(Hh[r.i0:r.i0 + n, c]) >= 1.5 * e)); end.append(C[r.i0 + n - 1, c] / e - 1)
    R[f"hit{n}"], R[f"ret{n}"] = hit, end


def bucket(col, x):
    if col in ("credit", "invocation"):
        return x.map({True: "yes", False: "no"})
    if col == "ic":
        lab = pd.cut(x.replace(np.inf, np.nan), [-np.inf, 1.5, 3, 10, np.inf], labels=["<1.5x", "1.5-3x", "3-10x", ">10x"]).astype(str)
        return pd.Series(np.where(np.isinf(x), "no interest", lab), index=x.index).replace("nan", "unknown")
    edges, labels = {"int_sales": ([-np.inf, 0.005, 0.02, 0.05, np.inf], ["<0.5%", "0.5-2%", "2-5%", ">5%"]),
                     "growth": ([-np.inf, 0, 0.15, 0.3, 0.6, np.inf], ["falling", "0-15%", "15-30%", "30-60%", ">60%"]),
                     "margin": ([-np.inf, 0, 0.05, 0.1, 0.2, np.inf], ["loss", "0-5%", "5-10%", "10-20%", ">20%"]),
                     "pledge_new": ([-np.inf, 0, 1, np.inf], ["none", "<1% of capital", ">=1%"]),
                     "pledge_rel": ([-np.inf, 0, 1, np.inf], ["none", "<1%", ">=1%"])}[col]
    return pd.cut(x, edges, labels=labels).astype(str).replace("nan", "unknown")


out = {}
for u, holds in (("V3", (126, 252)), ("BIG", (252, 504))):
    for n in holds:
        x = R[(R.u == u) & R[f"hit{n}"].notna()]
        print(f"\n=== {'V3 picks' if u == 'V3' else 'big orders >= 15%'} · {n // 21} months · {len(x)} picks · base: hit +50% {x[f'hit{n}'].mean():.0%}, "
              f"ended <= -30% {(x[f'ret{n}'] <= -0.3).mean():.0%} ===   (each bucket: hit / loser [n])")
        rank = []
        for col in ("ic", "int_sales", "growth", "margin", "credit", "pledge_new", "pledge_rel", "invocation"):
            g = x.groupby(bucket(col, x[col])).agg(n=(f"hit{n}", "size"), hit=(f"hit{n}", "mean"), lost=(f"ret{n}", lambda v: (v <= -0.3).mean()))
            gg = g[(g.n >= 30) & (g.index != "unknown")]
            rank.append((float(gg["hit"].max() - gg["hit"].min()) if len(gg) >= 2 else 0.0, col, g))
            out[f"{u}|{n}|{col}"] = {k: dict(n=int(r.n), hit=round(float(r.hit), 3), lost=round(float(r.lost), 3)) for k, r in g.iterrows()}
        for spread, col, g in sorted(rank, key=lambda z: -z[0]):
            print(f" {col:11s} spread {spread:4.0%} · " + " · ".join(f"{k} {r.hit:.0%}/{r.lost:.0%} [{int(r.n)}]" for k, r in g.iterrows()))
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-04-leverage-traits-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_leverage_traits.py", results=out), default=str) + "\n")
