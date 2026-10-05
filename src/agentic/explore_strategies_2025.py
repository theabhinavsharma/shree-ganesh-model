"""Which explored strategy did best in calendar 2025 and 2026 so far? (2026-10-05, EXPLORATION for Abhinav; no rule changed).

Weekly-batch strategies run on the registered ladder engine (sim_screen_rank_exit.run_exit, rule E0 = time exit, ladder sized
to the hold), averaged over the 5 weekly entry phases, entries from 2019 so the 2025 book is the strategy's real book:
  V3 (live rule) · G1 (V3 before its financials / fading-theme filters) · V3 with the P&L-feature model (EXP-2026-09-30-v3-pnl
  rerun model_P; V3T = same 2019-trained model without P&L) · V3 minus weak interest cover / minus credit events
  (EXP-2026-10-04-v3-leverage) · V3 with promoter buyers / pledge releasers first (EXP-2026-10-04-v3-promoter-signals) ·
  V3 held 3 or 12 months.
Event strategies (big orders, reference_trades.parquet, >= 15% / >= 50% of revenue, hot = industry heat >= 0.70): equal money in
every open trade, cash when none, held 12 / 18 / 24 months, 0.5% cost per trade (as in their tests).
Market: equal-weight average daily return of every priced stock (daily returns clipped to [-50%, +100%]); flatters a little.
Reported: calendar 2025, 2026 to the last data day, and 2025-01-01 -> last day combined. In-sample for every V3 variant.
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
import test_industry_fundamentals as tif  # noqa: E402
import test_v3_leverage as tl  # noqa: E402
import test_v3_pnl as tp  # noqa: E402
import v3_rule  # noqa: E402

D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
START = pd.Timestamp("2019-01-01")
PM = tp.scores(ROOT / "logs/leader_sleeve/v3_pnl/rerun_datafill-20261004/model_P/bakeoff_preds.parquet")
PT = tp.scores(ROOT / "logs/leader_sleeve/v3_pnl/rerun_datafill-20261004/model_T/bakeoff_preds.parquet")

# leverage flags (as EXP-2026-10-04-v3-leverage) and promoter signals (as EXP-2026-10-04-v3-promoter-signals)
IC = None; CR = {}
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket"])
L = L[L["bucket"].isin(["rating_down", "default_insolvency"])]
CR = {s: np.sort(pd.to_datetime(g["filed_at"]).values) for s, g in L.groupby("symbol")}
A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"]); A["trade_date"] = pd.to_datetime(A["trade_date"])
PBD = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}
E = pd.read_parquet(ROOT / "data/derived/pledge_events.parquet")
E = E[(E["event"] == "Release") & E["encumbrance"].str.contains("pledge", case=False, na=False)]
ER = {s: (g["broadcast_dt"].values, g["pct_of_capital"].clip(0, 100).values) for s, g in E.sort_values("broadcast_dt").groupby("symbol_now")}


def ic_weak(s, d):
    a = IC.get(nse_symbols.now(s))
    if a is None:
        return False
    i = np.searchsorted(a[0], np.datetime64(pd.Timestamp(d)), side="left") - 1
    return i >= 0 and a[1][i][0] == "weak"


def credit(s, d):
    a = [x for x in (CR.get(s), CR.get(nse_symbols.now(s))) if x is not None]
    if not a:
        return False
    a = np.concatenate(a); d = np.datetime64(pd.Timestamp(d))
    return bool(((a <= d) & (a > d - np.timedelta64(180, "D"))).any())


def pb(s, d):
    a = PBD.get(s)
    if a is None:
        return False
    i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
    return bool(i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") and a[1][i] > 0)


def pr(s, d):
    a = ER.get(nse_symbols.now(s))
    if a is None:
        return False
    m = (a[0] < np.datetime64(d)) & (a[0] >= np.datetime64(d - pd.Timedelta(days=90)))
    return float(a[1][m].sum()) >= 1.0


def yearly(nav: pd.Series) -> dict:
    nav = nav / nav.iloc[0]; last = nav.index[-1]
    e24 = nav[nav.index <= "2024-12-31"].iloc[-1]; e25 = nav[nav.index <= "2025-12-31"].iloc[-1]
    return {"2025": (e25 / e24 - 1) * 100, "2026": (nav.iloc[-1] / e25 - 1) * 100, "both": (nav.iloc[-1] / e24 - 1) * 100,
            "2024": (e24 / nav[nav.index <= "2023-12-31"].iloc[-1] - 1) * 100, "to": str(last.date())}


rows = {}
phases = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
V3s = {o: v3_rule.picks(X["F"], phases[o], imap, P, ctx) for o in range(5)}
IC = tl.interest_cover({nse_symbols.now(s) for v in V3s.values() for names in v.values() for s in names})
G1s = {o: v3_rule.picks(X["F"], phases[o], imap, P, dict(ctx, keep=lambda s, d: True)) for o in range(5)}
V3P = {o: v3_rule.picks(X["F"], [d for d in phases[o] if d >= pd.Timestamp("2020-01-01")], imap, PM, ctx) for o in range(5)}
V3T = {o: v3_rule.picks(X["F"], [d for d in phases[o] if d >= pd.Timestamp("2020-01-01")], imap, PT, ctx) for o in range(5)}


def boosted(o, sig):
    top = sre.TOPN; sre.TOPN = 100000
    full = tif.select_elig(X["F"], phases[o], imap, P, ctx["G1"]); sre.TOPN = top
    out = {}
    for d in phases[o]:
        pool = full.get(d, []); first = [s for s in pool[:25] if sig(s, d)]
        pool = first + [s for s in pool if s not in set(first)]
        out[d] = [s for s in pool[:9] if ctx["keep"](s, d)]
    return out


strategies = {
    "V3 (live)": (V3s, 126), "G1 (V3 before filters)": (G1s, 126),
    "V3 + P&L model (2019-trained)": (V3P, 126), "V3, same model without P&L": (V3T, 126),
    "V3 minus weak interest cover": ({o: {d: [s for s in v if not ic_weak(s, d)] for d, v in V3s[o].items()} for o in range(5)}, 126),
    "V3 minus credit events": ({o: {d: [s for s in v if not credit(s, d)] for d, v in V3s[o].items()} for o in range(5)}, 126),
    "V3 promoter buyers first": ({o: boosted(o, pb) for o in range(5)}, 126),
    "V3 pledge releasers first": ({o: boosted(o, pr) for o in range(5)}, 126),
    "V3 held 3 months": (V3s, 63), "V3 held 12 months": (V3s, 252),
}
for name, (sel, hold) in strategies.items():
    sre.HOLD = hold; ys = []
    for o in range(5):
        wk = sorted(sel[o]); nav, _ = sre.run_exit(D, X, sel[o], wk, "E0"); ys.append(yearly(nav))
    rows[name] = {k: float(np.mean([y[k] for y in ys])) for k in ("2024", "2025", "2026", "both")}
    print(f"  {name}: 2025 {rows[name]['2025']:+.1f}% · 2026 {rows[name]['2026']:+.1f}%", flush=True)
sre.HOLD = 126

# big-order event portfolios
PX = rp.load_panel(["open", "close"]); Ow = rp.wide(PX, "open", cal); Craw = rp.wide(PX, "close", cal); del PX
On, Cn = Ow.to_numpy(), Craw.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(Ow.columns)}
mkt = pd.Series((Craw / Craw.shift(1) - 1).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy(), index=cal)
T = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")


def event_nav(Tg: pd.DataFrame, hold: int) -> pd.Series:
    rs, no = np.zeros(len(cal)), np.zeros(len(cal))
    for r in Tg.itertuples():
        c = col.get(r.symbol)
        if c is None:
            continue
        a = int(r.i0); b = min(a + hold, len(cal)); e = On[a, c]
        if not (np.isfinite(e) and e > 0):
            continue
        path = Cn[a:b, c].astype(float); prv = np.concatenate([[e], path[:-1]])
        dr_ = np.where(np.isfinite(path) & np.isfinite(prv) & (prv > 0), path / prv - 1, 0.0); dr_[0] -= 0.005
        rs[a:b] += dr_; no[a:b] += 1
    return pd.Series(np.cumprod(1 + np.where(no > 0, rs / np.maximum(no, 1), 0.0)), index=cal)


for cut, lab in ((0.15, ">= 15%"), (0.5, ">= 50%")):
    for hold, hs in ((252, "12m"), (378, "18m"), (504, "24m")):
        g = T[T["ratio"] >= cut]
        rows[f"Big orders {lab}, held {hs}"] = {k: v for k, v in yearly(event_nav(g, hold)).items() if k != "to"}
        if hs == "12m":
            rows[f"Big orders {lab} in hot industries, held 12m"] = {k: v for k, v in yearly(event_nav(g[g.get("group", "") == "hot"], hold)).items() if k != "to"}
rows["Market (equal-weight all stocks)"] = {k: v for k, v in yearly(pd.Series(np.cumprod(1 + mkt.to_numpy()), index=cal)).items() if k != "to"}
last = yearly(pd.Series(np.cumprod(1 + mkt.to_numpy()), index=cal))["to"]
print(f"\n=== strategies ranked by 2025 + 2026-to-date ({last}) combined · % return ===")
print(f" {'strategy':48s} {'2024':>7s} {'2025':>7s} {'2026 YTD':>9s} {'2025+26':>8s}")
for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["both"]):
    print(f" {name:48s} {r['2024']:+7.1f} {r['2025']:+7.1f} {r['2026']:+9.1f} {r['both']:+8.1f}")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-05-strategies-2025-EXPLORATION",
                             status="EXPLORATION (no rule changed)", producer="src/agentic/explore_strategies_2025.py",
                             results={k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in rows.items()}, data_to=last)) + "\n")
