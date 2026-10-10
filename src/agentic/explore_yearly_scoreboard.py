"""Year-by-year scoreboard of every registered V3 test arm (2026-10-11, EXPLORATION for Abhinav: "har output registered test ka
2023, 2024, 2025 and 2026 till now mein split karke batana — i want recent years to outperform as well, beyond just the average").

Every arm is rebuilt exactly as its registered test defines it and run on the same engine (sim_screen_rank_exit.run_exit, 126-
session hold, 26-slot weekly ladder), with every ladder starting on 2019-01-01 so the calendar years compare like for like
(the bad-news tests were registered on a 2023+ window; here they start in 2019 too). Calendar-year return = NAV at the year's
last session / NAV at the previous year's last session - 1; 2026 = to the last data day. Averaged over the 5 weekly entry
schedules. Before costs beyond the engine's 0.5% round trip. Verdicts are the registered results (logs/experiments.jsonl).
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

YEARS = (2023, 2024, 2025, 2026)
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; P = sre.model_scores(); cal = D["cal"]
ctx = v3_rule.context(imap)
START = pd.Timestamp("2019-01-01")
ph = {o: [d for d in sp.weekly_grid(cal, o) if d >= START] for o in range(5)}
CLOSE = pd.Timedelta(hours=15, minutes=30)


def full_pool(o):
    top = sre.TOPN; sre.TOPN = 100000
    try:
        return tif.select_elig(X["F"], ph[o], imap, P, ctx["G1"])
    finally:
        sre.TOPN = top


V3 = {o: v3_rule.picks(X["F"], ph[o], imap, P, ctx) for o in range(5)}
POOL = {o: full_pool(o) for o in range(5)}

# ---- event flags
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "bucket"])
L["filed_at"] = pd.to_datetime(L["filed_at"]); L["sym"] = L["symbol"].map(nse_symbols.now)
BAD = ["default_insolvency", "mgmt_exit", "independent_director_exit", "auditor_exit", "suspension", "order_cancel", "delayed_results", "strike_disruption"]
EVB = {s: np.sort(g["filed_at"].values) for s, g in L[L["bucket"].isin(BAD)].groupby("sym")}
CR = {s: np.sort(g["filed_at"].values) for s, g in L[L["bucket"].isin(["rating_down", "default_insolvency"])].groupby("sym")}
E = pd.read_parquet(ROOT / "data/derived/pledge_events.parquet"); E["broadcast_dt"] = pd.to_datetime(E["broadcast_dt"])
EC = E[(E["event"].str.lower() == "creation") & E["encumbrance"].str.contains("pledge", case=False, na=False)]
NEWP = {s: np.sort(g["broadcast_dt"].values) for s, g in EC.groupby("symbol_now")}
ER = E[(E["event"] == "Release") & E["encumbrance"].str.contains("pledge", case=False, na=False)]
REL = {s: (g["broadcast_dt"].values, g["pct_of_capital"].clip(0, 100).values) for s, g in ER.sort_values("broadcast_dt").groupby("symbol_now")}
A = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"]); A["trade_date"] = pd.to_datetime(A["trade_date"])
PBD = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}


def within(arr, d, days, end_close=True):
    if arr is None:
        return False
    hi = np.datetime64(d + CLOSE) if end_close else np.datetime64(d)
    return bool(((arr <= hi) & (arr > hi - np.timedelta64(days, "D"))).any())


bad = lambda s, d: within(EVB.get(nse_symbols.now(s)), d, 90)  # noqa: E731
newpledge = lambda s, d: within(NEWP.get(nse_symbols.now(s)), d, 90)  # noqa: E731
credit = lambda s, d: any(within(CR.get(x), d, 180, False) for x in {s, nse_symbols.now(s)})  # noqa: E731


def pb(s, d):
    a = PBD.get(s)
    if a is None:
        return False
    i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
    return bool(i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") and a[1][i] > 0)


def pr(s, d):
    a = REL.get(nse_symbols.now(s))
    if a is None:
        return False
    m = (a[0] < np.datetime64(d)) & (a[0] >= np.datetime64(d - pd.Timedelta(days=90)))
    return float(a[1][m].sum()) >= 1.0


IC = tl.interest_cover({nse_symbols.now(s) for v in V3.values() for names in v.values() for s in names})


def ic_weak(s, d):
    a = IC.get(nse_symbols.now(s))
    if a is None:
        return False
    i = np.searchsorted(a[0], np.datetime64(pd.Timestamp(d)), side="left") - 1
    return i >= 0 and a[1][i][0] == "weak"


def drop(fn):
    return {o: {d: [s for s in v if not fn(s, d)] for d, v in V3[o].items()} for o in range(5)}


def refill(fn):
    return {o: {d: [s for s in [x for x in POOL[o].get(d, []) if not fn(x, d)][:9] if ctx["keep"](s, d)] for d in ph[o]} for o in range(5)}


def boost(fn):
    out = {}
    for o in range(5):
        out[o] = {}
        for d in ph[o]:
            pool = POOL[o].get(d, []); first = [s for s in pool[:25] if fn(s, d)]
            pool = first + [s for s in pool if s not in set(first)]
            out[o][d] = [s for s in pool[:9] if ctx["keep"](s, d)]
    return out


# ---- regime
Cw = rp.wide(rp.load_panel(["close"]), "close", cal)
idx = (1 + (Cw / Cw.shift(1) - 1).clip(-0.5, 1.0).mean(axis=1).fillna(0.0)).cumprod()
OFF = pd.Series((idx < idx.rolling(200, min_periods=200).mean()).to_numpy(), index=cal)
Xoff = dict(X, below=np.broadcast_to(OFF.to_numpy()[:, None], X["below"].shape))
SKIP = {o: {d: ([] if OFF.get(d, False) else V3[o].get(d, [])) for d in ph[o]} for o in range(5)}

# ---- new listings (as EXP-2026-10-11-v3-new-listings)
F = X["F"].sort_values(["symbol", "trade_date"]).copy()
firstd = D["px"].groupby("symbol")["trade_date"].min(); ren = set(nse_symbols.now_map().values())
F["age"] = F.groupby("symbol").cumcount(); F["first_close"] = F.groupby("symbol")["close"].transform("first")
F["young"] = F["symbol"].map(firstd).gt(pd.Timestamp("2015-01-10")) & ~F["symbol"].isin(ren) & F["age"].between(63, 251)


def nl_select(weekly, mode):
    W = F[F["trade_date"].isin(set(weekly)) & F["core"]].copy()
    W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
    W = W.merge(ctx["G1"][["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = (W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])
        tr = tr | (W["young"] if mode == "NLOPEN" else (W["young"] & (W["close"] / W["lo252"] - 1 >= 0.5) & (W["close"] / W["first_close"] - 1 >= 0.30) & (W["close"] > W["sma_50"])))
    Pool = W[tr].copy()
    Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
    m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
    Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
    g1 = Pool.groupby("trade_date").head(9).groupby("trade_date")["symbol"].apply(list).to_dict()
    return {d: [s for s in g1.get(d, []) if ctx["keep"](s, d)] for d in weekly}


# ---- other model scores
PN = pd.read_parquet(ROOT / "logs/leader_sleeve/anatomy_1p5x_noboom/bakeoff_preds.parquet", columns=["symbol", "trade_date", "ensemble"])
PN["trade_date"] = pd.to_datetime(PN["trade_date"]).astype("datetime64[ns]"); PN = PN.dropna(subset=["ensemble"]).sort_values("trade_date")
PM = tp.scores(ROOT / "logs/leader_sleeve/v3_pnl/rerun_datafill-20261004/model_P/bakeoff_preds.parquet")

ARMS = [
    ("V3 (live)", "—", {o: V3[o] for o in range(5)}, X, "E0"),
    ("G1 (V3 before its two filters)", "EXP-2026-09-30-v3-vs-g1: V3 passed vs G1", {o: v3_rule.picks(X["F"], ph[o], imap, P, dict(ctx, keep=lambda s, d: True)) for o in range(5)}, X, "E0"),
    ("V2 (V3 without the fading-theme filter)", "theme-engine: filter lost its pass on 2026-10-09", {o: v3_rule.picks(X["F"], ph[o], imap, P, dict(ctx, keep=lambda s, d: not ctx["fin"](s))) for o in range(5)}, X, "E0"),
    ("V3 + P&L in the model (2020+)", "EXP-2026-09-30-v3-pnl: no pass", {o: v3_rule.picks(X["F"], [d for d in ph[o] if d >= pd.Timestamp("2020-01-01")], imap, PM, ctx) for o in range(5)}, X, "E0"),
    ("V3 minus weak interest cover", "EXP-2026-10-04-v3-leverage: no pass", drop(ic_weak), X, "E0"),
    ("V3 minus credit events", "EXP-2026-10-04-v3-leverage: no pass", drop(credit), X, "E0"),
    ("V3 promoter buyers first", "EXP-2026-10-04-v3-promoter-signals: no pass", boost(pb), X, "E0"),
    ("V3 pledge releasers first", "EXP-2026-10-04-v3-promoter-signals: no pass", boost(pr), X, "E0"),
    ("V3 minus bad news (cash)", "EXP-2026-10-07-v3-badnews-veto: no pass", drop(bad), X, "E0"),
    ("V3 bad news swapped (refill)", "EXP-2026-10-11-v3-badnews-refill: no pass", refill(bad), X, "E0"),
    ("V3 minus new promoter pledges", "EXP-2026-10-10-v3-new-pledge-veto: no pass", drop(newpledge), X, "E0"),
    ("V3 on the boom-free model", "EXP-2026-10-10-model-minus-boom: no pass", {o: v3_rule.picks(X["F"], ph[o], imap, PN, ctx) for o in range(5)}, X, "E0"),
    ("V3 regime: skip buys", "EXP-2026-10-10-v3-regime-switch: no pass", SKIP, X, "E0"),
    ("V3 regime: skip + sell", "EXP-2026-10-10-v3-regime-switch: no pass", SKIP, Xoff, "E3"),
    ("V3 + new listings after 3m", "EXP-2026-10-11-v3-new-listings: PASS", {o: nl_select(ph[o], "NLOPEN") for o in range(5)}, X, "E0"),
    ("V3 + new listings, trend since listing", "EXP-2026-10-11-v3-new-listings: no pass", {o: nl_select(ph[o], "NL") for o in range(5)}, X, "E0"),
]


def yearly(nav):
    n = nav / nav.iloc[0]; ye = n.groupby(n.index.year).last()
    r = (ye / ye.shift(1).fillna(1.0) - 1) * 100
    yrs_ = (n.index[-1] - n.index[0]).days / 365.25
    return {**{int(y): float(r.get(y, np.nan)) for y in YEARS}, "cagr": float(n.iloc[-1] ** (1 / yrs_) - 1) * 100}


rows = []
for name, test, sel, x, rule in ARMS:
    ys = []
    for o in range(5):
        wk = sorted(sel[o]); nav, _ = sre.run_exit(D, x, sel[o], wk, rule); ys.append(yearly(nav))
    r = {k: float(np.mean([y[k] for y in ys])) for k in (*YEARS, "cagr")}
    rows.append({"arm": name, "test": test, **r}); print(f"  {name}: done", flush=True)
T = pd.DataFrame(rows).set_index("arm")
base = T.loc["V3 (live)"]
print(f"\n=== calendar-year return %, average of 5 entry schedules (2026 = to {cal[-1].date()}) · [years beating V3 out of 4]")
print(f" {'arm':40s} {'CAGR':>6s} " + " ".join(f"{y:>8d}" for y in YEARS) + "   beats V3   registered result")
for name, r in T.iterrows():
    wins = sum(r[y] > base[y] for y in YEARS) if name != "V3 (live)" else None
    print(f" {name:40s} {r['cagr']:6.1f} " + " ".join(f"{r[y]:+8.1f}" for y in YEARS) + (f"   {wins}/4      " if wins is not None else "   —        ") + r["test"])
OUT = ROOT / "logs/explorations/yearly_scoreboard"; OUT.mkdir(parents=True, exist_ok=True)
T.round(2).to_csv(OUT / "scoreboard.csv")
(OUT / "scoreboard.csv.manifest.json").write_text(json.dumps(dict(dataset="scoreboard.csv", producer="src/agentic/explore_yearly_scoreboard.py", status="EXPLORATION (no rule changed)",
    definitions=__doc__, columns={"arm": "rule variant", "test": "registered test and its verdict", "2023..2026": "calendar-year return, percent, average of 5 entry schedules (2026 to date)",
    "cagr": "percent a year 2019 -> last day, average of 5 schedules"}, data_to=str(cal[-1].date()), updated=datetime.now().isoformat(timespec="seconds")), indent=1))
(OUT / "README.md").write_text("# Year-by-year scoreboard of every registered V3 test (exploration)\n\nEach arm's calendar-year returns 2023-2026 next to V3. scoreboard.csv; definitions in its manifest.\n")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-11-yearly-scoreboard-EXPLORATION", status="EXPLORATION (no rule changed)",
                             producer="src/agentic/explore_yearly_scoreboard.py", results=T.round(1).reset_index().to_dict(orient="records"))) + "\n")
