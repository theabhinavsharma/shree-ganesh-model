"""Rs 45 lakh from Apr 2019 (2026-10-13; Abhinav: if I had 45L in 2019 - how much would it be now?). EXPLORATION.
Continuous 26-slice V3.2 ladder from 2019-04-01: headline (pre-tax, list-day close); realistic (after tax 20.8/13.0%, next open, open
lots taxed as sold on the last date); hindsight-free (bare G1 rules 2019-20, then the walk-forward chooser's sets, after tax + open);
equal-weight liquid market on the same plan after tax + open. Average of the 5 weekly schedules.
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "src/agentic")
import explore_v32_yearly_realistic as E, research_panel as rp, sim_leader_portfolio_7x as sp, sim_screen_rank_exit as sre, v3_rule, test_industry_fundamentals as tif
from test_v3_l1_newlist_cost import run_cost
from test_v32_hold_after_tax import engines, ST, LT
D = sp.load(None); X = sre.features(D); imap = sp.industry_maps()["analogs"]; cal = D["cal"]; P = sre.model_scores()
ctx = v3_rule.context(imap); F1 = v3_rule.frame_v31(X["F"], D); F1["core_p10"] = (F1["adv"] >= 1.0) & (F1["raw"] > 10)
L = rp.load_panel(["avg_traded_value_20d"]); L["adv"] = L["avg_traded_value_20d"] / 1e7
ADV = rp.wide(L, "adv", cal).reindex(columns=D["syms"]).to_numpy(); del L
S = pd.read_parquet(rp.PANEL, columns=["symbol", "trade_date", "series"]); S["trade_date"] = pd.to_datetime(S["trade_date"])
TW = rp.wide(S.assign(t=(S["series"] != "EQ").astype(float)), "t", cal).ffill(); del S
tcol = {s: i for i, s in enumerate(TW.columns)}; tpos = {d: i for i, d in enumerate(cal)}; TWn = TW.to_numpy(); del TW
t2t = lambda s, d: (s in tcol) and TWn[tpos[d], tcol[s]] == 1.0
cost = lambda i0, cols: np.where(ADV[i0, cols] >= 5, 0.005, np.where(ADV[i0, cols] >= 1, 0.010, 0.020))
A = pd.read_parquet("logs/leader_sleeve/anatomy_1p5x/rows.parquet", columns=["symbol", "trade_date", "prom_buys90"]); A["trade_date"] = pd.to_datetime(A["trade_date"])
PBd = {s: (g["trade_date"].values, g["prom_buys90"].values) for s, g in A.sort_values("trade_date").groupby("symbol")}
def pb(s, d):
    a = PBd.get(s)
    if a is None: return np.nan
    i = np.searchsorted(a[0], np.datetime64(d), side="right") - 1
    return a[1][i] if i >= 0 and (np.datetime64(d) - a[0][i]) <= np.timedelta64(7, "D") else np.nan
def ranked(weeks, core_col, topn=25):
    W = F1[F1["trade_date"].isin(set(weeks)) & F1[core_col].fillna(False).astype(bool)].copy()
    W["ind"] = W["symbol"].map(imap); W = W[W["ind"].notna() & W["ret60"].notna()]
    W = W.merge(ctx["G1"][["date", "industry"]].rename(columns={"date": "trade_date", "industry": "ind"}), on=["trade_date", "ind"])
    with np.errstate(invalid="ignore", divide="ignore"):
        tr = (W["close"] / W["lo252"] - 1 >= 0.5) & (W["ret252"] >= 0.30) & (W["close"] > W["sma_200"]) & (W["sma_50"] > W["sma_200"])
    Pool = W[tr].copy(); Lx = Pool[["symbol", "trade_date"]].reset_index().sort_values("trade_date"); Lx["trade_date"] = Lx["trade_date"].astype("datetime64[ns]")
    m = pd.merge_asof(Lx, P, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=7))
    Pool["score"] = m.set_index("index")["ensemble"].reindex(Pool.index).fillna(-np.inf)
    Pool = Pool.sort_values(["trade_date", "score", "symbol"], ascending=[True, False, True], kind="mergesort")
    g = Pool.groupby("trade_date").head(topn).groupby("trade_date")["symbol"].apply(list).to_dict(); return {d: g.get(d, []) for d in weeks}
def wf(weeks, x):
    R = ranked(weeks, "core_p10"); out = {}
    for d in weeks:
        pool = R[d]; first = [s for s in pool[:25] if pb(s, d) > 0]; pool = first + [s for s in pool if s not in set(first)]
        out[d] = [s for s in pool[:9] if not ctx["fin"](s) and not ctx["fading_only"](s, d) and not (x and t2t(s, d))]
    return out
run_tax, run_tax_open = engines()
liquid = F1[F1["core"].fillna(False).astype(bool) & F1["symbol"].isin(D["cidx"])].groupby("trade_date")["symbol"].apply(list).to_dict()
def yr(nav, a, b):
    x = nav[nav.index <= b]; y = nav[nav.index <= a]
    return x.iloc[-1] / (y.iloc[-1] if len(y) else nav.iloc[0])
res = []
for o in range(5):
    wk = [d for d in sp.weekly_grid(cal, o) if d >= pd.Timestamp("2019-04-01")]
    v32 = {d: [s for s in l if not t2t(s, d)] for d, l in v3_rule.picks_v31(F1, wk, imap, P, ctx).items()}
    keep = sre.TOPN; sre.TOPN = 9; base = tif.select_elig(X["F"], wk, imap, P, ctx["G1"]); sre.TOPN = keep; base = {d: base.get(d, []) for d in wk}
    n = {"headline": run_cost(D, X, v32, wk, "E0", cost)[0], "realistic": run_tax_open(D, X, v32, wk, "E0", cost, ST, LT)[0],
         "base": run_tax_open(D, X, base, wk, "E0", cost, ST, LT)[0], "wf0": run_tax_open(D, X, wf(wk, False), wk, "E0", cost, ST, LT)[0],
         "wf1": run_tax_open(D, X, wf(wk, True), wk, "E0", cost, ST, LT)[0], "market": run_tax_open(D, X, {d: liquid.get(d, []) for d in wk}, wk, "E0", cost, ST, LT)[0]}
    end = cal[-1]
    hf = yr(n["base"], pd.Timestamp("2000-01-01"), pd.Timestamp("2020-12-31")) * yr(n["wf0"], pd.Timestamp("2020-12-31"), pd.Timestamp("2022-12-31")) * yr(n["wf1"], pd.Timestamp("2022-12-31"), end)
    res.append(dict(o=o, headline=n["headline"].iloc[-1] / n["headline"].iloc[0], realistic=n["realistic"].iloc[-1] / n["realistic"].iloc[0],
                    hindsight_free=hf, market=n["market"].iloc[-1] / n["market"].iloc[0], start=wk[0].date(), end=end.date()))
    print(res[-1], flush=True)
R = pd.DataFrame(res); yrs = (pd.Timestamp(R.end.iloc[0]) - pd.Timestamp("2019-04-01")).days / 365.25
for k in ("headline", "realistic", "hindsight_free", "market"):
    print(f"{k:15s} Rs 45 L -> Rs {45 * R[k].mean():,.0f} L (range {45 * R[k].min():,.0f}-{45 * R[k].max():,.0f} L over 5 start weekdays) · x{R[k].mean():.1f} · {((R[k].mean()) ** (1 / yrs) - 1) * 100:.1f}%/yr")
