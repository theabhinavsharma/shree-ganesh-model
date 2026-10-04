"""Big order winners (2026-10-04, EXPLORATION for Abhinav's question; no rule changed).

"What if we only held non-alarming companies that won an order worth >= 50% of their revenue, bought within a week of
the order, and held 3 / 6 / 9 / 12 months?"
Events: data/derived/order_amounts.parquet rows whose current category is 'order' (wins only, not L1 bids), amount
read with high or medium confidence (INR; USD converted at the day's FRED rate), all years 2016+.
Revenue: trailing 12 months of net sales from pnl_quarterly_enriched.parquet (NSE results as first filed, including the
2014-17 old-format backfill), using only quarters filed BEFORE the order; four consecutive quarters on one basis,
consolidated preferred. Order / revenue >= 0.50 qualifies.
Non-alarming: no bad filing in the 90 days before (event_ledger.parquet, direction -1: default/insolvency, auditor or
MD/CEO/CFO exit, independent-director exit, suspension, order cancellation, delayed results, strike, regulatory action,
rating downgrade); trailing-12-month profit > 0; then-traded price >= Rs 20 and 20-day traded value >= Rs 1 cr the day
before entry. One trade per company per 30 days (first order).
Entry: open of the first session after the filing (d_actionable); variant: open 5 sessions later.
Exit: close after 63 / 126 / 189 / 252 sessions (3 / 6 / 9 / 12 months); 0.5% round-trip cost. Trades whose stock
stops trading keep their last price. Typical stock = median of every stock priced on the entry day, same dates.
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

H = {"3m": 63, "6m": 126, "9m": 189, "12m": 252, "15m": 315, "18m": 378, "24m": 504}
COST = 0.005
D = sp.load(None); cal = D["cal"]; pos = {d: i for i, d in enumerate(cal)}
px = D["px"]
O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
O = O[(O["cat_current"] == "order") & O["amount_confidence"].isin(["high", "medium"])].copy()
O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"]); O = O[O["amount_cr"] > 0]
O["ts"] = pd.to_datetime(O["ts"]); O["act"] = pd.to_datetime(O["d_actionable"])
HOT_MODE = __import__("os").environ.get("SGM_HOT") == "1"   # 2026-10-04: big order x hot industry, cleaned amounts
clean = {}
if HOT_MODE:   # cleaning rules fixed before any outcome was computed
    q = O["headline"].fillna("").str.contains(r"^(Clarification|News Verification|Reply to Clarification)|sought clarification|news item|media report", case=False, regex=True)
    clean["exchange question / clarification filings"] = int(q.sum()); O = O[~q]
    dis = O["headline_fulltext_agree"].astype(str) == "False"; clean["headline vs PDF amount disagree"] = int(dis.sum()); O = O[~dis]
    ui = O["amount_flags"].fillna("").str.contains("unit_inferred"); clean["unit guessed (no crore/lakh written)"] = int(ui.sum()); O = O[~ui]
n_orders = len(O)

# trailing-12-month revenue and profit known BEFORE the order (point in time)
Q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly_enriched.parquet")
Q["quarter_end"] = pd.to_datetime(Q["quarter_end"]); Q["filing_dt"] = pd.to_datetime(Q["filing_dt"])
unit = np.where(Q["source"] == "xbrl", 1e-7, 1e-2)                      # manifest: lakh except xbrl rupees
Q["sales_cr"], Q["pat_cr"] = Q["net_sales"] * unit, Q["pat"] * unit
Qs = {s: g.sort_values("filing_dt") for s, g in Q.groupby("symbol")}


def ttm(sym, t):
    g = Qs.get(sym)
    if g is None:
        return np.nan, np.nan
    g = g[g["filing_dt"] < t]
    for basis in ("con", "sa"):
        b = g[g["basis"] == basis].drop_duplicates("quarter_end", keep="last").sort_values("quarter_end").tail(4)
        if len(b) == 4 and (b["quarter_end"].iloc[-1] - b["quarter_end"].iloc[0]).days in range(260, 285) \
                and b["sales_cr"].notna().all() and (t - b["quarter_end"].iloc[-1]).days <= 200:
            return b["sales_cr"].sum(), b["pat_cr"].sum()
    return np.nan, np.nan


O[["rev_ttm_cr", "pat_ttm_cr"]] = [ttm(s, t) for s, t in zip(O["symbol"], O["ts"])]
O["ratio"] = O["amount_cr"] / O["rev_ttm_cr"]
cov = O.groupby(O["ts"].dt.year)["rev_ttm_cr"].apply(lambda v: v.notna().mean()).round(2)
MIN_RATIO = float(__import__("os").environ.get("SGM_MIN_RATIO", "0.5"))   # SGM_MIN_RATIO=0.1: size sweep (2026-10-04)
if HOT_MODE:
    clean["order > 10x trailing revenue"] = int((O["ratio"] > 10).sum()); O = O[~(O["ratio"] > 10)]
big = O[O["ratio"] >= MIN_RATIO].sort_values("ts").copy()
n_big = len(big)

# non-alarming filters
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "direction"])
Lb = L[L["direction"] < 0].groupby("symbol")["filed_at"].apply(lambda s: np.sort(s.values)).to_dict()
big["bad90"] = [bool(len(a := Lb.get(s, np.array([], dtype="datetime64[ns]")))) and
                ((a < np.datetime64(t)) & (a >= np.datetime64(t - pd.Timedelta(days=90)))).any() for s, t in zip(big["symbol"], big["ts"])]
big["i0"] = [cal.searchsorted(a) if a <= cal[-1] else -1 for a in big["act"]]
big = big[(big["i0"] > 0)]
PR = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])   # then-traded price, not adjusted
PR["raw_close"] = rp.raw_price(PR, "close")
PR["adv"] = PR["avg_traded_value_20d"] / 1e7                                     # Rs crore, same formula as sp.load
prev = PR.set_index(["symbol", "trade_date"])[["raw_close", "adv"]]
del PR
look = [prev.loc[(s, cal[i - 1])] if (s, cal[i - 1]) in prev.index else pd.Series({"raw_close": np.nan, "adv": np.nan}) for s, i in zip(big["symbol"], big["i0"])]
big["price_prev"] = [x["raw_close"] for x in look]; big["adv_prev_cr"] = [x["adv"] for x in look]
funnel = {"order wins with a trusted amount": n_orders, ">= 50% of trailing revenue": n_big}
f1 = big[~big["bad90"]]; funnel["no bad filing in 90 days"] = len(f1)
f2 = f1[f1["pat_ttm_cr"] > 0]; funnel["profitable (12 months)"] = len(f2)
f3 = f2[(f2["price_prev"] >= 20) & (f2["adv_prev_cr"] >= 1)]; funnel["price >= Rs 20 and >= Rs 1 cr traded a day"] = len(f3)
keep, last = [], {}
for r in f3.sort_values("ts").itertuples():
    if r.symbol in last and (r.ts - last[r.symbol]).days < 30:
        continue
    last[r.symbol] = r.ts; keep.append(r.Index)
T = f3.loc[keep].copy(); funnel["one per company per 30 days"] = len(T)
if HOT_MODE:   # hot = the company's industry heat percentile >= 0.70 (V3's definition) on the session BEFORE entry
    imap = sp.industry_maps()["analogs"]
    Sx = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet", columns=["date", "industry", "heat_pct"])
    Sx["date"] = pd.to_datetime(Sx["date"]); Sx = Sx.sort_values("date")
    look = pd.DataFrame({"industry": T["symbol"].map(imap).values, "date": [cal[i - 1] for i in T["i0"]], "k": T.index}).dropna(subset=["industry"]).sort_values("date")
    m = pd.merge_asof(look, Sx, on="date", by="industry", direction="backward", tolerance=pd.Timedelta(days=10)).set_index("k")
    T["heat"] = m["heat_pct"].reindex(T.index)
    T["group"] = np.where(T["heat"].isna(), "unknown", np.where(T["heat"] >= 0.70, "hot", "not hot"))
# repeat order = the same company had ANOTHER >= 50% order 31-365 days earlier (2026-10-04 follow-up question)
bigt = big.groupby("symbol")["ts"].apply(lambda s: np.sort(s.values)).to_dict()
T["repeat"] = [((a := bigt.get(s, np.array([], dtype="datetime64[ns]"))) < np.datetime64(t - pd.Timedelta(days=30))).any() and
               (a >= np.datetime64(t - pd.Timedelta(days=365))).any() and
               ((a < np.datetime64(t - pd.Timedelta(days=30))) & (a >= np.datetime64(t - pd.Timedelta(days=365)))).any()
               for s, t in zip(T["symbol"], T["ts"])]

PX = rp.load_panel(["open", "close"]); Ow, Cw = rp.wide(PX, "open", cal), rp.wide(PX, "close", cal).ffill(limit=300); del PX
allsyms = Cw.columns


def trade(s, i0, h):
    if s not in Ow.columns or i0 + h > len(cal):
        return np.nan
    e, x = Ow[s].iloc[i0], Cw[s].iloc[i0 + h - 1]
    return x / e - 1 - COST if np.isfinite(e) and e > 0 and np.isfinite(x) else np.nan


typ_cache = {}


def typical(i0, h):
    k = (i0, h)
    if k not in typ_cache:
        e, x = Ow.iloc[i0], Cw.iloc[i0 + h - 1]; r = (x / e - 1).replace([np.inf, -np.inf], np.nan)
        typ_cache[k] = float(r[(e > 0)].median())
    return typ_cache[k]


rows = []
for lag in (0, 4):
    for name, h in H.items():
        for r in T.itertuples():
            i0 = r.i0 + lag
            if i0 + h > len(cal):
                continue
            rows.append(dict(entry=("next open" if lag == 0 else "5 sessions later"), hold=name, symbol=r.symbol, date=cal[i0], year=cal[i0].year, repeat=r.repeat,
                             group=getattr(r, "group", "all"), ratio=r.ratio,
                             ret=trade(r.symbol, i0, h), typ=typical(i0, h)))
R = pd.DataFrame(rows).dropna(subset=["ret"])
pct = lambda v: f"{v:+.0%}"  # noqa: E731
print("funnel:", " -> ".join(f"{k} {v:,}" for k, v in funnel.items()))
print("orders with a usable trailing revenue, by year:", cov.to_dict())
print("\n=== bought at the next open, after 0.5% cost ===")
out = {}
for (e, h), g in R.groupby(["entry", "hold"], sort=False):
    s = dict(trades=len(g), avg=g.ret.mean(), median=g.ret.median(), up=(g.ret > 0).mean(), hit50=(g.ret >= 0.5).mean(),
             down30=(g.ret <= -0.3).mean(), typical=g.typ.mean(), beat=(g.ret > g.typ).mean())
    out[f"{e} {h}"] = {k: round(float(v), 4) for k, v in s.items()}
    if e == "next open" or h == "6m":
        print(f"{e:16s} {h:>3s}: {len(g):4d} trades · avg {pct(s['avg'])} · median {pct(s['median'])} · up {s['up']:.0%} · "
              f"+50% {s['hit50']:.0%} · -30% or worse {s['down30']:.0%} · typical stock {pct(s['typical'])} · beat typical {s['beat']:.0%}")
if HOT_MODE:
    print("cleaning removed:", clean, "· groups:", T["group"].value_counts().to_dict())
    for cut in (MIN_RATIO, 0.5):
        print(f"\n=== orders >= {cut:.0%} of trailing revenue · hot vs not-hot industry · bought at the next open ===")
        G = R[(R.entry == "next open") & (R.ratio >= cut) & (R.group != "unknown")]
        for (h, gname), g in G.groupby(["hold", "group"], sort=False):
            print(f"{h:>3s} {gname:7s}: {len(g):4d} trades · avg {pct(g.ret.mean())} · median {pct(g.ret.median())} · up {(g.ret > 0).mean():.0%} · "
                  f"+50% {(g.ret >= 0.5).mean():.0%} · 2x {(g.ret >= 1).mean():.0%} · -30% {(g.ret <= -0.3).mean():.0%} · typical stock {pct(g.typ.mean())} · beat {(g.ret > g.typ).mean():.0%}")
            out[f">={cut:.2f} {h} {gname}"] = dict(trades=len(g), avg=round(float(g.ret.mean()), 4), median=round(float(g.ret.median()), 4), up50=round(float((g.ret >= 0.5).mean()), 3))
if HOT_MODE and __import__("os").environ.get("SGM_YOY") == "1":   # year-on-year view (2026-10-04)
    print(f"\n=== YEAR BY YEAR · orders >= {MIN_RATIO:.0%} of revenue · 12-month hold · by year of the order ===")
    g = R[(R.entry == "next open") & (R.hold == "12m") & (R.group != "unknown")]
    cov_y = cov.to_dict()
    yy = g.groupby(["year", "group"]).agg(trades=("ret", "size"), median=("ret", "median"), avg=("ret", "mean"), up=("ret", lambda v: (v > 0).mean()),
                                          typical=("typ", "median")).unstack("group")
    for y in sorted(g.year.unique()):
        row = yy.loc[y]
        def cell(k):
            try:
                return f"{int(row[('trades', k)]):3d} tr · median {pct(row[('median', k)])} · avg {pct(row[('avg', k)])} · up {row[('up', k)]:.0%} · typical stock {pct(row[('typical', k)])}"
            except Exception:
                return "  0 tr"
        print(f"{y} (revenue known for {cov_y.get(y, float('nan')):.0%} of orders){' THIN' if cov_y.get(y, 0) < 0.75 else ''}\n   hot    : {cell('hot')}\n   not hot: {cell('not hot')}")
    if MIN_RATIO >= 0.5:   # few trades: list every hot 12m trade so the pattern can be read by eye
        print("\nevery hot trade, 12m hold (buy date, stock, order as % of revenue, return):")
        for r in g[g.group == "hot"].sort_values("date").itertuples():
            print(f"  {r.date.date()} {r.symbol:12s} {r.ratio:5.0%}  {r.ret:+.0%}")
    # order year x hold period: median trade (trade count), per group; a trade appears only once its full hold has passed
    G2 = R[(R.entry == "next open") & (R.group != "unknown")]
    for stat, gname in [(s_, g_) for s_ in ("median", "mean") for g_ in ("hot", "not hot")]:
        M = G2[G2.group == gname].groupby(["year", "hold"], sort=False).ret.agg([stat, "size"])
        print(f"\n=== {gname.upper()} · {'average' if stat == 'mean' else 'median'} trade (trades) by order year x hold ===\n year  " + "".join(f"{h:>13s}" for h in H))
        for y in sorted(G2.year.unique()):
            print(f" {y}  " + "".join(f"{pct(M.loc[(y, h), stat]):>8s} ({int(M.loc[(y, h), 'size']):2d})" if (y, h) in M.index else f"{'-':>13s}" for h in H)
                  + (" THIN" if cov_y.get(y, 0) < 0.75 else ""))
    # calendar-year returns of a portfolio holding every trade of a group for its hold, equal money per open trade, cash when none
    print("\n=== CALENDAR-YEAR returns: hold every trade of the group for the hold period, equal money in each open trade, cash when none ===")
    Cn = Cw.to_numpy(); On = Ow.to_numpy(); col = {s: i for i, s in enumerate(Cw.columns)}
    for hold_n, hold_s in [(63, "3m"), (126, "6m"), (189, "9m"), (252, "12m"), (315, "15m"), (378, "18m"), (504, "24m")]:
     for gname in ("hot", "not hot"):
        Tg = T[(T["group"] == gname) & (T["ratio"] >= MIN_RATIO)]
        ret_sum = np.zeros(len(cal)); n_open = np.zeros(len(cal))
        for r in Tg.itertuples():
            if r.symbol not in col: continue
            c = col[r.symbol]; i0 = r.i0; i1 = min(i0 + hold_n, len(cal))
            e = On[i0, c]
            if not (np.isfinite(e) and e > 0): continue
            path = Cn[i0:i1, c].astype(float); prev = np.concatenate([[e], path[:-1]])
            dr = np.where(np.isfinite(path) & np.isfinite(prev) & (prev > 0), path / prev - 1, 0.0); dr[0] -= 0.005
            ret_sum[i0:i1] += dr; n_open[i0:i1] += 1
        daily = np.where(n_open > 0, ret_sum / np.maximum(n_open, 1), 0.0)
        nav = pd.Series(np.cumprod(1 + daily), index=cal)
        yl = nav.groupby(nav.index.year).last(); yr = (yl / yl.shift(1).fillna(1.0) - 1)
        opn = pd.Series(n_open, index=cal).groupby(cal.year).mean()
        n21 = nav[nav.index >= "2021-01-01"]; yrs21 = (n21.index[-1] - n21.index[0]).days / 365.25
        cagr = (n21.iloc[-1] / n21.iloc[0]) ** (1 / yrs21) - 1; dd = (n21 / n21.cummax() - 1).min()
        print(f"  {hold_s:>3s} {gname:7s}: since 2021 {cagr:+.0%}/yr · worst fall {dd:.0%} · " + " · ".join(f"{y}: {v:+.0%}" for y, v in yr.items() if y >= 2021) + f" · avg open {opn[opn.index >= 2021].mean():.0f}")
if HOT_MODE and __import__("os").environ.get("SGM_OCT") == "1":   # (2026-10-04) start fresh in the first October week each year; do not-hot industries turn hot later?
    Cn = Cw.to_numpy(); On = Ow.to_numpy(); col = {s: i for i, s in enumerate(Cw.columns)}

    def fresh(Tg, s0, hold_n, end):   # new money from s0: only trades entered in [s0, end), equal money in each open trade, cash when none
        ret_sum = np.zeros(end - s0); n_open = np.zeros(end - s0); n = 0
        for r in Tg.itertuples():
            if r.symbol not in col or not (s0 <= r.i0 < end): continue
            c = col[r.symbol]; a = r.i0; b = min(a + hold_n, end); e = On[a, c]
            if not (np.isfinite(e) and e > 0): continue
            path = Cn[a:b, c].astype(float); prev = np.concatenate([[e], path[:-1]])
            dr = np.where(np.isfinite(path) & np.isfinite(prev) & (prev > 0), path / prev - 1, 0.0); dr[0] -= COST
            ret_sum[a - s0:b - s0] += dr; n_open[a - s0:b - s0] += 1; n += 1
        daily = np.where(n_open > 0, ret_sum / np.maximum(n_open, 1), 0.0)
        return np.prod(1 + daily) - 1, n, (n_open == 0).mean()
    holds_oct = [(189, "9m"), (252, "12m"), (378, "18m"), (504, "24m")]
    for horizon, hz in [(252, "12 months"), (504, "24 months")]:
        print(f"\n=== START FRESH IN THE FIRST OCTOBER WEEK · value after {hz} · hot / not hot · orders >= {MIN_RATIO:.0%} of revenue ===")
        print(" start      " + "".join(f"{h:>15s}" for _, h in holds_oct) + "   trades in   cash days (12m hold)   typical stock")
        for Y in range(2019, 2026):
            s0 = int(cal.searchsorted(pd.Timestamp(f"{Y}-10-01"))); end = min(s0 + horizon, len(cal)); part = "" if s0 + horizon <= len(cal) else " (to date)"
            cells, extra = [], ""
            for hold_n, hs in holds_oct:
                v = {g: fresh(T[(T["group"] == g) & (T["ratio"] >= MIN_RATIO)], s0, hold_n, end) for g in ("hot", "not hot")}
                cells.append(f"{pct(v['hot'][0])} / {pct(v['not hot'][0])}")
                if hs == "12m":
                    extra = f"   {v['hot'][1]:3d} / {v['not hot'][1]:3d}   {v['hot'][2]:.0%} / {v['not hot'][2]:.0%}"
                    out[f"oct{Y} {hz} 12m"] = dict(hot=round(float(v['hot'][0]), 4), not_hot=round(float(v['not hot'][0]), 4))
            print(f" Oct {Y}{part:10s}" + "".join(f"{c:>15s}" for c in cells) + extra + f"          {pct(typical(s0, end - s0))}")
    # cohorts: every order from Oct Y to Sep Y+1 gets the same money; each is sold exactly N months after its own buy (abhinav's framing, 2026-10-04)
    Rn = R[(R.entry == "next open") & (R.group != "unknown") & (R.ratio >= MIN_RATIO)].copy()
    Rn["coh"] = np.where(Rn.date.dt.month >= 10, Rn.date.dt.year, Rn.date.dt.year - 1)
    Tq = T[(T["ratio"] >= MIN_RATIO) & (T["group"] != "unknown")].copy(); dts = pd.DatetimeIndex([cal[i] for i in Tq["i0"]])
    Tq["coh"] = np.where(dts.month >= 10, dts.year, dts.year - 1); bought = Tq.groupby(["coh", "group"]).size()
    hc = ["9m", "12m", "18m", "24m"]
    print("\n=== OCTOBER COHORTS · same money in every order from Oct Y to Sep Y+1, each sold exactly N months after its own buy · avg (median) [sold/bought] ===")
    for gname in ("hot", "not hot"):
        print(f" {gname.upper()}\n cohort        " + "".join(f"{h:>24s}" for h in hc))
        for Y in range(2018, 2026):
            cells = []
            for h in hc:
                g = Rn[(Rn.coh == Y) & (Rn.group == gname) & (Rn.hold == h)]
                cells.append(f"{pct(g.ret.mean())} ({pct(g.ret.median())}) [{len(g)}/{bought.get((Y, gname), 0)}]" if len(g) else f"- [0/{bought.get((Y, gname), 0)}]")
            print(f" Oct {Y}-Sep {Y + 1 - 2000:02d}" + "".join(f"{c:>24s}" for c in cells))
    s0 = int(cal.searchsorted(pd.Timestamp("2024-10-01"))); end = min(s0 + 252, len(cal))
    print("eyeball · Oct 2024 start, hot, 12m hold, return to the earlier of exit or Oct 2025:")
    for r in T[(T["group"] == "hot") & (T["ratio"] >= MIN_RATIO)].itertuples():
        if r.symbol in col and s0 <= r.i0 < end:
            c = col[r.symbol]; x = min(r.i0 + 252, end) - 1
            print(f"  {cal[r.i0].date()} {r.symbol:12s} {Cn[x, c] / On[r.i0, c] - 1:+.0%} to {cal[x].date()}")
    # not-hot trades: does the industry turn hot (heat_pct >= 0.70) during the 24 months after entry, and when? (uses later data: an explanation, not a rule)
    Hp = Sx.pivot_table(index="date", columns="industry", values="heat_pct").reindex(cal).ffill(limit=7)
    icol = {k: i for i, k in enumerate(Hp.columns)}
    for dname, Hn in [("any single day >= 0.70", Hp.to_numpy()), ("20-session average >= 0.70", Hp.rolling(20, min_periods=15).mean().to_numpy())]:
     fl = []
     for r in T[(T["group"] == "not hot") & (T["ratio"] >= MIN_RATIO)].itertuples():
         ind = imap.get(r.symbol); a = r.i0
         if ind not in icol or r.symbol not in col or a + 504 > len(cal): continue
         hit = np.where(Hn[a:a + 504, icol[ind]] >= 0.70)[0]; j = a + int(hit[0]) if len(hit) else None
         c = col[r.symbol]; e = On[a, c]
         d = dict(m=np.nan if j is None else (j - a) / 21, r12=trade(r.symbol, a, 252), r24=trade(r.symbol, a, 504), pre=np.nan, post=np.nan, f12=np.nan, f12typ=np.nan)
         if j is not None and j + 1 < a + 504:
             f0 = On[j + 1, c]
             if np.isfinite(f0) and f0 > 0 and np.isfinite(e) and e > 0:
                 d["pre"] = f0 / e - 1; d["post"] = Cn[a + 503, c] / f0 - 1 - COST
             if j + 1 + 252 <= len(cal):
                 d["f12"] = trade(r.symbol, j + 1, 252); d["f12typ"] = typical(j + 1, 252)
         fl.append(d)
     F = pd.DataFrame(fl)
     print(f"\n=== NOT-HOT trades with a full 24 months ({len(F)}): does the industry turn hot later, and when? · turned hot = {dname} ===")
     print(" share turned hot by: " + " · ".join(f"{k}m {(F.m < k).mean():.0%}" for k in (3, 6, 9, 12, 18, 24)))
     F["when"] = pd.cut(F.m, [-0.01, 3, 9, 24], labels=["within 3m", "3-9m", "9-24m"]).astype(str).replace("nan", "never in 24m")
     for w in ["within 3m", "3-9m", "9-24m", "never in 24m"]:
         g = F[F.when == w]
         if not len(g): continue
         s = (f"{w:13s}: {len(g):3d} trades · 12m avg {pct(g.r12.mean())} median {pct(g.r12.median())} · 24m avg {pct(g.r24.mean())} median {pct(g.r24.median())}")
         if w != "never in 24m":
             s += f" · entry->turn avg {pct(g.pre.mean())} · turn->24m end avg {pct(g.post.mean())} median {pct(g.post.median())}"
         print(s); out[f"flip {dname[:6]} {w}"] = dict(trades=len(g), r24_avg=round(float(g.r24.mean()), 4), r24_median=round(float(g.r24.median()), 4))
     g = F.dropna(subset=["f12"])
     print(f"same {len(g)} trades · buy at the order, hold 12m: avg {pct(g.r12.mean())} · median {pct(g.r12.median())} · up {(g.r12 > 0).mean():.0%}\n"
           f"same {len(g)} trades · buy when the industry turns hot, hold 12m: avg {pct(g.f12.mean())} · median {pct(g.f12.median())} · up {(g.f12 > 0).mean():.0%} · typical stock {pct(g.f12typ.mean())}")
    Hh = R[(R.entry == "next open") & (R.hold == "12m") & (R.group == "hot") & (R.date <= cal[len(cal) - 504])]
    print(f"for reference · hot at the order, same window (entry by {cal[len(cal) - 504].date()}), hold 12m: {len(Hh)} trades · avg {pct(Hh.ret.mean())} · median {pct(Hh.ret.median())} · up {(Hh.ret > 0).mean():.0%}")
if HOT_MODE and __import__("os").environ.get("SGM_NOW") == "1":   # (2026-10-04) "is history repeating?": each October's backdrop vs what the next year of orders did
    Tq = T[(T["ratio"] >= MIN_RATIO) & (T["group"] != "unknown")]
    Rn = R[(R.entry == "next open") & (R.group != "unknown") & (R.ratio >= MIN_RATIO)].copy()
    Rn["coh"] = np.where(Rn.date.dt.month >= 10, Rn.date.dt.year, Rn.date.dt.year - 1)
    On_ = Ow.to_numpy(); Cn_ = Cw.to_numpy()
    print(f"\n=== EVERY OCTOBER · backdrop then vs the next 12 months of orders >= {MIN_RATIO:.0%} (12m hold) ===")
    print(" Oct    typical stock past 12m · past 24m · below 2-yr high   big orders past 12m (hot)   next year's orders: hot avg (median) n · not hot avg (median) n · typical stock next 12m")
    for Y in range(2019, 2027):
        s0 = min(int(cal.searchsorted(pd.Timestamp(f"{Y}-10-01"))), len(cal) - 1)
        live = np.isfinite(On_[s0 - 1]) & (On_[s0 - 1] > 0)
        below = np.nanmedian(Cn_[s0 - 1, live] / np.nanmax(Cn_[s0 - 504:s0, live], axis=0) - 1)
        w = Tq[(Tq.i0 >= s0 - 252) & (Tq.i0 < s0)]
        def c(gn):
            g = Rn[(Rn.coh == Y) & (Rn.hold == "12m") & (Rn.group == gn)]
            return f"{pct(g.ret.mean())} ({pct(g.ret.median())}) {len(g)}" if len(g) else "not done yet"
        nxt = pct(typical(s0, 252)) if s0 + 252 <= len(cal) else "-"
        print(f" {Y}   {pct(typical(s0 - 252, 252)):>6s} · {pct(typical(s0 - 504, 504)):>6s} · {pct(below):>6s}            {len(w):4d} ({(w.group == 'hot').mean():.0%})          {c('hot')} · {c('not hot')} · {nxt}")
    hot_now = Sx[Sx.date == Sx.date.max()].sort_values("heat_pct", ascending=False)
    print(f"hottest industries on {Sx.date.max().date()}: " + ", ".join(f"{r.industry} {r.heat_pct:.2f}" for r in hot_now.head(12).itertuples()))
    last = len(cal) - 1; rec = Tq[Tq.i0 >= last - 126].sort_values("i0")
    print(f"orders >= {MIN_RATIO:.0%} in the last 6 months: {len(rec)} ({(rec.group == 'hot').mean():.0%} hot) · return to {cal[last].date()} (median): hot "
          + pct(np.nanmedian([Cn_[last, Cw.columns.get_loc(s)] / On_[i, Cw.columns.get_loc(s)] - 1 for s, i, gg in zip(rec.symbol, rec.i0, rec.group) if gg == "hot" and s in Cw.columns])) + " · not hot "
          + pct(np.nanmedian([Cn_[last, Cw.columns.get_loc(s)] / On_[i, Cw.columns.get_loc(s)] - 1 for s, i, gg in zip(rec.symbol, rec.i0, rec.group) if gg == "not hot" and s in Cw.columns])))
    for r in rec[rec.group == "hot"].itertuples():
        if r.symbol in Cw.columns:
            k = Cw.columns.get_loc(r.symbol)
            print(f"  {cal[r.i0].date()} {r.symbol:12s} {imap.get(r.symbol, '?')[:28]:28s} {r.ratio:5.0%}  {Cn_[last, k] / On_[r.i0, k] - 1:+.0%} so far")
if HOT_MODE and __import__("os").environ.get("SGM_ROLL") == "1":   # (2026-10-04) buy every order over ANY 12-month window, sell each 12/18/24 months after its own buy
    Rr = R[(R.entry == "next open") & (R.group != "unknown") & (R.ratio >= MIN_RATIO)]
    rows3 = []
    for st in pd.date_range("2019-01-01", cal[-1], freq="MS"):
        en = st + pd.DateOffset(months=12); i_en = int(cal.searchsorted(en))
        for h, mo in (("12m", 12), ("18m", 18), ("24m", 24)):
            if i_en + H[h] > len(cal): continue   # every trade in the window must have finished its hold
            w = Rr[(Rr.date >= st) & (Rr.date < en) & (Rr.hold == h)]
            for gname, g in (("hot", w[w.group == "hot"]), ("not hot", w[w.group == "not hot"]), ("all", w)):
                if len(g): rows3.append(dict(start=st, hold=h, mo=mo, group=gname, n=len(g), avg=g.ret.mean(), med=g.ret.median(), typ=g.typ.mean()))
    W = pd.DataFrame(rows3)
    print(f"\n=== ANY 12-MONTH BUYING WINDOW (start = 1st of each month from Jan 2019) · same money in every order >= {MIN_RATIO:.0%} · each sold N months after its own buy ===")
    print(" windows overlap month to month, so 58 windows at 24m are roughly 5 independent years")
    for gname in ("all", "hot", "not hot"):
        for h in ("12m", "18m", "24m"):
            g = W[(W.group == gname) & (W.hold == h)]
            if not len(g): continue
            mo = int(g.mo.iloc[0]); wv = g.loc[g.avg.idxmin()]; bv = g.loc[g.avg.idxmax()]
            print(f" {gname:7s} {h}: {len(g):2d} windows ({g.start.min():%b %Y} to {g.start.max():%b %Y}) · window return median {pct(g.avg.median())} (≈ {pct((1 + g.avg.median()) ** (12 / mo) - 1)} a year)"
                  f" · lost money in {(g.avg < 0).mean():.0%} · worst {pct(wv.avg)} ({wv.start:%b %Y}) · best {pct(bv.avg)} ({bv.start:%b %Y}) · typical stock median {pct(g.typ.median())} · trades per window median {g.n.median():.0f}")
            out[f"roll {gname} {h}"] = dict(windows=len(g), median=round(float(g.avg.median()), 4), lost=round(float((g.avg < 0).mean()), 3))
    print("\n window starting · 12m / 18m / 24m hold · all orders avg (hot avg · not hot avg)")
    for st in sorted(W.start.unique()):
        if pd.Timestamp(st).month not in (1, 4, 7, 10): continue
        cells = []
        for h in ("12m", "18m", "24m"):
            q = W[(W.start == st) & (W.hold == h)].set_index("group")
            cells.append(f"{pct(q.loc['all', 'avg'])} ({pct(q.loc['hot', 'avg']) if 'hot' in q.index else '-'} · {pct(q.loc['not hot', 'avg']) if 'not hot' in q.index else '-'})" if "all" in q.index else "not finished")
        print(f" {pd.Timestamp(st):%b %Y}   " + "   ".join(f"{c:>26s}" for c in cells))
print("\n=== first order vs repeat order (bought at the next open) ===")
for (h, rep), g in R[R.entry == "next open"].groupby(["hold", "repeat"], sort=False):
    print(f"{h:>3s} {'repeat' if rep else 'first ':6s}: {len(g):4d} trades · avg {pct(g.ret.mean())} · median {pct(g.ret.median())} · up {(g.ret > 0).mean():.0%} · "
          f"+50% {(g.ret >= 0.5).mean():.0%} · -30% or worse {(g.ret <= -0.3).mean():.0%} · typical stock {pct(g.typ.mean())} · beat typical {(g.ret > g.typ).mean():.0%}")
    out[f"next open {h} {'repeat' if rep else 'first'}"] = dict(trades=len(g), avg=round(float(g.ret.mean()), 4), median=round(float(g.ret.median()), 4))
if MIN_RATIO < 0.5:
    T2 = T.set_index(["symbol", "ts"])["ratio"]
    for hh in ("24m", "12m"):
        g = R[(R.entry == "next open") & (R.hold == hh)].merge(T[["symbol", "i0", "ratio", "amount_cr", "rev_ttm_cr", "ts"]].assign(date=lambda x: [cal[i] for i in x["i0"]]), on=["symbol", "date"])
        g["size"] = pd.cut(g["ratio"], [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 1e9], right=False, labels=["10-25%", "25-50%", "50-75%", "75-100%", "100-150%", "150-250%", ">=250%"])
        print(f"\n=== {hh} hold, by order size (% of trailing revenue) ===")
        print(g.groupby("size", observed=True).agg(trades=("ret", "size"), doubled=("ret", lambda v: (v >= 1.0).mean()), up50=("ret", lambda v: (v >= 0.5).mean()),
              median=("ret", "median"), avg=("ret", "mean"), typical=("typ", "median"), lost30=("ret", lambda v: (v <= -0.3).mean())).round(2).to_string())
        if hh == "24m":
            print("\ntop 12 outliers (24m):")
            for r in g.nlargest(12, "ret").itertuples():
                print(f"  {r.ts.date()} {r.symbol:12s} {r.ret:+.0%} · order Rs {r.amount_cr:,.0f} cr vs revenue Rs {r.rev_ttm_cr:,.0f} cr ({r.ratio:.0%})")
g6 = R[(R.entry == "next open") & (R.hold == "6m")]
print("\n6-month trades by entry year (median trade vs median typical):")
print(g6.groupby("year").agg(trades=("ret", "size"), median=("ret", "median"), typical=("typ", "median"), beat=("ret", lambda v: 0)).assign(
    beat=g6.groupby("year").apply(lambda x: (x.ret > x.typ).mean())).round(2).to_string())
print("\n8 random qualifying orders (eyeball the amounts):")
for r in T.sample(min(8, len(T)), random_state=3).itertuples():
    print(f"  {r.ts.date()} {r.symbol:12s} order Rs {r.amount_cr:,.0f} cr · trailing revenue Rs {r.rev_ttm_cr:,.0f} cr ({r.ratio:.0%}) · {str(r.amount_snippet)[:110]}")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=("EXP-2026-10-04-hot-big-order-EXPLORATION" if HOT_MODE else ("EXP-2026-10-04-order-size-sweep-EXPLORATION" if MIN_RATIO < 0.5 else "EXP-2026-10-04-big-order-winners-v2-EXPLORATION")),
                             status="EXPLORATION (no rule changed; uses 2016-2026 incl. 2023+, so 2023+ is no longer clean for this idea)",
                             producer="src/agentic/explore_big_order_winners.py", funnel=funnel, results=out)) + "\n")
