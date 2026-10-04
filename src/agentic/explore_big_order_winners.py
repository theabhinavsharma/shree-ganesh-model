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
