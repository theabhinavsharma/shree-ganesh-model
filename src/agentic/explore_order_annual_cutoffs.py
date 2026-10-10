"""Per-year order size vs later returns, every cutoff and horizon (2026-10-10, EXPLORATION for Abhinav: "(order / tenure) / revenue
vs stock price after 3, 6, 9, 12, 18 months ... 50% hata do, kuch aur number pe bhi uptick aati hai kya?"; no rule changed).

Orders: the cleaned set of EXP-2026-10-08-big-order-annual (order_amounts category 'order', high/medium confidence, no
clarification filings, headline and text agree, unit not inferred; order_terms: awarded or unclear, not a non-order, repeat,
operating contract or roundup; execution period and own share stated). Per-year size = amount x own share (/1.18 if GST is
included) / max(execution years, 1) / trailing revenue known before the filing. Same filters as that test: no bad filing in
the prior 90 days, profitable, price >= Rs 20 and >= Rs 1 cr traded a day on the session before entry, one trade per company
per 30 days. Entry at the open of the first session after the filing.
Outcomes at 63 / 126 / 189 / 252 / 378 sessions (3 / 6 / 9 / 12 / 18 months): return to that close minus 0.5% cost, touched +50%
(max high >= 1.5x entry within the horizon), and the equal-weight average of every priced stock over the same dates (market).
Only trades whose horizon has ended count. Reported for entries 2023+ (where periods are mostly stated) and for all years 2016+.
This is an exploration: execution periods are stated for 54-72% of clean big orders in 2023+ (the registered test's data gate
was NOT READY), so the sized orders are a subset.
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

HZ = {"3m": 63, "6m": 126, "9m": 189, "12m": 252, "18m": 378}
BANDS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, np.inf]
BLAB = ["<5%", "5-10%", "10-15%", "15-20%", "20-30%", "30-50%", "50%+"]
COST, START = 0.005, pd.Timestamp("2016-01-01")
OUT = ROOT / "logs/explorations/order_annual_cutoffs"

D = sp.load(None); cal = D["cal"]
O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
O = O[(O["cat_current"] == "order") & O["amount_confidence"].isin(["high", "medium"])].copy()
O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"]); O = O[O["amount_cr"] > 0]
O["ts"] = pd.to_datetime(O["ts"]); O["act"] = pd.to_datetime(O["d_actionable"])
q = O["headline"].fillna("").str.contains(r"^(Clarification|News Verification|Reply to Clarification)|sought clarification|news item|media report", case=False, regex=True)
O = O[~q & ~(O["headline_fulltext_agree"].astype(str) == "False") & ~O["amount_flags"].fillna("").str.contains("unit_inferred")]
O = O[O["ts"] >= START]
Q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly_enriched.parquet")
Q["quarter_end"] = pd.to_datetime(Q["quarter_end"]); Q["filing_dt"] = pd.to_datetime(Q["filing_dt"])
unit = np.where(Q["source"] == "xbrl", 1e-7, 1e-2)
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
O = O[~(O["ratio"] > 10)].copy()
TM = pd.read_parquet(ROOT / "data/derived/order_terms.parquet", columns=["symbol", "seq_id", "firm", "non_order", "repeat_of", "operating", "exec_months", "own_share", "gst", "aggregate"])
O = O.merge(TM, on=["symbol", "seq_id"], how="left")
O["clean"] = O["firm"].isin(["awarded", "unclear"]) & ~O["non_order"].fillna(False).astype(bool) & O["repeat_of"].isna() \
    & ~O["operating"].fillna(False).astype(bool) & ~O["aggregate"].fillna(False).astype(bool)
O["sized"] = O["clean"] & O["exec_months"].notna() & O["own_share"].notna()
O["ratio_annual"] = O["amount_cr"] * O["own_share"] / np.where(O["gst"] == "incl", 1.18, 1.0) / np.maximum(O["exec_months"] / 12, 1.0) / O["rev_ttm_cr"]
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "direction"])
Lb = L[L["direction"] < 0].groupby("symbol")["filed_at"].apply(lambda s: np.sort(s.values)).to_dict()
O["bad90"] = [bool(len(a := Lb.get(s, np.array([], dtype="datetime64[ns]")))) and
              ((a < np.datetime64(t)) & (a >= np.datetime64(t - pd.Timedelta(days=90)))).any() for s, t in zip(O["symbol"], O["ts"])]
O["i0"] = [cal.searchsorted(a) if a <= cal[-1] else -1 for a in O["act"]]
O = O[O["i0"] > 0].copy()
PR = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
PR["raw_close"] = rp.raw_price(PR, "close"); PR["adv"] = PR["avg_traded_value_20d"] / 1e7
prev = PR.set_index(["symbol", "trade_date"])[["raw_close", "adv"]]; del PR
look = [prev.loc[(s, cal[i - 1])] if (s, cal[i - 1]) in prev.index else pd.Series({"raw_close": np.nan, "adv": np.nan}) for s, i in zip(O["symbol"], O["i0"])]
O["price_prev"] = [x["raw_close"] for x in look]; O["adv_prev_cr"] = [x["adv"] for x in look]
O = O[~O["bad90"] & (O["pat_ttm_cr"] > 0) & (O["price_prev"] >= 20) & (O["adv_prev_cr"] >= 1) & O["rev_ttm_cr"].notna()]


def one_per_30d(df):
    keep, last = [], {}
    for r in df.sort_values("ts").itertuples():
        if r.symbol in last and (r.ts - last[r.symbol]).days < 30:
            continue
        last[r.symbol] = r.ts; keep.append(r.Index)
    return df.loc[keep]


S = one_per_30d(O[O["sized"]]).copy()
U = one_per_30d(O[O["clean"] & ~O["sized"]]).copy()
PX = rp.load_panel(["open", "high", "close"]); Ow, Hw, Cw = (rp.wide(PX, c, cal) for c in ("open", "high", "close")); del PX
On, Hn, Cn = Ow.to_numpy(), Hw.to_numpy(), Cw.ffill(limit=300).to_numpy(); col = {s: i for i, s in enumerate(Ow.columns)}
mk = {}


def market(i0, h):
    if (i0, h) not in mk:
        e, x = On[i0], Cn[i0 + h - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
        mk[(i0, h)] = float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))
    return mk[(i0, h)]


def outcomes(df):
    rows = []
    for r in df.itertuples():
        c = col.get(r.symbol)
        if c is None or not np.isfinite(On[r.i0, c]) or On[r.i0, c] <= 0:
            continue
        e = On[r.i0, c]
        for lab, h in HZ.items():
            if r.i0 + h > len(cal) or not np.isfinite(Cn[r.i0 + h - 1, c]):
                continue
            rows.append(dict(symbol=r.symbol, year=cal[r.i0].year, ratio_annual=getattr(r, "ratio_annual", np.nan), hz=lab, ret=Cn[r.i0 + h - 1, c] / e - 1 - COST,
                             hit=bool(np.nanmax(Hn[r.i0:r.i0 + h, c]) >= 1.5 * e), mkt=market(r.i0, h)))
    return pd.DataFrame(rows)


RS, RU = outcomes(S), outcomes(U)
RS["band"] = pd.cut(RS["ratio_annual"], BANDS, labels=BLAB, right=False)
OUT.mkdir(parents=True, exist_ok=True)
tables = {}
for per, keep in (("2023+", lambda d: d["year"] >= 2023), ("all years 2016+", lambda d: d["year"] >= 2016)):
    rs, ru = RS[keep(RS)], RU[keep(RU)]
    print(f"\n######## entries {per} · orders with a stated period: {rs[rs.hz == '3m'].shape[0]} trades · without a period (clean): {ru[ru.hz == '3m'].shape[0]}")
    for hz in HZ:
        g = rs[rs.hz == hz]
        t = g.groupby("band", observed=False).apply(lambda x: pd.Series(dict(trades=len(x), stocks=x.symbol.nunique(), avg=x.ret.mean() * 100, median=x.ret.median() * 100,
                                                                       hit50=x.hit.mean() * 100, market=x.mkt.mean() * 100, beat_mkt=(x.ret - x.mkt).mean() * 100,
                                                                       beat_mkt_share=(x.ret > x.mkt).mean() * 100)))
        u = ru[ru.hz == hz]
        t.loc["no period stated"] = [len(u), u.symbol.nunique(), u.ret.mean() * 100, u.ret.median() * 100, u.hit.mean() * 100, u.mkt.mean() * 100, (u.ret - u.mkt).mean() * 100, (u.ret > u.mkt).mean() * 100]
        tables[f"{per}|{hz}"] = t
        print(f"\n  -- {hz} after entry --")
        print("  " + t.round(1).to_string().replace("\n", "\n  "))
    g = rs[rs.hz == "12m"]
    cum = pd.DataFrame({f">= {int(c*100)}%": dict(trades=(g.ratio_annual >= c).sum(), avg=g[g.ratio_annual >= c].ret.mean() * 100, median=g[g.ratio_annual >= c].ret.median() * 100,
                                                 beat_mkt=(g[g.ratio_annual >= c].ret - g[g.ratio_annual >= c].mkt).mean() * 100) for c in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)}).T
    print("\n  cumulative cutoffs at 12 months (per-year size at least ...):")
    print("  " + cum.round(1).to_string().replace("\n", "\n  "))
flat = pd.concat({k: v for k, v in tables.items()}, names=["slice", "band"]).reset_index()
flat.to_csv(OUT / "bands_by_horizon.csv", index=False)
(OUT / "bands_by_horizon.csv.manifest.json").write_text(json.dumps(dict(dataset="bands_by_horizon.csv", producer="src/agentic/explore_order_annual_cutoffs.py", status="EXPLORATION (no rule changed)",
    definitions=__doc__, columns=dict(slice="entry period | horizon", band="per-year order size as a share of trailing revenue; 'no period stated' = clean orders without an execution period",
    trades="trades with that horizon finished", stocks="distinct companies", avg="mean return after 0.5% cost, percent", median="median return, percent", hit50="percent touching +50% within the horizon",
    market="equal-weight average of every priced stock over the same dates, percent", beat_mkt="mean (return - market), percentage points", beat_mkt_share="percent of trades beating the market"),
    updated=datetime.now().isoformat(timespec="seconds")), indent=1))
(OUT / "README.md").write_text("# Per-year order size vs returns at 3-18 months (exploration)\n\nCleaned order wins with a stated execution period, banded by "
                               "per-year size (order x own share / execution years / trailing revenue), returns at 3/6/9/12/18 months vs the whole market. "
                               "bands_by_horizon.csv; definitions in its manifest. Exploration only.\n")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id="EXP-2026-10-10-order-annual-cutoffs-EXPLORATION", status="EXPLORATION (no rule changed)",
                             producer="src/agentic/explore_order_annual_cutoffs.py", output=str(OUT.relative_to(ROOT)),
                             results={k: v.round(1).reset_index().to_dict(orient="records") for k, v in tables.items() if k.endswith("|12m") or k.endswith("|6m")}), default=str) + "\n")
