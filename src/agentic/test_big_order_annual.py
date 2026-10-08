"""EXP-2026-10-08-big-order-annual (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: size each order win by what it adds to the company's revenue per year — the company's own share (not the
whole joint venture), without GST, spread over the execution period — after dropping non-orders, repeats and orders not yet
awarded (lowest bidder, letter of intent, MoU, rate contract). Does buying stocks with a big per-year order beat the market,
and beat the same rule on the headline amount?
Trades (all arms): order filings of category 'order' with a cleaned amount (as EXP-2026-10-04-big-order-hold: high / medium
confidence, no clarification / news-verification filings, headline and text agree, unit not inferred), trailing revenue and
profit known before the filing (ttm(), consolidated else standalone, 4 quarters), headline ratio <= 10, no bad filing in the
prior 90 days (event ledger direction -1), profitable, price >= Rs 20 and >= Rs 1 cr traded a day on the session before
entry, one trade per company per 30 days (within each arm). Entry at the open of the first session after the filing;
hold 252 sessions (12 months); 0.5% round-trip cost.
Arms:
  RAW15 / RAW50  headline amount / trailing revenue >= 15% / >= 50%  (reference, no verdict)
  ANN15 / ANN50  per-year size >= 15% / >= 50%, where per-year size = amount x own share (/ 1.18 if the amount includes GST)
                 / max(execution years, 1) / trailing revenue; only orders that are awarded (or unclear), not a non-order,
                 not a repeat, not an operating contract / concession (amount covers years of operation), not a roundup of
                 several orders (no single execution period), with an execution
                 period and an own share stated in the filing (data/derived/order_terms.parquet, build_order_terms.py)
Window: entries 2023-01-01 .. the last session whose 12-month hold has ended. Periods: entries 2023-2024 and 2025+.
PASS per ANN arm = ALL of
  (1) both periods: average trade > average equal-weight market return over the same dates, and median trade > 0 (lag 0)
  (2) (1) holds in >= 4 of 5 entry lags (0..4 sessions later)
  (3) average excess over the market (all entries, lag 0) > the RAW arm's at the same cut
  (4) portfolio max drawdown (equal money in each open trade, cash when none) not worse than the market shadow's by > 10 pts
  An arm with fewer than 30 trades in the window gets no verdict (too few to judge).
Before any outcome: trust/data_ready.gate(). Wiring check: the ANN code path with cleaning off, share 1, no GST, 12-month
period reproduces the RAW trades exactly.
SGM_COUNTS_ONLY=1: print the order funnel (no gate, no outcomes) and stop. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>.
SGM_REPRO=1: log nothing.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/agentic")); sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402

EXP_ID = "EXP-2026-10-08-big-order-annual"
OUTDIR = ROOT / "logs/leader_sleeve/big_order_annual"
H, COST, START, MIN_TRADES = 252, 0.005, pd.Timestamp("2023-01-01"), 30
PERIODS = {"2023-2024": (2023, 2024), "2025+": (2025, 2100)}
LAGS = range(5)
REPRO, COUNTS = os.environ.get("SGM_REPRO") == "1", os.environ.get("SGM_COUNTS_ONLY") == "1"

D = sp.load(None); cal = D["cal"]
O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
O = O[(O["cat_current"] == "order") & O["amount_confidence"].isin(["high", "medium"])].copy()
O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"]); O = O[O["amount_cr"] > 0]
O["ts"] = pd.to_datetime(O["ts"]); O["act"] = pd.to_datetime(O["d_actionable"])
q = O["headline"].fillna("").str.contains(r"^(Clarification|News Verification|Reply to Clarification)|sought clarification|news item|media report", case=False, regex=True)
O = O[~q & ~(O["headline_fulltext_agree"].astype(str) == "False") & ~O["amount_flags"].fillna("").str.contains("unit_inferred")]
O = O[O["ts"] >= START - pd.Timedelta(days=40)]
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
ALL = O.copy()                                                          # for the coverage check
O = O[~(O["ratio"] > 10)].copy()
TM = pd.read_parquet(ROOT / "data/derived/order_terms.parquet", columns=["symbol", "seq_id", "firm", "non_order", "repeat_of", "operating", "exec_months",
                                                                        "own_share", "own_share_source", "gst", "jv", "aggregate"])
O = O.merge(TM, on=["symbol", "seq_id"], how="left")
O["clean"] = O["firm"].isin(["awarded", "unclear"]) & ~O["non_order"].fillna(False).astype(bool) & O["repeat_of"].isna() & ~O["operating"].fillna(False).astype(bool) \
    & ~O["aggregate"].fillna(False).astype(bool)
O["sized"] = O["clean"] & O["exec_months"].notna() & O["own_share"].notna()


def annual(df: pd.DataFrame, on: bool = True) -> pd.Series:
    if not on:
        return df["ratio"]
    return df["amount_cr"] * df["own_share"] / np.where(df["gst"] == "incl", 1.18, 1.0) / np.maximum(df["exec_months"] / 12, 1.0) / df["rev_ttm_cr"]


O["ratio_annual"] = annual(O)
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "direction"])
Lb = L[L["direction"] < 0].groupby("symbol")["filed_at"].apply(lambda s: np.sort(s.values)).to_dict()
O["bad90"] = [bool(len(a := Lb.get(s, np.array([], dtype="datetime64[ns]")))) and
              ((a < np.datetime64(t)) & (a >= np.datetime64(t - pd.Timedelta(days=90)))).any() for s, t in zip(O["symbol"], O["ts"])]
O["i0"] = [cal.searchsorted(a) if a <= cal[-1] else -1 for a in O["act"]]
O = O[O["i0"] > 0].copy()
PR = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
PR = PR[PR["trade_date"] >= START - pd.Timedelta(days=60)]
PR["raw_close"] = rp.raw_price(PR, "close"); PR["adv"] = PR["avg_traded_value_20d"] / 1e7
prev = PR.set_index(["symbol", "trade_date"])[["raw_close", "adv"]]; del PR
look = [prev.loc[(s, cal[i - 1])] if (s, cal[i - 1]) in prev.index else pd.Series({"raw_close": np.nan, "adv": np.nan}) for s, i in zip(O["symbol"], O["i0"])]
O["price_prev"] = [x["raw_close"] for x in look]; O["adv_prev_cr"] = [x["adv"] for x in look]
O["eligible"] = ~O["bad90"] & (O["pat_ttm_cr"] > 0) & (O["price_prev"] >= 20) & (O["adv_prev_cr"] >= 1)
LAST_ENTRY = len(cal) - H


def build(mask: pd.Series) -> pd.DataFrame:
    f = O[mask & O["eligible"]].sort_values("ts")
    keep, last = [], {}
    for r in f.itertuples():
        if r.symbol in last and (r.ts - last[r.symbol]).days < 30:
            continue
        last[r.symbol] = r.ts; keep.append(r.Index)
    T = f.loc[keep].copy(); T["date"] = [cal[i] for i in T["i0"]]
    return T[(T["date"] >= START) & (T["i0"] <= LAST_ENTRY)]


ARMS = {"RAW15": O["ratio"] >= 0.15, "RAW50": O["ratio"] >= 0.5,
        "ANN15": O["sized"] & (O["ratio_annual"] >= 0.15), "ANN50": O["sized"] & (O["ratio_annual"] >= 0.5)}
TR = {a: build(m) for a, m in ARMS.items()}

# ---- order funnel (inputs only, no outcomes)
w = O[(O["ts"] >= START) & (O["ratio"] >= 0.15)]
yr = w["ts"].dt.year
fun = pd.DataFrame({"headline >= 15%": w.groupby(yr).size(), "clean (awarded, not repeat / non-order / operating / roundup)": w[w["clean"]].groupby(yr).size(),
                    "clean + period stated": w[w["clean"] & w["exec_months"].notna()].groupby(yr).size(),
                    "clean + period + own share": w[w["sized"]].groupby(yr).size(),
                    "per-year >= 15%": w[w["sized"] & (w["ratio_annual"] >= 0.15)].groupby(yr).size(),
                    "per-year >= 50%": w[w["sized"] & (w["ratio_annual"] >= 0.5)].groupby(yr).size()}).fillna(0).astype(int)
print("=== orders filed 2023+ with headline >= 15% of trailing revenue (before price / profit / bad-news filters) ===\n" + fun.T.to_string())
why = w[~w["clean"]]
print("dropped as not clean: " + " · ".join(f"{k} {v}" for k, v in {
    "not awarded (L1 / LoI / MoU / framework)": int((~why["firm"].isin(["awarded", "unclear"])).sum()), "repeat": int(why["repeat_of"].notna().sum()),
    "non-order": int(why["non_order"].fillna(False).astype(bool).sum()), "operating / concession": int(why["operating"].fillna(False).astype(bool).sum()),
    "roundup of several orders": int(why["aggregate"].fillna(False).astype(bool).sum()),
    "no terms row": int(why["firm"].isna().sum())}.items()))
print("trades in the window (entries 2023-01-01.." + str(cal[LAST_ENTRY].date()) + "): " + " · ".join(f"{a} {len(t)}" for a, t in TR.items()))
if COUNTS:
    raise SystemExit(0)

# ---- wiring check: ANN path with cleaning off, share 1, no GST, 12-month period == RAW
for cut, raw in ((0.15, "RAW15"), (0.5, "RAW50")):
    alt = build(annual(O, on=False) >= cut)
    same = alt[["symbol", "ts", "i0"]].reset_index(drop=True).equals(TR[raw][["symbol", "ts", "i0"]].reset_index(drop=True))
    print(f"wiring check {raw}: {len(alt)} vs {len(TR[raw])} trades · identical: {same}", flush=True)
    if not same:
        raise SystemExit("wiring check FAILED — no result reported")
ref = pd.read_parquet(ROOT / "logs/leader_sleeve/big_order_hold/reference_trades.parquet")
ref = ref[(ref["ts"] >= START) & (ref["i0"] <= LAST_ENTRY)]
ov = len(set(zip(ref["symbol"], ref["ts"])) & set(zip(TR["RAW15"]["symbol"], TR["RAW15"]["ts"])))
print(f"RAW15 vs the Oct-4 reference trades in the window: {ov} shared of {len(ref)} reference / {len(TR['RAW15'])} now (P&L filled since)")

# ---- data-readiness gate (before any outcome)
yrs = range(START.year, cal[-1].year + 1)
Oa = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet", columns=["cat_current", "d", "text_source"])
Oa = Oa[Oa["cat_current"] == "order"]; oy = pd.to_datetime(Oa["d"]).dt.year
Pq = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet", columns=["symbol", "quarter_end", "net_sales"]); Pq["quarter_end"] = pd.to_datetime(Pq["quarter_end"])
p_have = Pq.dropna(subset=["net_sales"]).drop_duplicates(["symbol", "quarter_end"]).groupby(Pq["quarter_end"].dt.year).size()
Cf = pd.read_parquet(ROOT / "data/derived/results_calendar_full.parquet", columns=["symbol", "toDate", "period"])
Cf = Cf[Cf["period"].astype(str).str.lower() == "quarterly"]; Cf["qe"] = pd.to_datetime(Cf["toDate"], format="%d-%b-%Y", errors="coerce")
Ci = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet", columns=["symbol", "period_to"]); Ci["qe"] = pd.to_datetime(Ci["period_to"], format="%d-%b-%Y", errors="coerce")
off = pd.concat([Cf[["symbol", "qe"]], Ci[["symbol", "qe"]]]).dropna().drop_duplicates()
Px = pd.to_datetime(pd.read_parquet(rp.PANEL, columns=["trade_date"])["trade_date"]); bh = dr.bhav_rows()
A = ALL[(ALL["ts"] >= START) & (ALL["ratio"] <= 10)][["ts", "rev_ttm_cr"]].copy()
C = O[(O["ts"] >= START) & O["clean"] & (O["ratio"] >= 0.15)][["ts", "exec_months"]]
checks = [dr.completeness("order filings with text read vs all order filings", Oa[Oa["text_source"].fillna("none") != "none"].groupby(oy).size(), Oa.groupby(oy).size(), yrs),
          dr.completeness("live P&L company-quarters vs NSE's results calendar + integrated filings", p_have, off.groupby(off["qe"].dt.year).size(), range(START.year - 1, cal[-1].year + 1)),
          dr.completeness("price panel stock-days vs NSE's own bhavcopy (EQ/BE/BZ rows per session)", Px.groupby(Px.dt.year).size(), bh.groupby(bh.index.year).sum(), yrs),
          dr.coverage("trailing revenue known at the filing (cleaned order amounts)", A, "ts", ["rev_ttm_cr"], yrs),
          dr.coverage("execution period stated, clean orders with headline >= 15% of revenue", C, "ts", ["exec_months"], yrs),
          dr.span("event ledger (bad-news filter)", L["filed_at"], START - pd.Timedelta(days=90), cal[-1] - pd.Timedelta(days=10)),
          dr.span("order terms", pd.to_datetime(pd.read_parquet(ROOT / "data/derived/order_terms.parquet", columns=["ts"])["ts"]), START, cal[-1] - pd.Timedelta(days=10))]
tag = os.environ.get("SGM_RERUN_TAG")
if not REPRO:
    dr.gate(EXP_ID, checks, run=tag or "RESULT")

# ---- outcomes
PX = rp.load_panel(["open", "close"]); Ow = rp.wide(PX, "open", cal); Craw = rp.wide(PX, "close", cal); del PX
Cw = Craw.ffill(limit=300); On, Cn = Ow.to_numpy(), Cw.to_numpy(); col = {s: i for i, s in enumerate(Ow.columns)}
mk: dict = {}


def market(i0: int) -> float:
    if i0 not in mk:
        e, x = On[i0], Cn[i0 + H - 1]; ok = np.isfinite(e) & (e > 0) & np.isfinite(x)
        mk[i0] = float(np.mean(np.clip(x[ok] / e[ok] - 1, -1, 10)))
    return mk[i0]


def trades(T: pd.DataFrame, lag: int) -> pd.DataFrame:
    rows = []
    for r in T.itertuples():
        i0 = r.i0 + lag
        if i0 + H > len(cal) or r.symbol not in col:
            continue
        e, x = On[i0, col[r.symbol]], Cn[i0 + H - 1, col[r.symbol]]
        if np.isfinite(e) and e > 0 and np.isfinite(x):
            rows.append(dict(symbol=r.symbol, i0=i0, date=cal[i0], year=cal[i0].year, ret=x / e - 1 - COST, mkt=market(i0)))
    return pd.DataFrame(rows)


def periods_ok(R: pd.DataFrame) -> tuple[bool, dict]:
    out, ok = {}, True
    for name, (a, b) in PERIODS.items():
        g = R[(R.year >= a) & (R.year <= b)] if len(R) else R
        good = bool(len(g) > 0 and g.ret.mean() > g.mkt.mean() and g.ret.median() > 0)
        ok &= good
        out[name] = dict(trades=len(g), avg=round(float(g.ret.mean()), 4) if len(g) else None, median=round(float(g.ret.median()), 4) if len(g) else None,
                         market_avg=round(float(g.mkt.mean()), 4) if len(g) else None, up=round(float((g.ret > 0).mean()), 3) if len(g) else None, ok=good)
    return ok, out


def nav(R: pd.DataFrame) -> tuple[float, float, float, float]:
    rs, no = np.zeros(len(cal)), np.zeros(len(cal))
    for r in R.itertuples():
        c = col[r.symbol]; a = r.i0; b = a + H; e = On[a, c]
        path = Cn[a:b, c].astype(float); prv = np.concatenate([[e], path[:-1]])
        d_ = np.where(np.isfinite(path) & np.isfinite(prv) & (prv > 0), path / prv - 1, 0.0); d_[0] -= COST
        rs[a:b] += d_; no[a:b] += 1
    mkt_d = (Craw / Craw.shift(1) - 1).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy()
    i_s, i_e = int(cal.searchsorted(START)), int(np.nonzero(no)[0].max()) + 1
    arm = np.cumprod(1 + np.where(no > 0, rs / np.maximum(no, 1), 0.0)[i_s:i_e]); sh = np.cumprod(1 + np.where(no > 0, mkt_d, 0.0)[i_s:i_e])
    yrs_ = (cal[i_e - 1] - cal[i_s]).days / 365.25
    dd = lambda n: float((n / np.maximum.accumulate(n) - 1).min())  # noqa: E731
    return arm[-1] ** (1 / yrs_) - 1, dd(arm), sh[-1] ** (1 / yrs_) - 1, dd(sh)


res, verdict = {}, []
for a, T in TR.items():
    lag = {k: periods_ok(R := trades(T, k)) + (R,) for k in LAGS}
    ok0, per0, R0 = lag[0]
    n_lags = sum(v[0] for v in lag.values())
    exc = float((R0.ret - R0.mkt).mean()) if len(R0) else np.nan
    cg, dd, scg, sdd = nav(R0) if len(R0) else (np.nan,) * 4
    res[a] = dict(periods=per0, lags_passing=f"{n_lags}/5", excess=round(exc, 4), portfolio=dict(cagr=round(cg, 4), maxdd=round(dd, 4)),
                  shadow=dict(cagr=round(scg, 4), maxdd=round(sdd, 4)), trades=len(R0), checks=dict(periods=ok0, lags=n_lags >= 4, drawdown=bool(dd >= sdd - 0.10)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if not REPRO and len(R0):
        R0.assign(arm=a).to_csv(OUTDIR / f"trades_{a}.csv", index=False)
for a in ("ANN15", "ANN50"):
    raw = "RAW15" if a == "ANN15" else "RAW50"
    res[a]["checks"]["beats_raw"] = bool(res[a]["excess"] > res[raw]["excess"])
    res[a]["passed"] = all(res[a]["checks"].values()) and res[a]["trades"] >= MIN_TRADES
    if res[a]["trades"] < MIN_TRADES:
        res[a]["note"] = f"fewer than {MIN_TRADES} trades: no verdict"
    if res[a]["passed"]:
        verdict.append(a)
print(f"\n=== {EXP_ID} · entries {START.date()}..{cal[LAST_ENTRY].date()} · 12-month hold · after 0.5% cost ===")
for a, r in res.items():
    print(f"{a}: {r['trades']} trades · avg excess over market {r['excess']:+.1%} · lags {r['lags_passing']} · portfolio {r['portfolio']['cagr']:+.1%}/yr, worst fall "
          f"{r['portfolio']['maxdd']:.1%} vs market shadow {r['shadow']['cagr']:+.1%}/yr, {r['shadow']['maxdd']:.1%}" + (f" · checks {r['checks']} -> {'PASS' if r['passed'] else 'FAIL'}" if "passed" in r else " · reference"))
    for p, e in r["periods"].items():
        if e["trades"]:
            print(f"    {p:9s}: {e['trades']:3d} trades · avg {e['avg']:+.0%} vs market {e['market_avg']:+.0%} · median {e['median']:+.0%} · up {e['up']:.0%} · {'ok' if e['ok'] else 'FAIL'}")
print(f"VERDICT: {', '.join(verdict) + ' PASS (provisional: one market period, 2023+)' if verdict else 'no arm passes'}")
if REPRO:
    raise SystemExit(0)
for a in TR:
    if (OUTDIR / f"trades_{a}.csv").exists():
        (OUTDIR / f"trades_{a}.csv.manifest.json").write_text(json.dumps(dict(dataset=f"trades_{a}.csv", producer="src/agentic/test_big_order_annual.py", experiment=EXP_ID,
            columns=dict(symbol="NSE symbol", i0="entry session index", date="entry session", year="entry year", ret="close after 252 sessions / entry open - 1 - 0.005 (fraction)",
                         mkt="average return of every stock priced on the entry day, same dates, clipped to [-100%, +1000%] (fraction)", arm="arm"),
            definitions=__doc__, updated=datetime.now().isoformat(timespec="seconds")), indent=1))
(OUTDIR / "README.md").write_text(f"# {EXP_ID}\n\nBig orders sized per year (own share, ex-GST, spread over the execution period) after cleaning, vs the headline "
                                  f"ratio, entries 2023+, 12-month hold. Verdict: {', '.join(verdict) + ' PASS (provisional)' if verdict else 'no arm passes'}. Trade lists "
                                  "per arm in trades_<arm>.csv (manifests alongside); the order terms come from data/derived/order_terms.parquet.\n")
with (ROOT / "logs/experiments.jsonl").open("a") as fh:
    fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"), verdict=verdict or "no arm passes",
                             arms=res, funnel=fun.to_dict()), default=str) + "\n")
