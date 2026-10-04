"""EXP-2026-10-04-big-order-hold-18-24m (registered in logs/experiments.jsonl before any outcome was computed).

Plain English: buy every company that files a big order win (>= 15% of its last 12 months' sales), any industry, at the
next open, and hold it 18 months (A18) or 24 months (A24). Does that beat the typical stock over the same dates, and does
a plan that buys every such order for a year rarely lose money?
Trades: frozen copy of explore_big_order_winners.py (SGM_HOT=1, SGM_MIN_RATIO=0.15) — cleaned order amounts, revenue and
profit known before the filing, no bad filing in the prior 90 days, profitable, price >= Rs 20, >= Rs 1 cr traded a day,
one trade per company per 30 days. Entry at the open of the first session after the filing; 0.5% round-trip cost.
Window: entries 2019-01-01..2024-12-31 whose full hold has ended; eras 2019-2022 and 2023+.
PASS per arm = ALL of
  (1) both eras: average trade > average matched typical stock, and median trade > 0 (entry lag 0)
  (2) (1) holds in >= 4 of 5 entry lags (0..4 sessions later)
  (3) 12-month buying windows (monthly starts, inside the entry span, equal money per trade) lose money in <= 25%
  (4) portfolio max drawdown (equal money in each open trade, cash when none) not worse than the market shadow's by > 10 pts
Before any outcome: trust/data_ready.gate(). Below the bar -> DATA-READY NOT READY, exit 3, no result.
Wiring check: the trade list must equal logs/leader_sleeve/big_order_hold/reference_trades.parquet (the exploration's).
After the verdict, INFO (not part of the verdict): 2025 buying windows so far.
SGM_GATE_ONLY=1: gate and stop. SGM_RERUN_TAG=<tag>: log -RERUN-<tag>. SGM_REPRO=1: log nothing.
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
sys.path.insert(0, str(ROOT / "src/agentic"))
sys.path.insert(0, str(ROOT / "src/agentic/trust"))
import data_ready as dr  # noqa: E402
import research_panel as rp  # noqa: E402
import sim_leader_portfolio_7x as sp  # noqa: E402

CUT = float(os.environ.get("SGM_CUT", "0.15"))   # 0.5 -> EXP-2026-10-04-big-order-hold-18-24m-50pct (registered variant)
EXP_ID = "EXP-2026-10-04-big-order-hold-18-24m" + ("" if CUT == 0.15 else f"-{CUT * 100:.0f}pct")
SUF = "" if CUT == 0.15 else f"_{CUT * 100:.0f}pct"
OUTDIR = ROOT / "logs/leader_sleeve/big_order_hold"
ARMS = {"A18": 378, "A24": 504}
COST, MIN_RATIO = 0.005, 0.15
START, END_ENTRY = pd.Timestamp("2019-01-01"), pd.Timestamp("2024-12-31")
ERAS = {"2019-2022": (2019, 2022), "2023+": (2023, 2100)}
LAGS = range(5)
REPRO = os.environ.get("SGM_REPRO") == "1"

# ---------------- trades: frozen copy of explore_big_order_winners.py (SGM_HOT=1, SGM_MIN_RATIO=0.15), 2026-10-04
D = sp.load(None); cal = D["cal"]
O = pd.read_parquet(ROOT / "data/derived/order_amounts.parquet")
O = O[(O["cat_current"] == "order") & O["amount_confidence"].isin(["high", "medium"])].copy()
O["amount_cr"] = O["order_amount_cr"].fillna(O["parsed_amount_cr"]); O = O[O["amount_cr"] > 0]
O["ts"] = pd.to_datetime(O["ts"]); O["act"] = pd.to_datetime(O["d_actionable"])
q = O["headline"].fillna("").str.contains(r"^(Clarification|News Verification|Reply to Clarification)|sought clarification|news item|media report", case=False, regex=True)
O = O[~q]
O = O[~(O["headline_fulltext_agree"].astype(str) == "False")]
O = O[~O["amount_flags"].fillna("").str.contains("unit_inferred")]
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
WINS = O.copy()                                                         # cleaned order wins, for the coverage check
O = O[~(O["ratio"] > 10)]
big = O[O["ratio"] >= MIN_RATIO].sort_values("ts").copy()
L = pd.read_parquet(ROOT / "data/derived/event_ledger.parquet", columns=["symbol", "filed_at", "direction"])
Lb = L[L["direction"] < 0].groupby("symbol")["filed_at"].apply(lambda s: np.sort(s.values)).to_dict()
big["bad90"] = [bool(len(a := Lb.get(s, np.array([], dtype="datetime64[ns]")))) and
                ((a < np.datetime64(t)) & (a >= np.datetime64(t - pd.Timedelta(days=90)))).any() for s, t in zip(big["symbol"], big["ts"])]
big["i0"] = [cal.searchsorted(a) if a <= cal[-1] else -1 for a in big["act"]]
big = big[(big["i0"] > 0)]
PR = rp.load_panel(["close", "price_adjustment_factor_to_present", "avg_traded_value_20d"])
PR["raw_close"] = rp.raw_price(PR, "close"); PR["adv"] = PR["avg_traded_value_20d"] / 1e7
prev = PR.set_index(["symbol", "trade_date"])[["raw_close", "adv"]]
del PR
look = [prev.loc[(s, cal[i - 1])] if (s, cal[i - 1]) in prev.index else pd.Series({"raw_close": np.nan, "adv": np.nan}) for s, i in zip(big["symbol"], big["i0"])]
big["price_prev"] = [x["raw_close"] for x in look]; big["adv_prev_cr"] = [x["adv"] for x in look]
f3 = big[~big["bad90"]]; f3 = f3[f3["pat_ttm_cr"] > 0]; f3 = f3[(f3["price_prev"] >= 20) & (f3["adv_prev_cr"] >= 1)]
keep, last = [], {}
for r in f3.sort_values("ts").itertuples():
    if r.symbol in last and (r.ts - last[r.symbol]).days < 30:
        continue
    last[r.symbol] = r.ts; keep.append(r.Index)
T = f3.loc[keep].copy()
imap = sp.industry_maps()["analogs"]                                    # hot group: only for the wiring check
Sx = pd.read_parquet(ROOT / "data/derived/industry_scores_policy.parquet", columns=["date", "industry", "heat_pct"])
Sx["date"] = pd.to_datetime(Sx["date"]); Sx = Sx.sort_values("date")
lk = pd.DataFrame({"industry": T["symbol"].map(imap).values, "date": [cal[i - 1] for i in T["i0"]], "k": T.index}).dropna(subset=["industry"]).sort_values("date")
m = pd.merge_asof(lk, Sx, on="date", by="industry", direction="backward", tolerance=pd.Timedelta(days=10)).set_index("k")
T["heat"] = m["heat_pct"].reindex(T.index)
T["group"] = np.where(T["heat"].isna(), "unknown", np.where(T["heat"] >= 0.70, "hot", "not hot"))

# ---------------- wiring check: same trades as the exploration, exactly
ref = pd.read_parquet(OUTDIR / "reference_trades.parquet")
mine = T[["symbol", "ts", "i0", "ratio", "group"]].reset_index(drop=True)
same = len(ref) == len(mine) and all(ref[c].reset_index(drop=True).equals(mine[c]) for c in ("symbol", "ts", "i0", "group")) \
    and np.allclose(ref["ratio"].to_numpy(), mine["ratio"].to_numpy(), rtol=0, atol=1e-12)
print(f"wiring check: {len(mine)} trades vs reference {len(ref)} · identical: {same}", flush=True)
if not same:
    raise SystemExit("wiring check FAILED — no result reported")

# ---------------- data-readiness gate (before any outcome)
PX = rp.load_panel(["open", "close"]); Ow = rp.wide(PX, "open", cal); Craw = rp.wide(PX, "close", cal); del PX
Cw = Craw.ffill(limit=300)
Tw = T[(T["ts"] >= START) & (T["ts"] <= END_ENTRY + pd.Timedelta(days=1))].copy()
Tw["date"] = [cal[i] for i in Tw["i0"]]; Tw = Tw[(Tw["date"] >= START) & (Tw["date"] <= END_ENTRY) & (Tw["ratio"] >= CUT)]
col = {s: i for i, s in enumerate(Cw.columns)}
On, Cn = Ow.to_numpy(), Cw.to_numpy()
Tw["has_px"] = [1.0 if s in col and np.isfinite(On[i, col[s]]) and On[i, col[s]] > 0 else np.nan for s, i in zip(Tw["symbol"], Tw["i0"])]
A = pd.read_csv(ROOT / "logs/news/announcements_completeness.csv").dropna(subset=["nse"]); A = A[A["nse"] > 0]
yrs = range(START.year, END_ENTRY.year + 1)
last_exit = cal[min(int(Tw["i0"].max()) + max(ARMS.values()) - 1, len(cal) - 1)]
checks = [dr.completeness("NSE announcements we hold vs NSE's own day feed (sample days)", A.groupby("year")["ours"].sum(), A.groupby("year")["nse"].sum(), yrs),
          dr.coverage("trailing revenue and profit known before the filing (cleaned order wins)", WINS, "ts", ["rev_ttm_cr", "pat_ttm_cr"], yrs),
          dr.coverage("entry price available (trades in the window)", Tw, "date", ["has_px"], yrs),
          dr.span("event ledger (bad-filing check)", L["filed_at"], START - pd.Timedelta(days=90), END_ENTRY),
          dr.span("P&L filings", Q["filing_dt"], START - pd.Timedelta(days=400), END_ENTRY),
          dr.span("price calendar", pd.Series(cal), START, cal[-1])]
dr.gate(EXP_ID, checks, run=os.environ.get("SGM_RERUN_TAG") or "RESULT")
if os.environ.get("SGM_GATE_ONLY") == "1":
    raise SystemExit(0)

# ---------------- outcomes
typ_cache: dict = {}


def typical(i0: int, h: int) -> float:
    k = (i0, h)
    if k not in typ_cache:
        e, x = Ow.iloc[i0], Cw.iloc[i0 + h - 1]; r = (x / e - 1).replace([np.inf, -np.inf], np.nan)
        typ_cache[k] = float(r[(e > 0)].median())
    return typ_cache[k]


def trades(h: int, lag: int) -> pd.DataFrame:
    rows = []
    for r in Tw.itertuples():
        i0 = r.i0 + lag
        if i0 + h > len(cal) or r.symbol not in col:
            continue
        e, x = On[i0, col[r.symbol]], Cn[i0 + h - 1, col[r.symbol]]
        if not (np.isfinite(e) and e > 0 and np.isfinite(x)):
            continue
        rows.append(dict(symbol=r.symbol, i0=i0, date=r.date, year=r.date.year, ret=x / e - 1 - COST, typ=typical(i0, h)))
    return pd.DataFrame(rows)


def era_ok(R: pd.DataFrame) -> tuple[bool, dict]:
    out, ok = {}, True
    for name, (a, b) in ERAS.items():
        g = R[(R.year >= a) & (R.year <= b)]
        good = len(g) > 0 and g.ret.mean() > g.typ.mean() and g.ret.median() > 0
        ok &= good
        out[name] = dict(trades=len(g), avg=round(float(g.ret.mean()), 4), median=round(float(g.ret.median()), 4),
                         typical_avg=round(float(g.typ.mean()), 4), up=round(float((g.ret > 0).mean()), 3), ok=bool(good))
    return ok, out


def nav(R: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    ret_sum = np.zeros(len(cal)); n_open = np.zeros(len(cal))
    for r in R.itertuples():
        c = col[r.symbol]; a = r.i0; b = a + h; e = On[a, c]
        path = Cn[a:b, c].astype(float); prv = np.concatenate([[e], path[:-1]])
        dr_ = np.where(np.isfinite(path) & np.isfinite(prv) & (prv > 0), path / prv - 1, 0.0); dr_[0] -= COST
        ret_sum[a:b] += dr_; n_open[a:b] += 1
    arm = np.where(n_open > 0, ret_sum / np.maximum(n_open, 1), 0.0)
    mkt_d = (Craw / Craw.shift(1) - 1).clip(-0.5, 1.0).mean(axis=1).fillna(0).to_numpy()
    shadow = np.where(n_open > 0, mkt_d, 0.0)
    i_s, i_e = int(cal.searchsorted(START)), int(np.nonzero(n_open)[0].max()) + 1
    idx = cal[i_s:i_e]
    return pd.Series(np.cumprod(1 + arm[i_s:i_e]), index=idx), pd.Series(np.cumprod(1 + shadow[i_s:i_e]), index=idx)


def stats(n: pd.Series) -> dict:
    yrs_ = (n.index[-1] - n.index[0]).days / 365.25
    return dict(cagr=round(float(n.iloc[-1] ** (1 / yrs_) - 1), 4), maxdd=round(float((n / n.cummax() - 1).min()), 4))


results, verdict = {}, {}
for arm, h in ARMS.items():
    lag_res = {}
    for lag in LAGS:
        R = trades(h, lag); ok, eras = era_ok(R); lag_res[lag] = (ok, eras, R)
    ok0, eras0, R0 = lag_res[0]
    n_lags = sum(v[0] for v in lag_res.values())
    win = []
    for st in pd.date_range(START, END_ENTRY, freq="MS"):
        en = st + pd.DateOffset(months=12)
        if en > END_ENTRY + pd.Timedelta(days=1) or int(cal.searchsorted(en)) + h > len(cal):
            continue
        g = R0[(R0.date >= st) & (R0.date < en)]
        if len(g):
            win.append(dict(start=st, trades=len(g), avg=g.ret.mean(), typ=g.typ.mean()))
    W = pd.DataFrame(win)
    lost = float((W.avg < 0).mean())
    na, ns = nav(R0, h); sa, ss = stats(na), stats(ns)
    p1, p2, p3, p4 = ok0, n_lags >= 4, lost <= 0.25, sa["maxdd"] >= ss["maxdd"] - 0.10
    verdict[arm] = bool(p1 and p2 and p3 and p4)
    results[arm] = dict(eras=eras0, lags_passing=f"{n_lags}/5", windows=len(W), windows_lost=round(lost, 3),
                        window_median=round(float(W.avg.median()), 4), window_worst=round(float(W.avg.min()), 4),
                        worst_start=str(W.loc[W.avg.idxmin(), "start"].date()), portfolio=sa, shadow=ss,
                        checks=dict(eras=p1, lags=p2, windows=p3, drawdown=p4), passed=verdict[arm])
    print(f"\n=== {arm} · hold {h} sessions · entries {START.date()}..{END_ENTRY.date()} with a full hold ===")
    for name, e in eras0.items():
        print(f"  {name:9s}: {e['trades']:3d} trades · avg {e['avg']:+.0%} vs typical stock {e['typical_avg']:+.0%} · median {e['median']:+.0%} · up {e['up']:.0%} · {'ok' if e['ok'] else 'FAIL'}")
    print("  entry lags 0-4: " + " · ".join(f"lag {k} {'ok' if v[0] else 'FAIL'} ({v[1]['2019-2022']['avg']:+.0%}/{v[1]['2023+']['avg']:+.0%})" for k, v in lag_res.items()) + f" -> {n_lags}/5")
    print(f"  12-month buying windows: {len(W)} · lost money in {lost:.0%} · median {W.avg.median():+.0%} · worst {W.avg.min():+.0%} "
          f"(start {W.loc[W.avg.idxmin(), 'start']:%b %Y}) · best {W.avg.max():+.0%}")
    print(f"  portfolio {START.date()}..{na.index[-1].date()}: {sa['cagr']:+.1%}/yr, worst fall {sa['maxdd']:.1%} · market shadow {ss['cagr']:+.1%}/yr, worst fall {ss['maxdd']:.1%}")
    print(f"  checks: eras {'✅' if p1 else '❌'} · lags {'✅' if p2 else '❌'} · windows {'✅' if p3 else '❌'} · drawdown {'✅' if p4 else '❌'} -> {'PASS' if verdict[arm] else 'FAIL'}")
    if not REPRO:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        R0.assign(arm=arm).to_csv(OUTDIR / f"trades_{arm}{SUF}.csv", index=False)
        (OUTDIR / f"trades_{arm}{SUF}.csv.manifest.json").write_text(json.dumps(dict(
            dataset=f"trades_{arm}{SUF}", producer="src/agentic/test_big_order_hold.py", experiment=EXP_ID, rows=len(R0),
            columns=dict(symbol="NSE symbol", i0="entry session index", date="entry session", year="entry year",
                         ret=f"close after {h} sessions / entry open - 1 - 0.005 (fraction)", typ="median return of every stock priced on the entry day, same dates (fraction)"),
            updated=datetime.now().isoformat(timespec="seconds")), indent=1))

tag = os.environ.get("SGM_RERUN_TAG")
passed = [a for a, v in verdict.items() if v]
print(f"\nVERDICT: {', '.join(passed) + ' PASS' if passed else 'no arm passes'}")
if not REPRO:
    with (ROOT / "logs/experiments.jsonl").open("a") as fh:
        fh.write(json.dumps(dict(ts=datetime.now().isoformat(timespec="seconds"), id=EXP_ID + (f"-RERUN-{tag}" if tag else "-RESULT"),
                                 verdict=passed or "no arm passes", arms=results), default=str) + "\n")

# ---------------- INFO, not part of the verdict: 2025 buying windows so far (2025 revenue known for fewer orders: thin)
print(f"\nINFO · orders >= {CUT:.0%} of revenue · buying windows starting in 2025, valued on {cal[-1].date()} · 12m hold: each trade at its 12-month exit if passed, else today · equal money per trade")
lastc = len(cal) - 1
for st in pd.date_range("2025-01-01", "2025-10-01", freq="MS"):
    en = st + pd.DateOffset(months=12)
    g = T[(T["ts"] >= st) & (T["ts"] < en) & (T["i0"] <= lastc) & (T["ratio"] >= CUT)]
    rr, tt, still = [], [], 0
    for r in g.itertuples():
        if r.symbol not in col:
            continue
        x_i = min(r.i0 + 252, len(cal)) - 1; e, x = On[r.i0, col[r.symbol]], Cn[x_i, col[r.symbol]]
        if np.isfinite(e) and e > 0 and np.isfinite(x):
            rr.append(x / e - 1 - COST); tt.append(typical(r.i0, x_i - r.i0 + 1)); still += r.i0 + 252 > len(cal)
    if rr:
        print(f"  {st:%b %Y}: {len(rr):3d} trades ({still} still inside their 12 months) · avg {np.mean(rr):+.0%} · median {np.median(rr):+.0%} · "
              f"up {np.mean(np.array(rr) > 0):.0%} · typical stock {np.mean(tt):+.0%}")
print(f"\nINFO · orders >= {CUT:.0%} of revenue bought in each month of 2025, each held 12 months (valued today if not yet finished)")
B = T[(T["ratio"] >= CUT)].copy(); B["buy"] = [cal[i] for i in B["i0"]]; B = B[(B.buy >= "2025-01-01") & (B.buy < "2026-01-01")]
mrows = []
for r in B.itertuples():
    if r.symbol not in col:
        continue
    e = On[r.i0, col[r.symbol]]; x_i = min(r.i0 + 252, len(cal)) - 1
    if np.isfinite(e) and e > 0 and np.isfinite(Cn[x_i, col[r.symbol]]):
        mrows.append(dict(m=r.buy.strftime("%b %Y"), k=r.buy.month, ret=Cn[x_i, col[r.symbol]] / e - 1 - COST, typ=typical(r.i0, x_i - r.i0 + 1), done=r.i0 + 252 <= len(cal), sym=r.symbol))
MB = pd.DataFrame(mrows)
for k, g in MB.groupby("k"):
    print(f"  {g.m.iloc[0]}: {len(g):3d} trades · finished {int(g.done.sum()):3d} · avg {g.ret.mean():+.0%} · median {g.ret.median():+.0%} · up {(g.ret > 0).mean():.0%} · typical stock {g.typ.mean():+.0%}"
          + (" · " + ", ".join(f"{s} {v:+.0%}" for s, v in zip(g.sym, g.ret)) if CUT >= 0.5 else ""))
g = MB[MB.done]
print(f"  all finished: {len(g)} · avg {g.ret.mean():+.0%} · median {g.ret.median():+.0%} · up {(g.ret > 0).mean():.0%} · typical stock {g.typ.mean():+.0%}")
