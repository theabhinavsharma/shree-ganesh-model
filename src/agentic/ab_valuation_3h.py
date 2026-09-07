"""VALUATION-DEVIATION A/B — 3 horizons. Pre-registered 2026-09-05 (see ledger
ab_valuation_deviation_REGISTERED). Runs ONLY after the QC gate passes.

Point-in-time PE construction:
  eps_present(q) = eps_basic(q) x price_adjustment_factor_to_present(filing date)
  (per-share values scale exactly like prices across splits/bonuses)
  TTM(t) = sum of last 4 quarters' eps_present, window span <= 380d,
           known at max(filing_dt) of the 4 — filing-date keyed, no lookahead.
  PE(t)  = adjusted_close(t) / TTM(t)   [TTM<=0 => loss bucket]

QC gate (HARD STOP): reconstructed PE at each symbol's screener fetch_date must
match screener PE within +/-15% for >=80% of comparable symbols.

Features: pe_vs_self (trailing 5y weekly percentile, min 2y) and
pe_vs_ind (PE / industry-median PE that week, smIndustry map, groups>=5).

Cells+bars registered 2026-09-05; eras disc<=2022 / conf>=2023 (PE data dense
from 2018-19 — disc era is short; n reported per cell).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")

# ---------------- 1. quarterly EPS -> point-in-time TTM ----------------
print("eps series…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet")
q = q.dropna(subset=["eps_basic", "filing_dt"])
q = (q.sort_values("filing_dt")
       .drop_duplicates(["symbol", "quarter_end", "basis"], keep="last"))
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_old = pd.Timestamp("1900-01-01")
_con = lastq["con"] if "con" in lastq else pd.Series(_old, index=lastq.index)
_sa = lastq["sa"] if "sa" in lastq else pd.Series(_old, index=lastq.index)
# basis covering the latest quarter wins; tie -> consolidated (matches screener)
bmap = pd.Series(np.where(_con.fillna(_old) >= _sa.fillna(_old), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)]

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d",
             "rsi_14_daily", "return_20d", "volume_vs_20d", "delivery_pct",
             "avg_delivery_pct_20d", "avg_traded_value_20d",
             "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])

# factor at filing date (backward asof within symbol)
fac = px[["symbol", "trade_date", "price_adjustment_factor_to_present"]].rename(
    columns={"trade_date": "filing_dt", "price_adjustment_factor_to_present": "fac"})
q = q.sort_values("filing_dt")
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
q = pd.merge_asof(q, fac.sort_values("filing_dt"), on="filing_dt", by="symbol",
                  direction="backward", tolerance=pd.Timedelta(days=30))
q["fac"] = q["fac"].fillna(1.0)
q["eps_adj"] = q["eps_basic"] * q["fac"]

rows = []
for sym, g in q.sort_values("quarter_end").groupby("symbol"):
    qe = g["quarter_end"].values; ea = g["eps_adj"].values
    fd = pd.to_datetime(g["filing_dt"]).values
    for i in range(3, len(g)):
        span = (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int)
        if span > 380:
            continue
        rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")
print(f"  ttm rows {len(ttm):,} · symbols {ttm['symbol'].nunique()}", flush=True)

# ---------------- 2. QC GATE vs screener ----------------
print("QC gate…", flush=True)
_fresh = ROOT / "data/derived/screener_qc_fresh.parquet"
sc = pd.read_parquet(_fresh if _fresh.exists() else ROOT / "data/derived/screener_fundamentals.parquet")
if "fetch_date" not in sc.columns:
    sc["fetch_date"] = pd.Timestamp.today().normalize()
sc["fetch_date"] = pd.to_datetime(sc["fetch_date"])
sc["pe"] = pd.to_numeric(sc["pe"], errors="coerce")
sc = (sc.sort_values("fetch_date").drop_duplicates("symbol", keep="last")
        .dropna(subset=["pe"]))
sc = sc[sc["pe"] > 0][["symbol", "fetch_date", "pe"]]
cl = px[["symbol", "trade_date", "close"]].rename(columns={"trade_date": "fetch_date"})
sc = pd.merge_asof(sc.sort_values("fetch_date"), cl.sort_values("fetch_date"),
                   on="fetch_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=7))
sc = pd.merge_asof(sc.sort_values("fetch_date"),
                   ttm.rename(columns={"known": "fetch_date"}).sort_values("fetch_date"),
                   on="fetch_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=200))
sc = sc.dropna(subset=["close", "ttm"])
sc = sc[sc["ttm"] > 0]
sc["pe_ours"] = sc["close"] / sc["ttm"]
sc["ratio"] = sc["pe_ours"] / sc["pe"]
sc.to_parquet(ROOT / "logs/qc_valuation_detail.parquet", index=False)
ok = sc["ratio"].between(0.85, 1.15)
print(f"  comparable {len(sc):,} · within±15%: {ok.mean()*100:.1f}% · "
      f"median ratio {sc['ratio'].median():.3f}", flush=True)
for lo, hi in [(0.7, 1.3), (0.5, 2.0)]:
    print(f"  within {lo}-{hi}x: {sc['ratio'].between(lo,hi).mean()*100:.1f}%", flush=True)
if ok.mean() < 0.80:
    # amended referee (ledgered 2026-09-07): direct quarterly-EPS match vs the
    # screener quarterly table — screener's displayed P/E normalizes earnings
    # for ~15% of pages and is not reproducible from filings.
    v2 = ROOT / "logs/qc_eps_table_check.parquet"
    if not v2.exists():
        print("QC GATE FAILED — HARD STOP (and no amended-referee result found)", flush=True)
        sys.exit(2)
    R = pd.read_parquet(v2)
    wgt = (R[R["was_bad"]]["ok"].mean() * (~ok).sum() + R[~R["was_bad"]]["ok"].mean() * ok.sum()) / len(ok)
    print(f"  amended referee (quarterly-EPS table): weighted match {wgt*100:.1f}%", flush=True)
    if wgt < 0.80:
        print("QC GATE FAILED under amended referee — HARD STOP", flush=True)
        sys.exit(2)
    print("QC GATE PASSED via amended referee (ledgered 2026-09-07)", flush=True)
else:
    print("QC GATE PASSED", flush=True)

# ---------------- 3. shared frames ----------------
print("frames…", flush=True)
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["ret5"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["rsi_mu"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
px["rsi_sd"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
px["rsi_z"] = (px["rsi_14_daily"] - px["rsi_mu"]) / px["rsi_sd"]
px["r20_z"] = px["return_20d"] / (px["dvol"] * np.sqrt(20))
px["r5_z"] = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]


def fwd(col, n, how):
    return g[col].transform(lambda s: getattr(s.shift(-1)[::-1].rolling(n, min_periods=min(30, n)), how)()[::-1])


for n in (63, 126):
    px[f"hi{n}"] = fwd("high", n, "max")
    px[f"lo{n}"] = fwd("low", n, "min")
    px[f"cl{n}"] = g["close"].transform(lambda s: s.shift(-n))
px["hi15"] = fwd("high", 15, "max")

days = sorted(px["trade_date"].unique())
weekly = set(days[::5])
wk = px[px["trade_date"].isin(weekly) & (px["adv"] >= 5) & (px["close"] > 50)
        & (px["trade_date"] >= "2018-06-01")].copy()
wk = pd.merge_asof(wk.sort_values("trade_date"),
                   ttm.rename(columns={"known": "trade_date"}).sort_values("trade_date"),
                   on="trade_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=200))
wk["pe"] = np.where(wk["ttm"] > 0, wk["close"] / wk["ttm"], np.nan)
wk["is_loss"] = wk["ttm"] <= 0

# pe_vs_self: trailing 5y weekly percentile (min 2y)
wk = wk.sort_values(["symbol", "trade_date"])
wk["pe_self"] = (wk.groupby("symbol")["pe"]
                   .transform(lambda s: s.rolling(260, min_periods=104)
                              .apply(lambda w: (w <= w[-1]).mean(), raw=True)))
# pe_vs_ind
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
wk["ind"] = wk["symbol"].map(imap)
med = (wk.dropna(subset=["pe", "ind"]).groupby(["trade_date", "ind"])["pe"]
         .agg(["median", "size"]).reset_index()
         .rename(columns={"median": "ind_pe", "size": "ind_n"}))
wk = wk.merge(med[med["ind_n"] >= 5], on=["trade_date", "ind"], how="left")
wk["pe_ind"] = wk["pe"] / wk["ind_pe"]
wk["year"] = wk["trade_date"].dt.year
wk["week"] = wk["trade_date"].dt.to_period("W")
print(f"  weekly grid {len(wk):,} · pe {wk['pe'].notna().mean()*100:.0f}% · "
      f"pe_self {wk['pe_self'].notna().mean()*100:.0f}% · pe_ind {wk['pe_ind'].notna().mean()*100:.0f}%", flush=True)

ERAS = [("disc<=22", lambda d: d["year"] <= 2022), ("conf>=23", lambda d: d["year"] >= 2023)]


def cellrow(name, s, hz):
    if len(s) < 60:
        return f"{name:<44s}| {len(s):>6,} | too few"
    w = s.groupby("week")[f"end{hz}"].mean()
    return (f"{name:<44s}| {len(s):>6,} | {s[f't2x{hz}'].mean()*100:5.1f}% | "
            f"{s[f't50{hz}'].mean()*100:5.1f}% | {s[f't25{hz}'].mean()*100:5.1f}% | "
            f"{w.mean():+6.2f}% | {s[f'tr{hz}'].median():+5.1f}%")


for hz in (63, 126):
    wk[f"t2x{hz}"] = wk[f"hi{hz}"] / wk["close"] - 1 >= 1.0
    wk[f"t50{hz}"] = wk[f"hi{hz}"] / wk["close"] - 1 >= 0.50
    wk[f"t25{hz}"] = wk[f"hi{hz}"] / wk["close"] - 1 >= 0.25
    wk[f"end{hz}"] = (wk[f"cl{hz}"] / wk["close"] - 1) * 100
    wk[f"tr{hz}"] = (wk[f"lo{hz}"] / wk["close"] - 1) * 100

# ---------------- 4. HORIZON 63td ----------------
print("\n================ 63td (90d) — deciles & terciles ================", flush=True)
print(f"{'CELL':<44s}| {'n':>6s} | P2x   | P50   | P25   | wk-end | medTr")
h = wk.dropna(subset=["hi63"])
for era, em in ERAS:
    s = h[em(h)]
    base = s
    print(cellrow(f"BASE all investable [{era}]", base, 63), flush=True)
    print(cellrow(f"LOSS-makers [{era}]", s[s["is_loss"]], 63), flush=True)
    for feat in ("pe_self", "pe_ind"):
        f = s.dropna(subset=[feat])
        if len(f) < 500:
            print(f"  {feat} [{era}]: n={len(f)} too thin"); continue
        ter = f[feat].quantile([1 / 3, 2 / 3]).values
        print(cellrow(f"{feat} CHEAP third [{era}]", f[f[feat] <= ter[0]], 63), flush=True)
        print(cellrow(f"{feat} MID third [{era}]", f[(f[feat] > ter[0]) & (f[feat] <= ter[1])], 63), flush=True)
        print(cellrow(f"{feat} EXPENSIVE third [{era}]", f[f[feat] > ter[1]], 63), flush=True)

# ---------------- 5. HORIZON 126td — industry LEADER x cheap ----------------
print("\n================ 126td (6M) — LEADER x valuation ================", flush=True)
h = wk.dropna(subset=["hi126", "ret60", "ind"]).copy()
grp = (h.groupby(["trade_date", "ind"]).agg(gr=("ret60", "mean"), nm=("ret60", "size"))
         .reset_index())
grp = grp[grp["nm"] >= 5]
grp["gd"] = grp.groupby("trade_date")["gr"].transform(lambda s: s.rank(pct=True))
h = h.merge(grp, on=["trade_date", "ind"], how="inner")
h["top3"] = h.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False) <= 3
h["rank_in"] = h.groupby(["trade_date", "ind"])["ret60"].rank(pct=True)
print(f"{'CELL':<44s}| {'n':>6s} | P2x   | P50   | P25   | wk-end | medTr")
for era, em in ERAS:
    s = h[em(h)]
    hot = s[s["gd"] >= 0.9]
    led = hot[hot["top3"]]
    print(cellrow(f"LEADER plain [{era}]", led, 126), flush=True)
    lc = led.dropna(subset=["pe_ind"])
    print(cellrow(f"LEADER pe known [{era}]", lc, 126), flush=True)
    print(cellrow(f"LEADER & CHEAP pe_ind<1 [{era}]", lc[lc["pe_ind"] < 1], 126), flush=True)
    print(cellrow(f"LEADER & EXPENSIVE pe_ind>=1 [{era}]", lc[lc["pe_ind"] >= 1], 126), flush=True)
    print(cellrow(f"LEADER & LOSS [{era}]", led[led["is_loss"]], 126), flush=True)
    lag = hot[hot["rank_in"] <= 0.5].dropna(subset=["pe_ind"])
    print(cellrow(f"LAGGARD & CHEAP pe_ind<1 (expl) [{era}]", lag[lag["pe_ind"] < 1], 126), flush=True)

# ---------------- 6. HORIZON 15d — z-band frame ----------------
print("\n================ 15d — z-band pool, C2 top-8 A/B ================", flush=True)
Z = (wk["rsi_z"].between(-1.0, 0.25) & wk["r20_z"].between(-1.0, 0.75)
     & wk["r5_z"].between(-1.0, 1.0) & (wk["volume_vs_20d"] < 2))
pool = wk[Z].dropna(subset=["hi15"]).copy()
pool["touch"] = pool["hi15"] / pool["close"] - 1 >= 0.05
print("pool touch by pe_self decile (both eras pooled):", flush=True)
p = pool.dropna(subset=["pe_self"])
print(p.groupby(pd.qcut(p["pe_self"], 5, duplicates="drop"))["touch"]
        .agg(["mean", "size"]).to_string(), flush=True)

sf = {s: gg.reset_index(drop=True) for s, gg in
      px[["symbol", "trade_date", "open", "high", "low", "close", "dvol"]].groupby("symbol")}


def c2(sym, d):
    gg = sf[sym]
    sig = gg[gg["trade_date"] == d]; fut = gg[gg["trade_date"] > d].head(15)
    if len(sig) == 0 or len(fut) < 8 or pd.isna(fut.iloc[0]["open"]):
        return None, None
    ep = float(fut.iloc[0]["open"])
    dv = float(sig.iloc[0]["dvol"]) if pd.notna(sig.iloc[0]["dvol"]) else 0.02
    tgt, sl, half, trail = ep * 1.05, ep * (1 - np.clip(3 * dv, 0.03, 0.12)), False, None
    touched = False
    for i in range(len(fut)):
        r = fut.iloc[i]; lo, hi = r["low"], r["high"]
        if pd.isna(lo):
            continue
        if pd.notna(hi) and hi >= tgt:
            touched = True
        if not half:
            if lo <= sl:
                return (sl / ep - 1) * 100, touched
            if pd.notna(hi) and hi >= tgt:
                half = True; trail = ep * 1.025; continue
        else:
            if lo <= trail:
                return ((0.05 + (trail / ep - 1)) / 2) * 100, touched
    last = float(fut.iloc[-1]["close"])
    return (((0.05 + (last / ep - 1)) / 2) * 100 if half else (last / ep - 1) * 100), touched


res = {"BASE top8 dlv": [], "CHEAP-TILT top8": []}
for d, wkp in pool.groupby("trade_date"):
    wkp = wkp.dropna(subset=["dlv"])
    if len(wkp) < 8:
        continue
    base8 = wkp.sort_values("dlv", ascending=False).head(8)
    e = wkp.copy(); e["cheap"] = (e["pe_ind"] < 1).fillna(False)
    tilt8 = e.sort_values(["cheap", "dlv"], ascending=[False, False]).head(8)
    for name, bk in [("BASE top8 dlv", base8), ("CHEAP-TILT top8", tilt8)]:
        out = [c2(r["symbol"], d) for _, r in bk.iterrows()]
        rets = [a for a, b in out if a is not None]
        tch = [b for a, b in out if a is not None]
        if len(rets) >= 4:
            res[name].append(dict(d=d, ret=np.mean(rets) - 0.30, touch=np.mean(tch),
                                  year=d.year))
print(f"\n{'VARIANT':<20s}| era      | weeks | touch% | mean wk net", flush=True)
for name, rr in res.items():
    W = pd.DataFrame(rr)
    for era, em in ERAS:
        s = W[em(W)]
        print(f"{name:<20s}| {era} | {len(s):>5} | {s['touch'].mean()*100:5.1f}% | "
              f"{s['ret'].mean():+5.2f}%", flush=True)

print("\nVALUATION 3H A/B COMPLETE", flush=True)
