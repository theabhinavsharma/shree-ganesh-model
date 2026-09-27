"""ANATOMY OF A 1.5x — what precedes a +50% move within ~95 sessions (EXP-2026-09-27-1p5x-anatomy).

Universe : ISIN-master equities with PIT market cap >= Rs50cr, weekly samples (every 5th session).
Targets  : y95  = max high over next 95 sessions >= 1.5 x entry close   (primary)
           y63  = same within 63 sessions;  s95 = close at +95 sessions >= 1.5 x (sustained)
Features : ~60, every one known at the entry close (see FEATURES dict below for the full list + meaning).
Part 1   : univariate lift per feature (deciles / flags), both eras (disc <= 2022, conf >= 2023).
Part 2   : LightGBM walk-forward (train on years < Y with target windows closed, predict Y; Y = 2019..last),
           AUC / precision@top-k / lift by year and era; importance = gain + out-of-sample permutation AUC drop.
Part 3   : ALL-IN portfolio (user spec 2026-09-27): 100% capital enters at once, equal weight, held 95 sessions,
           then fully rotated; 19 phase offsets; 0.5% round trip per rotation. Arms: model top-10 / top-20
           (out-of-sample predictions, tradable names ADV>=5cr & close>50), BASELINE rule, G+H rule, EW market.
Output   : logs/leader_sleeve/anatomy_1p5x/ (rows.parquet + manifest, README.md, results.json); stdout -> log.
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT / "src/agentic"))
from generate_hybrid_basket import non_equity  # noqa: E402
from event_materiality_study import CATS  # noqa: E402

H, H2 = 95, 63
OUTD = ROOT / "logs/leader_sleeve/anatomy_1p5x"; OUTD.mkdir(parents=True, exist_ok=True)
ERA = lambda d: np.where(pd.to_datetime(d).dt.year >= 2023, "conf", "disc")
T0 = datetime.now()
say = lambda *a: print(f"[{(datetime.now()-T0).seconds:>4}s]", *a, flush=True)

# ============ panel ============
sm = pd.read_parquet(ROOT / "data/derived/security_master.parquet")
fund = set(sm.loc[sm["is_fund_unit"], "symbol"])
cols = ["symbol", "trade_date", "high", "low", "close", "return_1d", "volume_vs_20d", "volume_vs_60d", "traded_value_vs_60d",
        "delivery_pct", "avg_delivery_pct_20d", "delivery_pct_vs_20d", "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
        "sma_50", "sma_200", "avg_traded_value_20d"]
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=cols)
px = px[~px["symbol"].isin(fund) & ~non_equity(px["symbol"])].copy()
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol")
for k in (5, 20, 60, 126, 252):
    px[f"ret{k}"] = g["close"].pct_change(k, fill_method=None)
px["hi252"] = g["high"].transform(lambda s: s.rolling(252, min_periods=60).max())
px["lo252"] = g["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
px["off_high"] = px["close"] / px["hi252"] - 1
px["off_low"] = px["close"] / px["lo252"] - 1
px["dvol20"] = g["return_1d"].transform(lambda s: s.rolling(20, min_periods=15).std())
px["dvol60"] = g["return_1d"].transform(lambda s: s.rolling(60, min_periods=40).std())
uc = ((px["return_1d"] >= 0.0495) & (px["close"] >= px["high"] * 0.999)).astype(float)
px["uc20"] = uc.groupby(px["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).sum())
px["up20"] = (px["return_1d"] > 0).astype(float).groupby(px["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).sum())
px["px_sma50"] = px["close"] / px["sma_50"] - 1
px["px_sma200"] = px["close"] / px["sma_200"] - 1
px["sma50_200"] = px["sma_50"] / px["sma_200"] - 1
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["age_yrs"] = (px["trade_date"] - g["trade_date"].transform("min")).dt.days / 365.25
fwd_max = lambda n: g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(n, min_periods=int(n * .8)).max()[::-1])
px["fh95"] = fwd_max(H) / px["close"]; px["fh63"] = fwd_max(H2) / px["close"]
px["fc95"] = g["close"].shift(-H) / px["close"]
px["core"] = (px["adv"] >= 5) & (px["close"] > 50)
mc = pd.read_parquet(ROOT / "data/derived/mcap_pit.parquet", columns=["symbol", "trade_date", "mcap_cr"])
mc["trade_date"] = pd.to_datetime(mc["trade_date"])
px = px.merge(mc, on=["symbol", "trade_date"], how="left")
days = np.array(sorted(px["trade_date"].unique()))
say(f"panel {len(px):,} rows · {px['symbol'].nunique()} symbols")

# market regime (daily)
core_d = px[px["core"] & px["sma_50"].notna()]
mkt = pd.DataFrame({"mkt_breadth": core_d.assign(u=core_d["close"] > core_d["sma_50"]).groupby("trade_date")["u"].mean(),
                    "mkt_med_ret20": px[px["mcap_cr"] >= 50].groupby("trade_date")["ret20"].median(),
                    "mkt_ew_ret60": px[px["mcap_cr"] >= 50].groupby("trade_date")["ret60"].mean()})

# ============ weekly sample ============
wkdays = set(days[::5])
S = px[px["trade_date"].isin(wkdays) & (px["mcap_cr"] >= 50) & (px["trade_date"] >= "2016-01-01")].copy()
del px
S = S.join(mkt, on="trade_date")
S["log_mcap"] = np.log10(S["mcap_cr"]); S["log_adv"] = np.log10(S["adv"].clip(lower=1e-3)); S["log_px"] = np.log10(S["close"].clip(lower=0.01))

# industry (nse4 map)
sc = pd.read_parquet(ROOT / "data/derived/screener_industry.parquet")
sc = sc[sc["status"].str.startswith("OK")].dropna(subset=["industry"])
imap = sc.set_index("symbol")["industry"].map(html.unescape)
al = pd.read_csv(ROOT / "data/derived/industry_analyst_labels.csv")
imap = pd.concat([imap, al[~al["symbol"].isin(imap.index)].set_index("symbol")["analog_symbol"].map(imap).dropna()])
S["ind"] = S["symbol"].map(imap)
gi = S.groupby(["trade_date", "ind"])
S["ind_n"] = gi["ret60"].transform("count")
S["ind_heat"] = gi["ret60"].transform("mean")
S["ind_ret20"] = gi["ret20"].transform("mean")
S["ind_breadth"] = gi["px_sma50"].transform(lambda s: (s > 0).mean())
S.loc[S["ind_n"] < 5, ["ind_heat", "ind_ret20", "ind_breadth"]] = np.nan
hp = S.dropna(subset=["ind_heat"]).drop_duplicates(["trade_date", "ind"])[["trade_date", "ind", "ind_heat"]]
hp["ind_heat_pct"] = hp.groupby("trade_date")["ind_heat"].rank(pct=True)
S = S.merge(hp[["trade_date", "ind", "ind_heat_pct"]], on=["trade_date", "ind"], how="left")
S["rank_in_ind_pct"] = S.groupby(["trade_date", "ind"])["ret60"].rank(pct=True, ascending=False)
say(f"weekly sample {len(S):,} rows")

# ============ fundamentals (PIT) ============
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end"], keep="last").sort_values(["symbol", "quarter_end"])
to_cr = np.where(q["source"] == "xbrl", 1e-7, 1e-2)                         # pnl_quarterly manifest units
q["sales_cr"] = q["net_sales"] * to_cr
qg = q.groupby("symbol")
q["eps_ttm"] = qg["eps_basic"].transform(lambda s: s.rolling(4).sum())
q["eps_yoy"] = (q["eps_basic"] - qg["eps_basic"].shift(4)) / qg["eps_basic"].shift(4).abs().clip(lower=0.1)
q["sales_yoy"] = q["sales_cr"] / qg["sales_cr"].shift(4) - 1
q["loss_to_profit"] = ((q["eps_basic"] > 0) & (qg["eps_basic"].shift(4) <= 0)).astype(float)
q["known"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
F = q[["symbol", "known", "eps_ttm", "eps_yoy", "sales_yoy", "loss_to_profit"]].rename(columns={"known": "trade_date"}).sort_values("trade_date")
F["res_dt"] = F["trade_date"]
S = pd.merge_asof(S.sort_values("trade_date"), F, on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
S["days_since_results"] = (S["trade_date"] - S["res_dt"]).dt.days
S["profitable"] = (S["eps_ttm"] > 0).astype(float).where(S["eps_ttm"].notna())
S["pe"] = np.where(S["eps_ttm"] > 0, S["close"] / S["eps_ttm"], np.nan)
S["pe_ind"] = S["pe"] / S.groupby(["trade_date", "ind"])["pe"].transform("median")
S[["eps_yoy", "sales_yoy"]] = S[["eps_yoy", "sales_yoy"]].clip(-5, 20)
say("fundamentals merged")

# ============ events / insiders / block deals / promoter ============
def window_count(ev: pd.DataFrame, days_back: int, name: str):
    """count events per symbol in (t - days_back, t] for every sampled row."""
    global S
    ev = ev.dropna(subset=["d"]).sort_values(["symbol", "d"])
    arr = {s: v["d"].values.astype("datetime64[ns]") for s, v in ev.groupby("symbol")}
    out = np.zeros(len(S))
    t = S["trade_date"].values.astype("datetime64[ns]"); lo = t - np.timedelta64(days_back, "D")
    for s, idx in S.groupby("symbol").indices.items():
        a = arr.get(s)
        if a is not None:
            out[idx] = np.searchsorted(a, t[idx], side="right") - np.searchsorted(a, lo[idx], side="right")
    S[name] = out

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "desc", "attchmntText", "sort_date"])
ah["txt"] = ah["desc"].fillna("") + " " + ah["attchmntText"].fillna("")
ah["d"] = pd.to_datetime(ah["sort_date"], errors="coerce").dt.normalize()
ah["cat"] = None
for c, p in CATS.items():
    m = ah["cat"].isna() & ah["txt"].str.contains(p, case=False, regex=True)
    ah.loc[m, "cat"] = c
for c in CATS:
    window_count(ah[ah["cat"] == c][["symbol", "d"]], 60, f"ev60_{c}")
window_count(ah[["symbol", "d"]], 30, "filings30_all")
del ah
er = pd.read_parquet(ROOT / "logs/leader_sleeve/event_rows_mcap50_20260924.parquet", columns=["symbol", "d", "cat", "order_to_rev"])
er = er[er["cat"].isin(["order", "tender_L1"]) & er["order_to_rev"].notna()].rename(columns={"d": "trade_date"}).sort_values("trade_date")
er["ord_dt"] = er["trade_date"]
S = pd.merge_asof(S.sort_values("trade_date"), er[["symbol", "trade_date", "order_to_rev", "ord_dt"]], on="trade_date", by="symbol",
                  direction="backward", tolerance=pd.Timedelta(days=90)).rename(columns={"order_to_rev": "last_order_to_rev_90d"})

pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet", columns=["symbol", "personCategory", "acqMode", "buyValue", "sellValue", "intimDt"]) \
    if "symbol" in __import__("pyarrow.parquet", fromlist=["x"]).read_schema(ROOT / "data/derived/pit_history.parquet").names else None
if pit is not None:
    pit["d"] = pd.to_datetime(pit["intimDt"], errors="coerce", dayfirst=True).dt.normalize()
    prom = pit[pit["personCategory"].fillna("").str.contains("Promoter", case=False)]
    bv = pd.to_numeric(prom["buyValue"], errors="coerce").fillna(0); sv = pd.to_numeric(prom["sellValue"], errors="coerce").fillna(0)
    window_count(prom[bv > 0][["symbol", "d"]], 90, "prom_buys90")
    window_count(prom[sv > 0][["symbol", "d"]], 90, "prom_sells90")
blk = pd.read_parquet(ROOT / "data/derived/block_deals_history.parquet", columns=["BD_SYMBOL", "BD_DT_DATE", "BD_BUY_SELL"])
blk = blk.rename(columns={"BD_SYMBOL": "symbol"}); blk["d"] = pd.to_datetime(blk["BD_DT_DATE"], format="mixed", errors="coerce").dt.normalize()
window_count(blk[blk["BD_BUY_SELL"].astype(str).str.upper().str.startswith("B")][["symbol", "d"]], 60, "block_buys60")
window_count(blk[blk["BD_BUY_SELL"].astype(str).str.upper().str.startswith("S")][["symbol", "d"]], 60, "block_sells60")
shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet", columns=["symbol", "quarter_end", "promoter_pct"])
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce"); shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["trade_date"] = shp["qe"] + pd.Timedelta(days=21)
S = pd.merge_asof(S.sort_values("trade_date"), shp[["symbol", "trade_date", "promoter_pct", "prom_delta"]].sort_values("trade_date"),
                  on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
say("events/insiders/blocks/promoter merged")

# ============ feature registry ============
FEATURES = {
    # tape
    "ret5": "1-week return", "ret20": "1-month return", "ret60": "3-month return", "ret126": "6-month return", "ret252": "12-month return",
    "off_high": "distance below 52-week high (0 = at high)", "off_low": "distance above 52-week low",
    "dvol20": "20d daily volatility", "dvol60": "60d daily volatility",
    "volume_vs_20d": "today's volume / 20d avg", "volume_vs_60d": "today's volume / 60d avg", "traded_value_vs_60d": "today's traded value / 60d avg",
    "delivery_pct": "delivery % today", "avg_delivery_pct_20d": "delivery % 20d avg", "delivery_pct_vs_20d": "delivery % vs its 20d avg",
    "rsi_14_daily": "RSI daily", "rsi_14_weekly": "RSI weekly", "rsi_14_monthly": "RSI monthly",
    "px_sma50": "price vs 50DMA", "px_sma200": "price vs 200DMA", "sma50_200": "50DMA vs 200DMA",
    "uc20": "upper-circuit closes in last 20d", "up20": "up-days in last 20d",
    # size / liquidity
    "log_mcap": "log10 market cap (Rs cr)", "log_adv": "log10 20d avg traded value (Rs cr)", "log_px": "log10 price", "age_yrs": "years listed (panel, 2015 floor)",
    # industry
    "ind_heat_pct": "industry 60d-return heat percentile", "ind_heat": "industry mean 60d return", "ind_ret20": "industry mean 20d return",
    "ind_breadth": "share of industry above 50DMA", "ind_n": "industry size (names)", "rank_in_ind_pct": "own 60d-return rank within industry (0 = leader)",
    # fundamentals
    "profitable": "TTM EPS > 0", "pe": "PE (TTM)", "pe_ind": "PE / industry median PE", "eps_yoy": "latest quarter EPS YoY",
    "sales_yoy": "latest quarter sales YoY", "loss_to_profit": "latest quarter profit after year-ago loss", "days_since_results": "days since last results filing",
    # filings / events
    **{f"ev60_{c}": f"'{c}' filings in last 60d" for c in CATS}, "filings30_all": "all filings last 30d",
    "last_order_to_rev_90d": "largest recent order / TTM revenue (90d)",
    # insiders / flows / ownership
    "prom_buys90": "promoter PIT buy disclosures 90d", "prom_sells90": "promoter PIT sell disclosures 90d",
    "block_buys60": "block-deal buys 60d", "block_sells60": "block-deal sells 60d",
    "promoter_pct": "promoter holding %", "prom_delta": "promoter holding change last quarter (pp)",
    # market
    "mkt_breadth": "share of liquid stocks above 50DMA", "mkt_med_ret20": "median stock 20d return", "mkt_ew_ret60": "equal-weight market 60d return",
}
FEATS = [f for f in FEATURES if f in S.columns]
S["y95"] = (S["fh95"] >= 1.5).astype(float).where(S["fh95"].notna())
S["y63"] = (S["fh63"] >= 1.5).astype(float).where(S["fh63"].notna())
S["s95"] = (S["fc95"] >= 1.5).astype(float).where(S["fc95"].notna())
S["era"] = ERA(S["trade_date"])
L = S.dropna(subset=["y95"])
base = L.groupby("era")[["y95", "y63", "s95"]].mean() * 100
say(f"labelled rows {len(L):,} · features {len(FEATS)}")
print("\nBASE RATES (%):\n" + base.round(2).to_string())

# ============ Part 1: univariate lift ============
uni = []
for f in FEATS:
    x = L[f]
    if x.notna().mean() < 0.05:
        continue
    if x.dropna().nunique() <= 3 or f.startswith(("ev60_", "prom_", "block_", "filings")) or f in ("uc20", "loss_to_profit", "profitable"):
        b = pd.Series(np.where(x.isna(), "n/a", np.where(x > 0, "yes/>0", "no/0")), index=L.index)
    else:
        b = pd.qcut(x.rank(method="first"), 10, labels=[f"d{i}" for i in range(1, 11)]).astype(str).where(x.notna(), "n/a")
    for bk, sub in L.groupby(b):
        if bk == "n/a":
            continue
        r = {"feature": f, "bucket": bk, "lo": float(sub[f].min()), "hi": float(sub[f].max())}
        for e in ("disc", "conf"):
            E = sub[sub["era"] == e]
            r[f"{e}_n"] = len(E); r[f"{e}_p"] = E["y95"].mean() * 100 if len(E) else np.nan
            r[f"{e}_lift"] = r[f"{e}_p"] / base.loc[e, "y95"] if len(E) else np.nan
        uni.append(r)
U = pd.DataFrame(uni)
U["min_lift"] = U[["disc_lift", "conf_lift"]].min(axis=1)
U["both_n_ok"] = (U["disc_n"] >= 300) & (U["conf_n"] >= 300)
print("\n=== PART 1: buckets with lift >= 1.5x in BOTH eras (n >= 300 each), strongest first ===")
top = U[U["both_n_ok"] & (U["min_lift"] >= 1.5)].sort_values("min_lift", ascending=False)
for _, r in top.head(45).iterrows():
    print(f"{r['feature']:<24} {r['bucket']:<7} [{r['lo']:>9.3g} .. {r['hi']:>9.3g}]  disc {r['disc_p']:>5.1f}% ({r['disc_lift']:.1f}x, n{r['disc_n']:,})  "
          f"conf {r['conf_p']:>5.1f}% ({r['conf_lift']:.1f}x, n{r['conf_n']:,})  — {FEATURES[r['feature']]}")
print("\n=== buckets that SUPPRESS 1.5x (<= 0.5x in both eras) ===")
for _, r in U[U["both_n_ok"] & (U[["disc_lift", "conf_lift"]].max(axis=1) <= 0.5)].sort_values("min_lift").head(20).iterrows():
    print(f"{r['feature']:<24} {r['bucket']:<7} [{r['lo']:>9.3g} .. {r['hi']:>9.3g}]  disc {r['disc_lift']:.2f}x  conf {r['conf_lift']:.2f}x  — {FEATURES[r['feature']]}")
print("\n=== ERA-FLIPPERS (>=1.5x in one era, <=1.0x in the other) — not trustworthy ===")
for _, r in U[U["both_n_ok"] & (((U["disc_lift"] >= 1.5) & (U["conf_lift"] <= 1)) | ((U["conf_lift"] >= 1.5) & (U["disc_lift"] <= 1)))].head(15).iterrows():
    print(f"{r['feature']:<24} {r['bucket']:<7} disc {r['disc_lift']:.2f}x  conf {r['conf_lift']:.2f}x")

# ============ Part 2: walk-forward LightGBM ============
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
S["pred"] = np.nan
years = sorted(S["trade_date"].dt.year.unique())
imp_gain = pd.Series(0.0, index=FEATS); perm = []
print("\n=== PART 2: walk-forward LightGBM (train years < Y, windows closed; predict Y) ===")
for Y in [y for y in years if y >= 2019]:
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=150)
    tr = L[L["trade_date"] < cut]; te_mask = S["trade_date"].dt.year == Y
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    m.fit(tr[FEATS], tr["y95"])
    S.loc[te_mask, "pred"] = m.predict_proba(S.loc[te_mask, FEATS])[:, 1]
    imp_gain += pd.Series(m.booster_.feature_importance("gain"), index=FEATS)
    te = S[te_mask & S["y95"].notna()]
    if len(te) and te["y95"].nunique() == 2:
        auc = roc_auc_score(te["y95"], te["pred"])
        k1 = te.nlargest(max(1, len(te) // 100), "pred")["y95"].mean() * 100
        k5 = te.nlargest(max(1, len(te) // 20), "pred")["y95"].mean() * 100
        print(f"  {Y}: train {len(tr):,} · test {len(te):,} · base {te['y95'].mean()*100:4.1f}% · AUC {auc:.3f} · "
              f"top1% {k1:4.1f}% ({k1/(te['y95'].mean()*100):.1f}x) · top5% {k5:4.1f}% ({k5/(te['y95'].mean()*100):.1f}x)", flush=True)
        if Y >= 2023:                                     # out-of-sample permutation importance
            smp = te.sample(min(len(te), 60000), random_state=Y)
            b0 = roc_auc_score(smp["y95"], m.predict_proba(smp[FEATS])[:, 1])
            rng = np.random.default_rng(Y)
            for f in FEATS:
                X = smp[FEATS].copy(); X[f] = rng.permutation(X[f].values)
                perm.append((Y, f, b0 - roc_auc_score(smp["y95"], m.predict_proba(X)[:, 1])))
O = S[S["pred"].notna() & S["y95"].notna()]
for e in ("disc", "conf"):
    E = O[O["era"] == e]
    if len(E):
        k1 = E.nlargest(len(E) // 100, "pred")["y95"].mean() * 100
        print(f"  ERA {e}: AUC {roc_auc_score(E['y95'], E['pred']):.3f} · base {E['y95'].mean()*100:.1f}% · top1% {k1:.1f}% ({k1/(E['y95'].mean()*100):.1f}x)")
P = pd.DataFrame(perm, columns=["year", "feature", "auc_drop"]).groupby("feature")["auc_drop"].mean().sort_values(ascending=False)
G = (imp_gain / imp_gain.sum()).sort_values(ascending=False)
print("\nTOP FEATURES — out-of-sample permutation AUC drop (2023+) and share of model gain:")
for f in P.index[:25]:
    print(f"  {f:<24} AUC drop {P[f]:+.4f} · gain {G.get(f, 0)*100:4.1f}% — {FEATURES[f]}")

# ============ Part 3: ALL-IN portfolio, hold 95, 19 phases ============
pxr = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet", columns=["symbol", "trade_date", "close"])
pxr["trade_date"] = pd.to_datetime(pxr["trade_date"])
R = pxr.pivot(index="trade_date", columns="symbol", values="close").sort_index().pct_change(fill_method=None)
del pxr
dpos = {d: i for i, d in enumerate(R.index)}
wk = sorted(S["trade_date"].unique())
T = S[S["core"]].copy()                                    # tradable names only


def rule_picks(broad_heat: bool, gate: bool) -> dict:
    Wd = S if broad_heat else S[S["core"]]
    Wd = Wd.dropna(subset=["ind", "ret60"]).copy()
    Wd["n"] = Wd.groupby(["trade_date", "ind"])["ret60"].transform("size"); Wd = Wd[Wd["n"] >= 5]
    h = Wd.groupby(["trade_date", "ind"])["ret60"].mean().rename("h").reset_index()
    h["hot"] = h.groupby("trade_date")["h"].rank(pct=True) >= 0.9
    Wd = Wd.merge(h[["trade_date", "ind", "hot"]], on=["trade_date", "ind"])
    Wd["rk"] = Wd.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False, method="first")
    Pk = Wd[Wd["hot"] & (Wd["rk"] <= 3) & (Wd["ret252"] > 0.5) & Wd["core"]]
    if gate:
        Pk = Pk[Pk["mkt_breadth"].groupby(Pk["trade_date"]).transform("first") >= 0.5]  # same-day close breadth (known at entry close)
    return Pk.groupby("trade_date")["symbol"].apply(list).to_dict()


def model_picks(n: int) -> dict:
    M = T.dropna(subset=["pred"])
    return M.sort_values("pred", ascending=False).groupby("trade_date").head(n).groupby("trade_date")["symbol"].apply(list).to_dict()


def allin(picks: dict, start: pd.Timestamp) -> list:
    res = []
    for ph in range(19):
        d0 = [d for d in wk if d >= start][ph::19]
        nav = [1.0]; dates = []
        for d in d0:
            i0 = dpos.get(d)
            if i0 is None:
                continue
            end = min(i0 + H, len(R) - 1)
            names = picks.get(d, [])
            if names:
                path = (1 + R.iloc[i0 + 1:end + 1][names].fillna(0)).cumprod().mean(axis=1).values * (1 - 0.005)
            else:
                path = np.ones(end - i0)
            nav.extend(list(nav[-1] * path)); dates.extend(list(R.index[i0 + 1:end + 1]))
        s = pd.Series(nav[1:], index=dates); s = s[~s.index.duplicated()]
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        res.append(dict(cagr=((s.iloc[-1]) ** (1 / yrs) - 1) * 100, dd=(s / s.cummax() - 1).min() * 100))
    return res


bench = T.groupby("trade_date")["symbol"].apply(list).to_dict()
arms = {"MODEL top-10": model_picks(10), "MODEL top-20": model_picks(20), "BASELINE rule": rule_picks(False, False),
        "G+H rule": rule_picks(True, True), "EW liquid market": bench}
print("\n=== PART 3: ALL-IN, 100% capital, hold 95 sessions, full rotation, 19 phase offsets (2019-01 .. end; model is out-of-sample) ===")
summary = {}
for name, pk in arms.items():
    r = pd.DataFrame(allin(pk, pd.Timestamp("2019-01-01")))
    summary[name] = dict(cagr_med=r["cagr"].median(), cagr_min=r["cagr"].min(), cagr_max=r["cagr"].max(), dd_med=r["dd"].median(), dd_worst=r["dd"].min())
    print(f"  {name:<18} CAGR median {r['cagr'].median():>+6.1f}% (worst phase {r['cagr'].min():+.1f}, best {r['cagr'].max():+.1f}) · "
          f"maxDD median {r['dd'].median():.1f}% (worst {r['dd'].min():.1f}%)", flush=True)

# ============ save ============
keep = ["symbol", "trade_date", "era", "close", "mcap_cr", "adv", "ind", "core", "y95", "y63", "s95", "fh95", "fc95", "pred"] + FEATS
S[keep].to_parquet(OUTD / "rows.parquet", index=False)
U.to_csv(OUTD / "univariate_lift.csv", index=False)
json.dump(dict(base=base.round(3).to_dict(), perm_auc_drop=P.round(5).to_dict(), gain_share=G.round(5).to_dict(), allin=summary),
          open(OUTD / "results.json", "w"), indent=1, default=float)
(OUTD / "rows.parquet.manifest.json").write_text(json.dumps(dict(
    dataset="1.5x anatomy rows", experiment="EXP-2026-09-27-1p5x-anatomy", producer="src/agentic/anatomy_1p5x.py", rows=len(S),
    columns={**{k: v for k, v in FEATURES.items() if k in FEATS}, "y95": "1 if max high over next 95 sessions >= 1.5x entry close",
             "y63": "same within 63 sessions", "s95": "1 if close at +95 sessions >= 1.5x", "fh95": "max high next 95 / close", "fc95": "close +95 / close",
             "pred": "walk-forward LightGBM out-of-sample P(y95) (2019+ only)", "core": "ADV>=5cr & close>50 (tradable)"},
    units="returns/ratios are fractions; mcap_cr, adv in Rs crore", updated=datetime.now().isoformat(timespec="seconds")), indent=1))
(OUTD / "README.md").write_text(
    "# 1.5x anatomy (EXP-2026-09-27-1p5x-anatomy)\n\n"
    "What precedes a +50% move within 95 sessions, for every stock-week with PIT market cap >= Rs50cr since 2016.\n\n"
    "- `rows.parquet` — one row per stock-week: ~60 features known at the entry close, targets, out-of-sample model score (see manifest).\n"
    "- `univariate_lift.csv` — P(+50% within 95 sessions) by feature bucket, per era, with lift vs base.\n"
    "- `results.json` — base rates, permutation importance, gain share, all-in portfolio summary.\n"
    "- Full console output: `logs/leader_sleeve/anatomy_1p5x_20260927.log`.\n\n"
    "Eras: disc = 2016-2022, conf = 2023+. A finding is only trusted if it holds in both.\n")
say("ANATOMY COMPLETE")
