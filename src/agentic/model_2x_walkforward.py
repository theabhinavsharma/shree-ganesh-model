"""THE 2X MODEL — P(touch 2x within x td | everything knowable at T), walk-forward 2016→2026.

Pre-registered 2026-08-31. Questions this run answers:
  1. FIND THE x: horizon ∈ {60, 100, 180, 252} td — where is 2x most predictable?
  2. DOES THE NEW DATA HELP: ablation on identical folds —
       PRICE variant  : price/volume/delivery/market features only
       FULL variant   : PRICE + event-history features (announcements, order wins,
                        insider PIT, promoter SHP deltas, block deals, results cadence)
  3. THE PROBABILITY: isotonic-calibrated P per name per week + calibration table.

Discipline:
  • weekly grid, investable rows (ADV>=5cr, close>50)
  • features strictly from data <= T (SHP lagged 45d past quarter-end; PIT/announcements
    by event date; NaN where a feed hadn't started — LGBM handles missing natively)
  • labels: fwd max(high)/close(T) - 1 >= 100% within x td; incomplete windows excluded
  • walk-forward: score year Y with a model trained ONLY on rows whose label window
    closes before Y starts; retrained每 year; isotonic calibrated on the train tail
  • metrics: AUC-PR vs base rate, precision@top-10/week, calibration deciles,
    and the portfolio view (top-10/wk touch-2x rate + mean end return)

Output: research/model_2x/predictions_<variant>_<x>.parquet · report printed at end.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT))
OUT = ROOT / "research/model_2x"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [60, 100, 180, 252]
SCORE_YEARS = list(range(2018, 2027))

# ─────────────────────────── price panel ───────────────────────────
print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d", "return_20d",
             "rsi_14_daily", "volume_vs_20d", "delivery_pct", "avg_delivery_pct_20d",
             "avg_traded_value_20d", "sma_50", "sma_200"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol", sort=False)
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["ret5"] = g["close"].pct_change(5)
px["ret60"] = g["close"].pct_change(60)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["rsi_mu"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=100).mean())
px["rsi_sd"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=100).std())
px["rsi_z"] = (px["rsi_14_daily"] - px["rsi_mu"]) / px["rsi_sd"]
px["hi252"] = g["close"].transform(lambda s: s.rolling(252, min_periods=60).max())
px["off52"] = px["close"] / px["hi252"] - 1
px["vs50"] = px["close"] / px["sma_50"] - 1
px["vs200"] = px["close"] / px["sma_200"] - 1
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["dlv10"] = g["dlv"].transform(lambda s: s.rolling(10, min_periods=3).mean())
up = np.where(px["return_1d"] > 0, px["volume_vs_20d"], np.nan)
dn = np.where(px["return_1d"] < 0, px["volume_vs_20d"], np.nan)
px["_u"], px["_d"] = up, dn
px["up15"] = g["_u"].transform(lambda s: s.rolling(15, min_periods=3).mean())
px["dn15"] = g["_d"].transform(lambda s: s.rolling(15, min_periods=3).mean())
px["vol_asym"] = px["up15"] / px["dn15"]
px["age_td"] = g.cumcount()
mkt = px[px["adv"] >= 5].groupby("trade_date").agg(
    mkt_breadth=("vs200", lambda s: (s > 0).mean()), mkt_ret20=("return_20d", "median")).reset_index()
px = px.merge(mkt, on="trade_date", how="left")

print("labels…", flush=True)
for x in HORIZONS:
    px[f"fwdhi_{x}"] = g["high"].transform(
        lambda s, n=x: s.shift(-1)[::-1].rolling(n, min_periods=n).max()[::-1])
    px[f"y2x_{x}"] = ((px[f"fwdhi_{x}"] / px["close"] - 1) >= 1.0).astype(float)
    px.loc[px[f"fwdhi_{x}"].isna(), f"y2x_{x}"] = np.nan
px[f"end_{HORIZONS[1]}"] = g["close"].transform(lambda s: s.shift(-HORIZONS[1]))

days = np.array(sorted(px["trade_date"].unique()))
weekly = set(days[::5])
grid = px[(px["trade_date"].isin(weekly)) & (px["adv"] >= 5) & (px["close"] > 50)
          & (px["trade_date"] >= "2016-06-01")].copy()
print(f"grid rows: {len(grid):,}", flush=True)

# ─────────────────────────── event features ───────────────────────────
print("event features…", flush=True)
import re
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "sort_date", "desc"])
ah["d"] = pd.to_datetime(ah["sort_date"], errors="coerce").dt.normalize()
ah = ah.dropna(subset=["d"])
ORDER_RE = re.compile(r"bagging|receipt of order|award.{0,12}(order|contract)|work order|letter of (intent|award)|purchase order|order (received|book|win)", re.I)
RESULT_RE = re.compile(r"financial result|results.{0,20}(quarter|year)|outcome of board meeting", re.I)
ah["is_order"] = ah["desc"].fillna("").str.contains(ORDER_RE)
ah["is_result"] = ah["desc"].fillna("").str.contains(RESULT_RE)

pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet")
pit["d"] = pd.to_datetime(pit["date"], errors="coerce", dayfirst=True).dt.normalize()
symc = [c for c in pit.columns if "symbol" in c.lower()][0]
pit["sym"] = pit[symc]
pcat = [c for c in pit.columns if "person" in c.lower() or "category" in c.lower()]
pit["is_prom"] = pit[pcat[0]].astype(str).str.contains("promoter", case=False) if pcat else True
pit["dpct"] = pd.to_numeric(pit.get("afterAcqSharesPer"), errors="coerce") - pd.to_numeric(pit.get("befAcqSharesPer"), errors="coerce")

shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["known"] = shp["qe"] + pd.Timedelta(days=45)

blk = pd.read_parquet(ROOT / "data/derived/block_deals_history.parquet")
blk["d"] = pd.to_datetime(blk["BD_DT_DATE"], errors="coerce", dayfirst=True).dt.normalize()
blk["sym"] = blk["BD_SYMBOL"]

cal = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")
cal["d"] = pd.to_datetime(cal["filing_date"], format="%d-%b-%Y %H:%M", errors="coerce")
cal.loc[cal["d"].isna(), "d"] = pd.to_datetime(cal.loc[cal["d"].isna(), "broadcast"],
                                               format="%d-%b-%Y %H:%M:%S", errors="coerce")
cal2 = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
cal2["d"] = pd.to_datetime(cal2["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
prints = pd.concat([cal[["symbol", "d"]], cal2[["symbol", "d"]]]).dropna()
prints["d"] = prints["d"].dt.normalize()

def trailing_counts(events: pd.DataFrame, symcol: str, valcol: str | None, name: str, win: int) -> None:
    """grid[name] = sum(val) / count of events in (T-win, T] per symbol, via searchsorted."""
    ev = {s: (gg["d"].values.astype("datetime64[ns]"),
              (gg[valcol].values if valcol else np.ones(len(gg))))
          for s, gg in events.sort_values("d").groupby(symcol)}
    out = np.full(len(grid), np.nan)
    for s, idx in grid.groupby("symbol").indices.items():
        e = ev.get(s)
        if e is None:
            out[idx] = 0.0
            continue
        dates, vals = e
        cum = np.concatenate([[0.0], np.nancumsum(vals)])
        ts = grid["trade_date"].values[idx]
        hi = np.searchsorted(dates, ts, side="right")
        lo = np.searchsorted(dates, ts - np.timedelta64(win, "D"), side="right")
        out[idx] = cum[hi] - cum[lo]
    grid[name] = out

trailing_counts(ah[ah["is_order"]], "symbol", None, "orders_90d", 90)
trailing_counts(ah, "symbol", None, "filings_30d", 30)
trailing_counts(pit[pit["dpct"] > 0], "sym", "dpct", "insider_buy_pct_90d", 90)
trailing_counts(pit[pit["is_prom"] & (pit["dpct"] > 0)], "sym", "dpct", "promoter_buy_pct_90d", 90)
trailing_counts(pit[pit["dpct"] < 0], "sym", "dpct", "insider_sell_pct_90d", 90)
trailing_counts(blk, "sym", None, "blocks_90d", 90)

# promoter SHP delta known at T (merge_asof on 'known')
grid = grid.sort_values("trade_date")
sh = shp.dropna(subset=["prom_delta"]).sort_values("known")[["symbol", "known", "prom_delta"]]
grid = pd.merge_asof(grid, sh.rename(columns={"known": "trade_date", "prom_delta": "shp_prom_delta"}),
                     on="trade_date", by="symbol", direction="backward",
                     tolerance=pd.Timedelta(days=200))
# days since last results print
pr = prints.sort_values("d").rename(columns={"d": "trade_date"})
pr["last_print"] = pr["trade_date"]
grid = pd.merge_asof(grid, pr[["symbol", "trade_date", "last_print"]], on="trade_date",
                     by="symbol", direction="backward", tolerance=pd.Timedelta(days=400))
grid["days_since_print"] = (grid["trade_date"] - grid["last_print"]).dt.days

PRICE_FEATS = ["return_1d", "return_20d", "ret5", "ret60", "rsi_14_daily", "rsi_z", "dvol",
               "off52", "vs50", "vs200", "volume_vs_20d", "dlv", "dlv10", "vol_asym",
               "adv", "age_td", "mkt_breadth", "mkt_ret20"]
EVENT_FEATS = ["orders_90d", "filings_30d", "insider_buy_pct_90d", "promoter_buy_pct_90d",
               "insider_sell_pct_90d", "blocks_90d", "shp_prom_delta", "days_since_print"]
grid = grid.dropna(subset=[f for f in PRICE_FEATS if f not in ("mkt_breadth", "mkt_ret20")]).copy()
grid["year"] = grid["trade_date"].dt.year
print(f"model rows: {len(grid):,} · event-feature nonzero share: "
      + ", ".join(f"{c}:{(grid[c].fillna(0)!=0).mean()*100:.0f}%" for c in EVENT_FEATS), flush=True)

# ─────────────────────────── walk-forward ───────────────────────────
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

def run(variant: str, feats: list[str], x: int) -> pd.DataFrame:
    ycol = f"y2x_{x}"
    preds = []
    for Y in SCORE_YEARS:
        cutoff = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=int(x * 1.6))
        tr = grid[(grid["trade_date"] <= cutoff) & grid[ycol].notna()]
        sc = grid[(grid["year"] == Y) & grid[ycol].notna()]
        if len(tr) < 60_000 or len(sc) == 0:
            continue
        pos = tr[ycol].mean()
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=96,
                               min_child_samples=150, feature_fraction=0.8,
                               bagging_fraction=0.8, bagging_freq=5,
                               scale_pos_weight=min(30, (1 - pos) / max(pos, 1e-4)),
                               random_state=42, verbose=-1, n_jobs=-1)
        m.fit(tr[feats], tr[ycol])
        tail = tr[tr["trade_date"] >= tr["trade_date"].quantile(0.8)]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(tail[feats])[:, 1], tail[ycol])
        p = iso.transform(m.predict_proba(sc[feats])[:, 1])
        d = sc[["symbol", "trade_date", ycol, f"end_{HORIZONS[1]}", "close"]].copy()
        d["p"] = p
        preds.append(d)
        print(f"  [{variant} x={x}] {Y}: train {len(tr):,} (base {pos*100:.2f}%) → scored {len(sc):,}", flush=True)
    out = pd.concat(preds, ignore_index=True)
    out.to_parquet(OUT / f"predictions_{variant}_{x}.parquet", index=False)
    return out

def evaluate(tag: str, d: pd.DataFrame, x: int) -> str:
    ycol = f"y2x_{x}"
    base = d[ycol].mean()
    ap = average_precision_score(d[ycol], d["p"])
    top10 = d.sort_values("p", ascending=False).groupby("trade_date").head(10)
    p10 = top10[ycol].mean()
    d = d.copy()
    d["dec"] = pd.qcut(d["p"], 10, labels=False, duplicates="drop")
    top_dec = d[d["dec"] == d["dec"].max()]
    calib = f"{top_dec['p'].mean()*100:.1f}%pred/{top_dec[ycol].mean()*100:.1f}%real"
    return (f"| {tag} | {base*100:.2f}% | {ap*100:.2f}% | {ap/base:.2f}x | "
            f"{p10*100:.1f}% ({p10/base:.1f}x) | {calib} |")

print("\n| model | base P(2x) | AUC-PR | PR lift | top-10/wk precision | top-decile calib |")
print("|---|---|---|---|---|---|")
results = {}
for x in HORIZONS:
    for variant, feats in [("PRICE", PRICE_FEATS), ("FULL", PRICE_FEATS + EVENT_FEATS)]:
        d = run(variant, feats, x)
        results[(variant, x)] = d
        print(evaluate(f"{variant} x={x}", d, x), flush=True)

# feature importance for the best FULL model (x=100 reference)
d = grid[grid[f"y2x_100"].notna()]
m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=96, min_child_samples=150,
                       feature_fraction=0.8, random_state=42, verbose=-1, n_jobs=-1)
m.fit(d[PRICE_FEATS + EVENT_FEATS], d["y2x_100"])
imp = pd.Series(m.feature_importances_, index=PRICE_FEATS + EVENT_FEATS).sort_values(ascending=False)
print("\nfeature importance (gain order, full-fit reference — NOT walk-forward):")
print(imp.to_string())
print("\nMODEL RUN COMPLETE", flush=True)
