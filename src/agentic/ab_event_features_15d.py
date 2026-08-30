"""A/B — do the new event-history features improve the 15D/+5% prediction?

Pre-registered 2026-08-31. Governs whether event features enter the production
15d stack (classifier/engines). Ships ONLY if FULL beats PRICE on the contract
metric (precision@top-8/week) in BOTH eras and in the since-April subwindow.

Target : touch +5% within 15td (the contract), weekly grid, contract universe
         (ADV>=5cr, close>50).
Arms   : PRICE = production-style price/volume/delivery/market features
         FULL  = PRICE + event features (orders_90d, filings_30d, days_since_print,
                 insider/promoter PIT 90d, SHP promoter delta, results_dump_10d flag,
                 pre_print_thrust flag)
Folds  : yearly walk-forward 2018→2026, train rows' 15td label windows fully closed
         before the scored year; LGBM + isotonic; identical params both arms.
Metrics: AUC-PR, precision@top-8/week (the basket), since-2026-04 subwindow.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "close", "return_1d", "return_20d",
             "rsi_14_daily", "volume_vs_20d", "delivery_pct", "avg_delivery_pct_20d",
             "avg_traded_value_20d", "sma_50", "sma_200"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol", sort=False)
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["ret5"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["rsi_mu"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=100).mean())
px["rsi_sd"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=100).std())
px["rsi_z"] = (px["rsi_14_daily"] - px["rsi_mu"]) / px["rsi_sd"]
px["r20_z"] = px["return_20d"] / (px["dvol"] * np.sqrt(20))
px["r5_z"] = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["hi252"] = g["close"].transform(lambda s: s.rolling(252, min_periods=60).max())
px["off52"] = px["close"] / px["hi252"] - 1
px["vs50"] = px["close"] / px["sma_50"] - 1
px["vs200"] = px["close"] / px["sma_200"] - 1
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["dlv10"] = g["dlv"].transform(lambda s: s.rolling(10, min_periods=3).mean())
px["age_td"] = g.cumcount()
mkt = px[px["adv"] >= 5].groupby("trade_date").agg(
    mkt_breadth=("vs200", lambda s: (s > 0).mean()), mkt_ret20=("return_20d", "median")).reset_index()
px = px.merge(mkt, on="trade_date", how="left")
px["fwdhi15"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(15, min_periods=15).max()[::-1])
px["y"] = ((px["fwdhi15"] / px["close"] - 1) >= 0.05).astype(float)
px.loc[px["fwdhi15"].isna(), "y"] = np.nan

days = np.array(sorted(px["trade_date"].unique()))
weekly = set(days[::5])
grid = px[px["trade_date"].isin(weekly) & (px["adv"] >= 5) & (px["close"] > 50)
          & (px["trade_date"] >= "2016-06-01") & px["y"].notna()].copy()
print(f"grid: {len(grid):,} rows · base P(+5%/15td) {grid['y'].mean()*100:.1f}%", flush=True)

print("event features…", flush=True)
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "sort_date", "desc"])
ah["d"] = pd.to_datetime(ah["sort_date"], errors="coerce").dt.normalize()
ORDER_RE = re.compile(r"bagging|receipt of order|award.{0,12}(?:order|contract)|work order|letter of (?:intent|award)|purchase order|order (?:received|book|win)", re.I)
orders = ah[ah["desc"].fillna("").str.contains(ORDER_RE)].dropna(subset=["d"])

pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet")
pit["d"] = pd.to_datetime(pit["date"], errors="coerce", dayfirst=True).dt.normalize()
symc = [c for c in pit.columns if "symbol" in c.lower()][0]
pit["sym"] = pit[symc]
pit["dpct"] = pd.to_numeric(pit.get("afterAcqSharesPer"), errors="coerce") - pd.to_numeric(pit.get("befAcqSharesPer"), errors="coerce")

shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["known"] = shp["qe"] + pd.Timedelta(days=45)

cal = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")
cal["d"] = pd.to_datetime(cal["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
cal.loc[cal["d"].isna(), "d"] = pd.to_datetime(cal.loc[cal["d"].isna(), "filing_date"],
                                               format="%d-%b-%Y %H:%M", errors="coerce")
cal2 = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
cal2["d"] = pd.to_datetime(cal2["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
prints = pd.concat([cal[["symbol", "d"]], cal2[["symbol", "d"]]]).dropna()
prints["d"] = prints["d"].dt.normalize()

def trailing(events, symcol, valcol, name, win):
    ev = {s: (gg["d"].values.astype("datetime64[ns]"),
              (gg[valcol].values if valcol else np.ones(len(gg))))
          for s, gg in events.sort_values("d").groupby(symcol)}
    out = np.full(len(grid), 0.0)
    for s, idx in grid.groupby("symbol").indices.items():
        e = ev.get(s)
        if e is None:
            continue
        dates, vals = e
        cum = np.concatenate([[0.0], np.nancumsum(vals)])
        ts = grid["trade_date"].values[idx]
        hi = np.searchsorted(dates, ts, side="right")
        lo = np.searchsorted(dates, ts - np.timedelta64(win, "D"), side="right")
        out[idx] = cum[hi] - cum[lo]
    grid[name] = out

trailing(orders, "symbol", None, "orders_90d", 90)
trailing(ah.dropna(subset=["d"]), "symbol", None, "filings_30d", 30)
trailing(pit[pit["dpct"] > 0].dropna(subset=["d"]), "sym", "dpct", "insider_buy_90d", 90)
trailing(pit[pit["dpct"] < 0].dropna(subset=["d"]), "sym", "dpct", "insider_sell_90d", 90)

grid = grid.sort_values("trade_date")
sh = shp.dropna(subset=["prom_delta"]).sort_values("known")[["symbol", "known", "prom_delta"]]
grid = pd.merge_asof(grid, sh.rename(columns={"known": "trade_date", "prom_delta": "shp_prom_delta"}),
                     on="trade_date", by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
pr = prints.sort_values("d").rename(columns={"d": "trade_date"})
pr["last_print"] = pr["trade_date"]
grid = pd.merge_asof(grid, pr[["symbol", "trade_date", "last_print"]], on="trade_date",
                     by="symbol", direction="backward", tolerance=pd.Timedelta(days=400))
grid["days_since_print"] = (grid["trade_date"] - grid["last_print"]).dt.days

# flags from the negative findings: results-dump and pre-print thrust in the last 10 sessions
print("flags…", flush=True)
pm = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in prints.groupby("symbol")}
def near(mp, s, t, lo, hi):
    a = mp.get(s)
    if a is None:
        return False
    return np.searchsorted(a, t + np.timedelta64(hi, "D"), "right") > np.searchsorted(a, t + np.timedelta64(lo, "D"), "left")
px["big_dump"] = (px["return_1d"] <= -0.05) & (px["volume_vs_20d"] >= 3)
px["big_thrust"] = (px["return_1d"] >= 0.05) & (px["volume_vs_20d"] >= 3)
dmp = {s: gg.loc[gg["big_dump"], "trade_date"].values.astype("datetime64[ns]") for s, gg in px.groupby("symbol")}
thr = {s: gg.loc[gg["big_thrust"], "trade_date"].values.astype("datetime64[ns]") for s, gg in px.groupby("symbol")}
t64 = grid["trade_date"].values.astype("datetime64[ns]")
def had(mp, lo, hi):
    return np.array([near(mp, s, t, lo, hi) for s, t in zip(grid["symbol"], t64)])
grid["dump_10d"] = had(dmp, -14, 0)
grid["thrust_10d"] = had(thr, -14, 0)
# conjunction with prints: dump/thrust within ±4d of a print in the last 14d
grid["results_dump_10d"] = grid["dump_10d"] & had(pm, -18, 0)
grid["pre_print_thrust"] = grid["thrust_10d"] & np.array([near(pm, s, t, 0, 14) for s, t in zip(grid["symbol"], t64)])

PRICE = ["return_1d", "return_20d", "ret5", "rsi_14_daily", "rsi_z", "r20_z", "r5_z", "dvol",
         "off52", "vs50", "vs200", "volume_vs_20d", "dlv", "dlv10", "adv", "age_td",
         "mkt_breadth", "mkt_ret20"]
EVENT = ["orders_90d", "filings_30d", "insider_buy_90d", "insider_sell_90d",
         "shp_prom_delta", "days_since_print", "results_dump_10d", "pre_print_thrust"]
grid[["results_dump_10d", "pre_print_thrust"]] = grid[["results_dump_10d", "pre_print_thrust"]].astype(float)
grid["year"] = grid["trade_date"].dt.year

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

def run(feats):
    preds = []
    for Y in range(2018, 2027):
        cutoff = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=30)
        tr = grid[grid["trade_date"] <= cutoff]
        sc = grid[grid["year"] == Y]
        if len(tr) < 60_000 or len(sc) == 0:
            continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.04, num_leaves=96,
                               min_child_samples=150, feature_fraction=0.8, bagging_fraction=0.8,
                               bagging_freq=5, random_state=42, verbose=-1, n_jobs=-1)
        m.fit(tr[feats], tr["y"])
        tail = tr[tr["trade_date"] >= tr["trade_date"].quantile(0.8)]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(m.predict_proba(tail[feats])[:, 1], tail["y"])
        d = sc[["symbol", "trade_date", "y", "year"]].copy()
        d["p"] = iso.transform(m.predict_proba(sc[feats])[:, 1])
        preds.append(d)
        print(f"    {Y} done", flush=True)
    return pd.concat(preds, ignore_index=True)

def report(tag, d):
    for era, m in [("2018-22", d["year"] <= 2022), ("2023-26", d["year"] >= 2023),
                   ("sinceApr26", d["trade_date"] >= "2026-04-01")]:
        s = d[m]
        if len(s) < 500:
            continue
        top8 = s.sort_values("p", ascending=False).groupby("trade_date").head(8)
        print(f"| {tag} {era} | {s['y'].mean()*100:.1f}% | {average_precision_score(s['y'], s['p'])*100:.1f}% | "
              f"{top8['y'].mean()*100:.1f}% | {top8['y'].mean()/s['y'].mean():.2f}x |", flush=True)

print("\n| arm+era | base touch | AUC-PR | top-8/wk touch | lift |")
print("|---|---|---|---|---|")
for tag, feats in [("PRICE", PRICE), ("FULL", PRICE + EVENT)]:
    print(f"  running {tag}…", flush=True)
    report(tag, run(feats))
print("AB 15D COMPLETE", flush=True)
