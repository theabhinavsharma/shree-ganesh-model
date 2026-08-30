"""A/B round 2 — reaction-signed print feature + veto/boost RULES on the 15d basket.

Pre-registered 2026-08-31 after the lag-response sweep (good print +5-7pp stable 90d;
bad print -7pp for ~10 sessions; block deal -3..-7pp out to 180d; orders dead).

Arms (identical yearly walk-forward folds, target = touch +5%/15td, contract universe):
  PRICE      : baseline model, top-8/wk by p            (reproduces round-1 numbers)
  FEAT       : PRICE + {days_since_good_print, good_print_90d, days_since_bad_print,
               days_since_block} as model features, top-8 by p
  VETO       : PRICE picks, but candidates with bad-print<=14 cal-days or
               block-deal<=90d are excluded BEFORE taking the top-8
  VETO+BOOST : after veto, take good-print(<=90d) names first (ordered by p) from the
               top-24 candidates, fill the rest by p
Ship bar: beat PRICE top-8 touch in 2018-22 AND 2023-26 AND since-Apr-2026.
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
grid = px[px["trade_date"].isin(set(days[::5])) & (px["adv"] >= 5) & (px["close"] > 50)
          & (px["trade_date"] >= "2016-06-01") & px["y"].notna()].copy()

print("signed print events…", flush=True)
cal = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")
cal["d"] = pd.to_datetime(cal["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
cal.loc[cal["d"].isna(), "d"] = pd.to_datetime(cal.loc[cal["d"].isna(), "filing_date"],
                                               format="%d-%b-%Y %H:%M", errors="coerce")
cal2 = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
cal2["d"] = pd.to_datetime(cal2["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
prints = pd.concat([cal[["symbol", "d"]], cal2[["symbol", "d"]]]).dropna()
prints["d"] = prints["d"].dt.normalize()
nxt = px[["symbol", "trade_date", "return_1d", "volume_vs_20d"]].sort_values("trade_date")
pr = pd.merge_asof(prints.sort_values("d"), nxt.rename(columns={"trade_date": "d"}),
                   on="d", by="symbol", direction="forward", tolerance=pd.Timedelta(days=4))
good = pr[(pr["return_1d"] >= 0.05) & (pr["volume_vs_20d"] >= 2)][["symbol", "d"]]
bad = pr[(pr["return_1d"] <= -0.05) & (pr["volume_vs_20d"] >= 2)][["symbol", "d"]]
blk = pd.read_parquet(ROOT / "data/derived/block_deals_history.parquet")
blk["d"] = pd.to_datetime(blk["BD_DT_DATE"], errors="coerce", dayfirst=True).dt.normalize()
blk = blk.rename(columns={"BD_SYMBOL": "symbol"})[["symbol", "d"]].dropna()

t64 = grid["trade_date"].values.astype("datetime64[ns]")
def days_since(ev):
    mp = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in ev.groupby("symbol")}
    out = np.full(len(grid), 9999.0)
    for s, idx in grid.groupby("symbol").indices.items():
        a = mp.get(s)
        if a is None:
            continue
        ts = t64[idx]
        pos = np.searchsorted(a, ts, side="right") - 1
        ok = pos >= 0
        out[idx[ok]] = (ts[ok] - a[pos[ok]]) / np.timedelta64(1, "D")
    return out

grid["ds_good"] = days_since(good)
grid["ds_bad"] = days_since(bad)
grid["ds_block"] = days_since(blk)
grid["good_90"] = (grid["ds_good"] <= 90).astype(float)
grid["veto"] = (grid["ds_bad"] <= 14) | (grid["ds_block"] <= 90)
grid["year"] = grid["trade_date"].dt.year
print(f"grid {len(grid):,} · veto rate {grid['veto'].mean()*100:.1f}% · good-print-90d {grid['good_90'].mean()*100:.1f}%", flush=True)

PRICE = ["return_1d", "return_20d", "ret5", "rsi_14_daily", "rsi_z", "r20_z", "r5_z", "dvol",
         "off52", "vs50", "vs200", "volume_vs_20d", "dlv", "dlv10", "adv", "age_td",
         "mkt_breadth", "mkt_ret20"]
FEAT = PRICE + ["ds_good", "good_90", "ds_bad", "ds_block"]

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

def run(feats, tag):
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
        d = sc[["symbol", "trade_date", "y", "year", "veto", "good_90"]].copy()
        d["p"] = iso.transform(m.predict_proba(sc[feats])[:, 1])
        preds.append(d)
        print(f"    [{tag}] {Y}", flush=True)
    return pd.concat(preds, ignore_index=True)

def top8(d, mode):
    outs = []
    for _, wk in d.groupby("trade_date"):
        wk = wk.sort_values("p", ascending=False)
        if mode == "raw":
            outs.append(wk.head(8))
        elif mode == "veto":
            outs.append(wk[~wk["veto"]].head(8))
        elif mode == "veto_boost":
            cand = wk[~wk["veto"]].head(24)
            pick = pd.concat([cand[cand["good_90"] == 1], cand[cand["good_90"] == 0]]).head(8)
            outs.append(pick)
    return pd.concat(outs)

def report(tag, sel):
    row = [tag]
    for era, m in [("2018-22", sel["year"] <= 2022), ("2023-26", sel["year"] >= 2023),
                   ("sinceApr26", sel["trade_date"] >= "2026-04-01")]:
        s = sel[m]
        row.append(f"{s['y'].mean()*100:.1f}%" if len(s) > 100 else "—")
    print("| " + " | ".join(row) + " |", flush=True)

print("\n| arm | 2018-22 top8 | 2023-26 top8 | sinceApr26 top8 |")
print("|---|---|---|---|")
p_price = run(PRICE, "PRICE")
report("PRICE (baseline)", top8(p_price, "raw"))
report("PRICE + VETO", top8(p_price, "veto"))
report("PRICE + VETO + BOOST", top8(p_price, "veto_boost"))
p_feat = run(FEAT, "FEAT")
report("FEAT (signed-print features)", top8(p_feat, "raw"))
report("FEAT + VETO + BOOST", top8(p_feat, "veto_boost"))
print("AB RULES COMPLETE", flush=True)
