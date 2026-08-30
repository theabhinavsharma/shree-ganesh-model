"""1.5x CONFIDENCE MODEL — P(+50% within 100td from ENTRY) for confirmed ignitions.

Pre-registered 2026-08-31. Event = thrust (+8%/3x) with day-5 confirmation (close
above thrust close, mean vol d+1..5 >= 1.5x). Universe ADV>=1.5cr & close>25 —
the sleeve's expanded universe. ENTRY = confirmation-day close; the label and all
returns are measured from entry, so the model's P is the number you can act on.

Features at entry (all knowable): thrust anatomy (size, volume, delivery),
confirmation anatomy (5d return/volume/delivery), structure (off-52wk, age, ADV,
band flag, dvol), market context (breadth, mkt 20d), event corpus (post-print flag,
order-win flag, promoter PIT buys 90d, SHP promoter delta, days since print,
filings 30d).

Walk-forward: yearly 2018→2026, train on events whose 100td label window closes
before the scored year; LGBM + isotonic on train tail. Output:
  research/model_150/oos_predictions.parquet   (all OOS scored events)
  research/model_150/decile_table.txt          (confidence → realized P)
  research/model_150/live_scores_<date>.csv    (today's open sleeve signals scored
                                                by a model trained on all history)
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT = ROOT / "research/model_150"
OUT.mkdir(parents=True, exist_ok=True)

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "low", "close", "return_1d", "volume_vs_20d",
             "delivery_pct", "avg_delivery_pct_20d", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
g = px.groupby("symbol", sort=False)
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["age"] = g.cumcount()
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["hi252"] = g["close"].transform(lambda s: s.rolling(252, min_periods=60).max())
px["off52"] = px["close"] / px["hi252"] - 1
mkt = px[px["adv"] >= 5].groupby("trade_date").agg(
    mkt_breadth=("off52", lambda s: (s > -0.1).mean()),
    mkt_ret20=("return_1d", "mean")).reset_index()
px = px.merge(mkt, on="trade_date", how="left")

print("events…", flush=True)
idx = px.index[(px["return_1d"] >= 0.08) & (px["volume_vs_20d"] >= 3)
               & (px["adv"] >= 1.5) & (px["close"] > 25) & (px["trade_date"] >= "2016-06-01")]
rows = []
sym_arr = px["symbol"].values
for i in idx:
    s = sym_arr[i]
    if i + 5 >= len(px) or sym_arr[i + 5] != s:
        continue
    thr, conf = px.iloc[i], px.iloc[i + 5]
    seg = px.iloc[i + 1:i + 6]
    if conf["close"] <= thr["close"] or seg["volume_vs_20d"].mean() < 1.5:
        continue
    fut = px.iloc[i + 6:i + 106]
    fut = fut[fut["symbol"] == s]
    if len(fut) < 60:
        continue
    entry = conf["close"]
    rows.append(dict(symbol=s, thrust_date=thr["trade_date"], entry_date=conf["trade_date"],
        entry=entry,
        y150=float(fut["high"].max() / entry - 1 >= 0.50),
        end100=(fut["close"].iloc[-1] / entry - 1) * 100,
        thrust_ret=thr["return_1d"], thrust_vol=thr["volume_vs_20d"], thrust_dlv=thr["dlv"],
        ft_ret5=conf["close"] / thr["close"] - 1, ft_vol5=seg["volume_vs_20d"].mean(),
        ft_dlv5=seg["dlv"].mean(), off52=thr["off52"], age=thr["age"], adv=thr["adv"],
        dvol=thr["dvol"], band=float(thr["adv"] < 5 or thr["close"] <= 50),
        mkt_breadth=thr["mkt_breadth"], mkt_ret20=thr["mkt_ret20"]))
ev = pd.DataFrame(rows)
# dedup one per symbol per 45td
ev = ev.sort_values(["symbol", "thrust_date"])
ev["gap"] = ev.groupby("symbol")["thrust_date"].diff().dt.days.fillna(999)
ev = ev[ev["gap"] > 45].drop(columns="gap").reset_index(drop=True)
print(f"confirmed ignition events: {len(ev):,} · base P(+50%/100td from entry): {ev['y150'].mean()*100:.1f}%", flush=True)

print("event-corpus features…", flush=True)
cal = pd.read_parquet(ROOT / "data/derived/results_calendar_history.parquet")
cal["d"] = pd.to_datetime(cal["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
cal2 = pd.read_parquet(ROOT / "data/derived/results_calendar_integrated.parquet")
cal2["d"] = pd.to_datetime(cal2["broadcast"], format="%d-%b-%Y %H:%M:%S", errors="coerce")
prints = pd.concat([cal[["symbol", "d"]], cal2[["symbol", "d"]]]).dropna()
prints["d"] = prints["d"].dt.normalize()
pm = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in prints.groupby("symbol")}
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "sort_date", "desc"])
ah["d"] = pd.to_datetime(ah["sort_date"], errors="coerce").dt.normalize()
ORDER_RE = re.compile(r"bagging|receipt of order|award.{0,12}(?:order|contract)|work order|letter of (?:intent|award)|purchase order|order (?:received|book|win)", re.I)
om = {s: gg["d"].values.astype("datetime64[ns]")
      for s, gg in ah[ah["desc"].fillna("").str.contains(ORDER_RE)].dropna(subset=["d"]).groupby("symbol")}
am = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in ah.dropna(subset=["d"]).groupby("symbol")}
pit = pd.read_parquet(ROOT / "data/derived/pit_history.parquet")
pit["d"] = pd.to_datetime(pit["date"], errors="coerce", dayfirst=True).dt.normalize()
symc = [c for c in pit.columns if "symbol" in c.lower()][0]
pit["dpct"] = pd.to_numeric(pit.get("afterAcqSharesPer"), errors="coerce") - pd.to_numeric(pit.get("befAcqSharesPer"), errors="coerce")
bm = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in pit[pit["dpct"] > 0].dropna(subset=["d"]).groupby(symc)}
shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["known"] = shp["qe"] + pd.Timedelta(days=45)

def cnt(mp, s, t, lo, hi):
    a = mp.get(s)
    if a is None:
        return 0
    return int(np.searchsorted(a, t + np.timedelta64(hi, "D"), "right")
               - np.searchsorted(a, t + np.timedelta64(lo, "D"), "left"))

t64 = ev["thrust_date"].values.astype("datetime64[ns]")
ev["post_print"] = [cnt(pm, s, t, -5, 0) > 0 for s, t in zip(ev["symbol"], t64)]
ev["order_win"] = [cnt(om, s, t, -5, 5) > 0 for s, t in zip(ev["symbol"], t64)]
ev["prom_buys_90d"] = [cnt(bm, s, t, -90, 0) for s, t in zip(ev["symbol"], t64)]
ev["filings_30d"] = [cnt(am, s, t, -30, 0) for s, t in zip(ev["symbol"], t64)]
def days_since(mp, s, t):
    a = mp.get(s)
    if a is None:
        return np.nan
    i = np.searchsorted(a, t, "right")
    return float((t - a[i - 1]) / np.timedelta64(1, "D")) if i else np.nan
ev["days_since_print"] = [days_since(pm, s, t) for s, t in zip(ev["symbol"], t64)]
sh = shp.sort_values("known")[["symbol", "known", "prom_delta"]].rename(columns={"known": "entry_date"})
ev = pd.merge_asof(ev.sort_values("entry_date"), sh, on="entry_date", by="symbol",
                   direction="backward", tolerance=pd.Timedelta(days=200))

FEATS = ["thrust_ret", "thrust_vol", "thrust_dlv", "ft_ret5", "ft_vol5", "ft_dlv5",
         "off52", "age", "adv", "dvol", "band", "mkt_breadth", "mkt_ret20",
         "post_print", "order_win", "prom_buys_90d", "filings_30d",
         "days_since_print", "prom_delta"]
ev[["post_print", "order_win"]] = ev[["post_print", "order_win"]].astype(float)
ev["year"] = ev["entry_date"].dt.year

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

oos = []
for Y in range(2018, 2027):
    cutoff = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=160)
    tr = ev[ev["entry_date"] <= cutoff]
    sc = ev[ev["year"] == Y]
    if len(tr) < 500 or len(sc) == 0:
        continue
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=60, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=5,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(tr[FEATS], tr["y150"])
    tail = tr[tr["entry_date"] >= tr["entry_date"].quantile(0.7)]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(m.predict_proba(tail[FEATS])[:, 1], tail["y150"])
    d = sc.copy()
    d["p"] = iso.transform(m.predict_proba(sc[FEATS])[:, 1])
    oos.append(d)
    print(f"  {Y}: train {len(tr):,} → scored {len(sc):,}", flush=True)
oos = pd.concat(oos, ignore_index=True)
oos.to_parquet(OUT / "oos_predictions.parquet", index=False)

base = oos["y150"].mean()
ap = average_precision_score(oos["y150"], oos["p"])
print(f"\nOOS events {len(oos):,} · base {base*100:.1f}% · AUC-PR {ap*100:.1f}% ({ap/base:.2f}x)")
oos["dec"] = pd.qcut(oos["p"], 10, labels=False, duplicates="drop")
tbl = oos.groupby("dec").agg(n=("y150", "size"), pred=("p", "mean"),
                             real=("y150", "mean"), end100=("end100", "mean"))
lines = ["confidence decile → predicted vs realized P(+50%/100td) · mean end-100td ret"]
for dec, r in tbl.iterrows():
    lines.append(f"  D{int(dec)+1:>2d}: n={int(r['n']):>4d}  pred {r['pred']*100:5.1f}%  real {r['real']*100:5.1f}%  end100 {r['end100']:+6.1f}%")
print("\n".join(lines), flush=True)
(OUT / "decile_table.txt").write_text("\n".join(lines))

# ── live scoring: current model on today's open sleeve signals ──
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                       min_child_samples=60, feature_fraction=0.8, random_state=42,
                       verbose=-1, n_jobs=-1)
m.fit(ev[FEATS], ev["y150"])
tail = ev[ev["entry_date"] >= ev["entry_date"].quantile(0.7)]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(m.predict_proba(tail[FEATS])[:, 1], tail["y150"])
recent = ev[ev["entry_date"] >= ev["entry_date"].max() - pd.Timedelta(days=10)].copy()
if len(recent):
    recent["p"] = iso.transform(m.predict_proba(recent[FEATS])[:, 1])
    live = recent.sort_values("p", ascending=False)[["symbol", "thrust_date", "entry_date", "entry", "adv", "band", "p"]]
    live.to_csv(OUT / f"live_scores_{pd.Timestamp.today().date()}.csv", index=False)
    print("\nLIVE CONFIRMED-IGNITION SIGNALS (scored by full-history model):")
    for _, r in live.iterrows():
        print(f"  {r['symbol']:12s} entry {r['entry_date'].date()} @ {r['entry']:.1f}  ADV {r['adv']:.1f}cr {'BAND' if r['band'] else 'CORE'}  P(+50%/100td) = {r['p']*100:.0f}%")
imp = pd.Series(m.feature_importances_, index=FEATS).sort_values(ascending=False)
print("\nfeature importance:\n" + imp.to_string())
print("CONFIDENCE MODEL COMPLETE", flush=True)
