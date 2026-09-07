"""FRESH vs EXTENDED leaders (registered 2026-09-08) + live yet-to-run screen.

Historical: hot-industry (grp ret60 top decile) top-3 leaders split by trailing
252d return <= 50% (FRESH) vs > 50% (EXTENDED). Outcomes at 126td.
Live: today's fresh leaders, laggard+promoter-up names, and rising (near-hot)
industries' strongest yet-to-run members (labeled UNTESTED).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "low", "close", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["ret252"] = g["close"].pct_change(252)
px["hi126"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=60).max()[::-1])
px["lo126"] = g["low"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=60).min()[::-1])
px["cl126"] = g["close"].transform(lambda s: s.shift(-126))
px["adv"] = px["avg_traded_value_20d"] / 1e7

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
px["ind"] = px["symbol"].map(imap)

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna()
        & px["ret60"].notna() & px["ret252"].notna()].copy()
grp = (wk.groupby(["trade_date", "ind"]).agg(gr=("ret60", "mean"), nm=("ret60", "size"))
         .reset_index())
grp = grp[grp["nm"] >= 5]
grp["gd"] = grp.groupby("trade_date")["gr"].transform(lambda s: s.rank(pct=True))
wk = wk.merge(grp[["trade_date", "ind", "gd", "gr"]], on=["trade_date", "ind"], how="inner")
wk["top3"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False) <= 3
wk["x2"] = wk["hi126"].notna() & (wk["hi126"] / wk["close"] - 1 >= 1.0)
wk["x50"] = wk["hi126"].notna() & (wk["hi126"] / wk["close"] - 1 >= 0.50)
wk["end126"] = (wk["cl126"] / wk["close"] - 1) * 100
wk["trough"] = (wk["lo126"] / wk["close"] - 1) * 100
wk["year"] = wk["trade_date"].dt.year
wk["week"] = wk["trade_date"].dt.to_period("W")

H = wk[wk["hi126"].notna()]
led = H[(H["gd"] >= 0.9) & H["top3"]]
print(f"\n{'CELL':<26s}| era      |     n | P2x/126 | P50 | wk-end | medTr", flush=True)
for era, em in [("disc<=22", led["year"] <= 2022), ("conf>=23", led["year"] >= 2023)]:
    L = led[em]
    for name, m in [("LEADER all", pd.Series(True, index=L.index)),
                    ("FRESH (ret252<=50%)", L["ret252"] <= 0.50),
                    ("EXTENDED (ret252>50%)", L["ret252"] > 0.50)]:
        S = L[m]
        if len(S) < 60:
            print(f"{name:<26s}| {era} | {len(S):>5,} | too few"); continue
        w = S.groupby("week")["end126"].mean()
        print(f"{name:<26s}| {era} | {len(S):>5,} |  {S['x2'].mean()*100:5.2f}% | {S['x50'].mean()*100:4.1f}% |"
              f" {w.mean():+6.2f}% | {S['trough'].median():+5.1f}%", flush=True)

# ---------------- live screen ----------------
last_dt = px["trade_date"].max()
snap = px[(px["trade_date"] == last_dt) & (px["adv"] >= 1.5) & (px["close"] > 25)].dropna(subset=["ret60", "ret252", "ind"]).copy()
sg = (snap.groupby("ind").agg(gr=("ret60", "mean"), nm=("ret60", "size"))
          .query("nm >= 5"))
sg["gd"] = sg["gr"].rank(pct=True)
snap = snap.merge(sg[["gd", "gr"]], on="ind", how="inner")
snap["rk"] = snap.groupby("ind")["ret60"].rank(ascending=False)

print(f"\n===== LIVE ({last_dt.date()}): FRESH leaders in HOT industries (validated-adjacent) =====", flush=True)
fresh = snap[(snap["gd"] >= 0.9) & (snap["rk"] <= 3) & (snap["ret252"] <= 0.50)]
for _, r in fresh.sort_values("gr", ascending=False).iterrows():
    print(f"  {r['symbol']:<14}{r['ind'][:32]:<33} grp60 {r['gr']*100:+5.1f}% · own60 {r['ret60']*100:+6.1f}% · 12M {r['ret252']*100:+6.1f}%", flush=True)

print(f"\n===== LIVE: RISING industries (decile 0.75-0.9, UNTESTED cell) — strongest yet-to-run members =====", flush=True)
rising = snap[(snap["gd"] >= 0.75) & (snap["gd"] < 0.9) & (snap["rk"] <= 3) & (snap["ret252"] <= 0.30)]
for _, r in rising.sort_values("gr", ascending=False).head(12).iterrows():
    print(f"  {r['symbol']:<14}{r['ind'][:32]:<33} grp60 {r['gr']*100:+5.1f}% · own60 {r['ret60']*100:+6.1f}% · 12M {r['ret252']*100:+6.1f}%", flush=True)

print(f"\n===== LIVE: HOT-group laggards with promoter buying (the +19.2%-mean cell) =====", flush=True)
try:
    shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
    shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
    shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
    shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
    shp["d"] = shp.groupby("symbol")["promoter_pct"].diff()
    pmap = shp.groupby("symbol")["d"].last()
    lag = snap[(snap["gd"] >= 0.9) & (snap.groupby("ind")["ret60"].rank(pct=True) <= 0.5)].copy()
    lag["prom_d"] = lag["symbol"].map(pmap)
    lag = lag[lag["prom_d"] > 0]
    if len(lag) == 0:
        print("  (none currently)", flush=True)
    for _, r in lag.sort_values("prom_d", ascending=False).iterrows():
        print(f"  {r['symbol']:<14}{r['ind'][:32]:<33} own60 {r['ret60']*100:+6.1f}% · 12M {r['ret252']*100:+6.1f}% · promΔ +{r['prom_d']:.2f}", flush=True)
except Exception as e:
    print("  SHP unavailable:", str(e)[:60], flush=True)
print("\nFRESH LEADER COMPLETE", flush=True)
