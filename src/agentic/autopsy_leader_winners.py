"""Winners autopsy INSIDE the hot-industry leader cell (reg. 2026-09-08).
For every ex-ante feature: P(2x/126td) and P(+50%/126td) per bucket, both eras.
Output ranked by conf-era P2x separation. Exploratory — no ship bar.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Code/Zoom")

print("prep…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "close", "return_1d", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["ret252"] = g["close"].pct_change(252)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["hi126"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=100).max()[::-1])
px["adv"] = px["avg_traded_value_20d"] / 1e7

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
px["ind"] = px["symbol"].map(imap)

# TTM + SUE point-in-time
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].copy()
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
q = q.sort_values(["symbol", "quarter_end"])
q["sue"] = q.groupby("symbol")["eps_basic"].diff(4) / q["close"] if "close" in q else np.nan
rows, sue_rows = [], []
for sym, gg in q.groupby("symbol"):
    qe = gg["quarter_end"].values; ea = gg["eps_basic"].values
    fd = pd.to_datetime(gg["filing_dt"]).values
    for i in range(3, len(gg)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) > 380:
            continue
        rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
    for i in range(4, len(gg)):
        sue_rows.append((sym, fd[i], ea[i] - ea[i - 4]))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")
sue = pd.DataFrame(sue_rows, columns=["symbol", "known", "sue_raw"]).sort_values("known")

fs = pd.read_parquet(ROOT / "data/derived/filing_scores.parquet")
fs["event_dt"] = pd.to_datetime(fs["event_dt"])
neg = fs[fs["catdir"] == -1][["symbol", "event_dt"]].sort_values("event_dt")
pos = fs[fs["catdir"] == 1][["symbol", "event_dt"]].sort_values("event_dt")

shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_d"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["known"] = shp["qe"] + pd.Timedelta(days=45)
pm = shp.dropna(subset=["prom_d"]).sort_values("known")[["symbol", "known", "prom_d"]]

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna()
        & px["ret60"].notna() & px["hi126"].notna()].copy()
for src, col, tol in [(ttm.rename(columns={"known": "trade_date"}), "ttm", 200),
                      (sue.rename(columns={"known": "trade_date"}), "sue_raw", 120),
                      (pm.rename(columns={"known": "trade_date"}), "prom_d", 200)]:
    wk = pd.merge_asof(wk.sort_values("trade_date"), src.sort_values("trade_date"),
                       on="trade_date", by="symbol", direction="backward",
                       tolerance=pd.Timedelta(days=tol))
wk["pe"] = np.where(wk["ttm"] > 0, wk["close"] / wk["ttm"], np.nan)
med = (wk.dropna(subset=["pe"]).groupby(["trade_date", "ind"])["pe"]
         .agg(["median", "size"]).reset_index().rename(columns={"median": "ind_pe"}))
wk = wk.merge(med[med["size"] >= 5][["trade_date", "ind", "ind_pe"]],
              on=["trade_date", "ind"], how="left")
wk["pe_ind"] = wk["pe"] / wk["ind_pe"]
wk["sue_ps"] = wk["sue_raw"] / wk["close"]

grp = (wk.groupby(["trade_date", "ind"]).agg(gr=("ret60", "mean"), nm=("ret60", "size"))
         .reset_index())
grp = grp[grp["nm"] >= 5]
grp["gd"] = grp.groupby("trade_date")["gr"].transform(lambda s: s.rank(pct=True))
wk = wk.merge(grp[["trade_date", "ind", "gd", "gr", "nm"]], on=["trade_date", "ind"], how="inner")
wk["rk"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False)
led = wk[(wk["gd"] >= 0.9) & (wk["rk"] <= 3)].copy()

# landmine/booster counts 90d (loop only over the leader rows — small)
for nm2, ev in [("neg90", neg), ("pos90", pos)]:
    evd = {s: gg["event_dt"].values.astype("datetime64[ns]") for s, gg in ev.groupby("symbol")}
    t64 = led["trade_date"].values.astype("datetime64[ns]")
    syms = led["symbol"].values
    out = np.zeros(len(led))
    for i in range(len(led)):
        a = evd.get(syms[i])
        if a is None:
            continue
        out[i] = (np.searchsorted(a, t64[i], "right")
                  - np.searchsorted(a, t64[i] - np.timedelta64(90, "D"), "right"))
    led[nm2] = out

led["x2"] = led["hi126"] / led["close"] - 1 >= 1.0
led["x50"] = led["hi126"] / led["close"] - 1 >= 0.50
led["year"] = led["trade_date"].dt.year

def B(s, edges, labels):
    return pd.cut(s, edges, labels=labels)

led["conf"] = led["year"] >= 2023
FEATURES = {
    "pe_ind": B(led["pe_ind"], [0, .35, .7, 1, 1.5, 99], ["<0.35", ".35-.7", ".7-1", "1-1.5", ">1.5"]),
    "profitability": np.where(led["ttm"].isna(), "no-data", np.where(led["ttm"] > 0, "PROFIT", "LOSS")),
    "sue_latest": B(led.groupby(led["conf"])["sue_ps"].transform(lambda s: s.rank(pct=True)),
                    [0, .33, .67, 1], ["low", "mid", "high"]),
    "prom_delta": np.where(led["prom_d"].isna(), "no-data",
                  np.where(led["prom_d"] > 0.25, "buying", np.where(led["prom_d"] < -0.25, "selling", "flat"))),
    "adv_size": B(led["adv"], [0, 10, 30, 100, 1e9], ["5-10cr", "10-30cr", "30-100cr", ">100cr"]),
    "price": B(led["close"], [0, 100, 300, 1000, 1e9], ["50-100", "100-300", "300-1k", ">1k"]),
    "own_ret60": B(led["ret60"], [0, .3, .6, 1.0, 99], ["<30%", "30-60%", "60-100%", ">100%"]),
    "freshness": B(led["ret252"].fillna(0), [-1, .2, .5, 1.0, 99], ["12M<20%", "20-50%", "50-100%", ">100%"]),
    "rank_in_grp": led["rk"].astype(int).astype(str),
    "grp_heat": B(led["gr"], [0, .15, .25, 9], ["grp<15%", "15-25%", ">25%"]),
    "grp_size": B(led["nm"], [0, 7, 15, 999], ["5-7", "8-15", ">15"]),
    "landmine90": np.where(led["neg90"] > 0, "yes", "no"),
    "booster90": np.where(led["pos90"] > 0, "yes", "no"),
    "dvol": B(led["dvol"], [0, .02, .035, 9], ["calm", "mid", "wild"]),
}
print(f"\nleader cell rows: {len(led):,} (conf {led['conf'].sum():,})", flush=True)
print(f"{'FEATURE=bucket':<28s}| era      |     n | P2x   | P50   | lift2x", flush=True)
for era, em in [("conf>=23", led["conf"]), ("disc<=22", ~led["conf"])]:
    E = led[em]
    b2, b5 = E["x2"].mean(), E["x50"].mean()
    print(f"--- {era}: base P2x {b2*100:.1f}% · P50 {b5*100:.1f}% ---", flush=True)
    for fname, fv in FEATURES.items():
        v = pd.Series(np.asarray(fv), index=led.index)[em]
        for bucket in v.dropna().unique():
            S = E[v == bucket]
            if len(S) < 60:
                continue
            print(f"{fname+'='+str(bucket):<28s}| {era} | {len(S):>5,} | {S['x2'].mean()*100:5.1f}% | "
                  f"{S['x50'].mean()*100:5.1f}% | {S['x2'].mean()/b2:5.2f}x", flush=True)
print("\nLEADER WINNERS AUTOPSY COMPLETE", flush=True)
