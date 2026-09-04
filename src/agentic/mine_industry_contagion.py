"""INDUSTRY-CONTAGION MINER — pre-registered 2026-09-03 (user hypothesis).

Hypothesis: industry-first selection (hot theme → quality laggards within it)
finds 2x-in-6-months. Doubler audit support: 7 themes held 2/3 of the 78.

Frame: weekly grid 2016-06→2026-08, investable (ADV>=5cr, >Rs50). Industry =
NSE smIndustry (mode per symbol from the announcements corpus). Groups with
>=5 investable members that week. Horizon: 126td (~6 months).

Cells (registered before results):
  BASE            : any investable stock-week
  IND-HOT         : stock's group in top decile by group 60d equal-weight return
  IND-HOT-LEADER  : hot group AND stock in top-3 of group by own 60d return (chase)
  IND-HOT-LAGGARD : hot group AND stock's own 60d return in bottom half of group
  IND-COLD        : group in bottom decile (control)
  +PROM-UP        : laggard + promoter stake delta > 0 (SHP known at T; 2021+ only)
  +NO-PLEDGE      : laggard + zero pledge-increase filings trailing 12m (2016+)
Outcomes: P(2x/126td), P(+50%/126td), weekly-cohort mean end-126td, med trough.
Ship bar: LAGGARD cell conf-era P(2x/126) >= 1.5x base with n>=200.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("loading…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "low", "close", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["adv"] = px["avg_traded_value_20d"] / 1e7
def fwd(col, n, how):
    return g[col].transform(lambda s: getattr(s.shift(-1)[::-1].rolling(n, min_periods=min(30, n)), how)()[::-1])
px["hi126"] = fwd("high", 126, "max")
px["lo126"] = fwd("low", 126, "min")
px["cl126"] = g["close"].transform(lambda s: s.shift(-126))

print("industry map…", flush=True)
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
px["ind"] = px["symbol"].map(imap)
print(f"  mapped {px['ind'].notna().mean()*100:.0f}% of rows · {imap.nunique()} industries", flush=True)

days = sorted(px["trade_date"].unique())
weekly = set(days[::5])
grid = px[px["trade_date"].isin(weekly) & (px["adv"] >= 5) & (px["close"] > 50)
          & (px["trade_date"] >= "2016-06-01") & px["hi126"].notna()
          & px["ind"].notna() & px["ret60"].notna()].copy()

print("group stats…", flush=True)
gs = grid.groupby(["trade_date", "ind"])["ret60"].agg(["mean", "count", "rank"]).reset_index() if False else None
grp = grid.groupby(["trade_date", "ind"]).agg(grp_ret60=("ret60", "mean"), n_mem=("ret60", "size")).reset_index()
grp = grp[grp["n_mem"] >= 5]
grp["grp_decile"] = grp.groupby("trade_date")["grp_ret60"].transform(lambda s: s.rank(pct=True))
grid = grid.merge(grp, on=["trade_date", "ind"], how="inner")
grid["own_rank_in_grp"] = grid.groupby(["trade_date", "ind"])["ret60"].rank(pct=True)
grid["own_top3"] = grid.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False) <= 3

print("quality overlays…", flush=True)
shp = pd.read_parquet(ROOT / "data/derived/stock_shareholding.parquet")
shp["qe"] = pd.to_datetime(shp["quarter_end"], errors="coerce")
shp["promoter_pct"] = pd.to_numeric(shp["promoter_pct"], errors="coerce")
shp = shp.dropna(subset=["qe"]).sort_values(["symbol", "qe"])
shp["prom_delta"] = shp.groupby("symbol")["promoter_pct"].diff()
shp["known"] = shp["qe"] + pd.Timedelta(days=45)
sh = shp.dropna(subset=["prom_delta"]).sort_values("known")[["symbol", "known", "prom_delta"]]
grid = grid.sort_values("trade_date")
grid = pd.merge_asof(grid, sh.rename(columns={"known": "trade_date"}), on="trade_date",
                     by="symbol", direction="backward", tolerance=pd.Timedelta(days=200))
ann2 = pd.read_parquet(ROOT / "data/events_full_history/normalized/stock_announcements.parquet",
                       columns=["symbol", "event_date", "is_pledge_change"])
ah2 = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet", columns=["symbol", "sort_date", "desc"])
ah2["d"] = pd.to_datetime(ah2["sort_date"], errors="coerce").dt.normalize()
pledge = ah2[ah2["desc"].fillna("").str.contains("pledge", case=False)].dropna(subset=["d"])
pm = {s: gg["d"].values.astype("datetime64[ns]") for s, gg in pledge.groupby("symbol")}
t64 = grid["trade_date"].values.astype("datetime64[ns]")
def had_pledge(s, t):
    a = pm.get(s)
    if a is None:
        return False
    return np.searchsorted(a, t, "right") > np.searchsorted(a, t - np.timedelta64(365, "D"), "right") - 1 and \
           np.searchsorted(a, t, "right") - np.searchsorted(a, t - np.timedelta64(365, "D"), "right") > 0
grid["pledge_12m"] = [had_pledge(s, t) for s, t in zip(grid["symbol"], t64)]

grid["t2x"] = grid["hi126"] / grid["close"] - 1 >= 1.0
grid["t50"] = grid["hi126"] / grid["close"] - 1 >= 0.50
grid["end126"] = (grid["cl126"] / grid["close"] - 1) * 100
grid["trough"] = (grid["lo126"] / grid["close"] - 1) * 100
grid["year"] = grid["trade_date"].dt.year
grid["week"] = grid["trade_date"].dt.to_period("W")

CELLS = {
    "BASE (any investable)": pd.Series(True, index=grid.index),
    "IND-HOT (group top decile 60d)": grid["grp_decile"] >= 0.9,
    "IND-HOT-LEADER (top-3 in hot grp)": (grid["grp_decile"] >= 0.9) & grid["own_top3"],
    "IND-HOT-LAGGARD (bottom half)": (grid["grp_decile"] >= 0.9) & (grid["own_rank_in_grp"] <= 0.5),
    "IND-COLD (bottom decile)": grid["grp_decile"] <= 0.1,
    "LAGGARD + promoter delta>0": (grid["grp_decile"] >= 0.9) & (grid["own_rank_in_grp"] <= 0.5) & (grid["prom_delta"] > 0),
    "LAGGARD + no pledge 12m": (grid["grp_decile"] >= 0.9) & (grid["own_rank_in_grp"] <= 0.5) & (~grid["pledge_12m"]),
}
print(f"\ngrid {len(grid):,} rows\n")
print(f"{'CELL':<36s}| era      |     n | P(2x/126) | P(+50/126) | wk-end126 | medTr | yrs+")
for name, m in CELLS.items():
    for era, em in [("disc<=22", grid["year"] <= 2022), ("conf>=23", grid["year"] >= 2023)]:
        s = grid[m.fillna(False) & em]
        if len(s) < 60:
            print(f"{name:<36s}| {era} | {len(s):>5,} | too few"); continue
        wk = s.groupby("week")["end126"].mean(); yr = s.groupby("year")["end126"].mean()
        print(f"{name:<36s}| {era} | {len(s):>5,} |    {s['t2x'].mean()*100:5.1f}% |     {s['t50'].mean()*100:5.1f}% |   {wk.mean():+6.2f}% | {s['trough'].median():+5.1f}% | {(yr>0).mean()*100:3.0f}%", flush=True)
print("INDUSTRY CONTAGION MINER COMPLETE", flush=True)
