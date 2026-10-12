"""Deep-cheap leader bands (registered 2026-09-08): leader sleeve restricted to
pe_ind bands — the UFLEX/MMFL question. Same 126td frame as sim_leader_sleeve."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Code/Zoom")
COST = 0.5

print("prep…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "close", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret60"] = g["close"].pct_change(60)
px["hi126"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=100).max()[::-1])
px["cl126"] = g["close"].transform(lambda s: s.shift(-126))
px["adv"] = px["avg_traded_value_20d"] / 1e7

ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
px["ind"] = px["symbol"].map(imap)

q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].copy()
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()
rows = []
for sym, gg in q.sort_values("quarter_end").groupby("symbol"):
    qe = gg["quarter_end"].values; ea = gg["eps_basic"].values
    fd = pd.to_datetime(gg["filing_dt"]).values
    for i in range(3, len(gg)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) > 380:
            continue
        rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")

days = sorted(px["trade_date"].unique())
wk = px[(px["adv"] >= 5) & (px["close"] > 50) & px["trade_date"].isin(set(days[::5]))
        & (px["trade_date"] >= "2016-06-01") & px["ind"].notna()
        & px["ret60"].notna() & px["hi126"].notna() & px["cl126"].notna()].copy()
wk = pd.merge_asof(wk.sort_values("trade_date"),
                   ttm.rename(columns={"known": "trade_date"}).sort_values("trade_date"),
                   on="trade_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=200))
wk["pe"] = np.where(wk["ttm"] > 0, wk["close"] / wk["ttm"], np.nan)
med = (wk.dropna(subset=["pe"]).groupby(["trade_date", "ind"])["pe"]
         .agg(["median", "size"]).reset_index().rename(columns={"median": "ind_pe"}))
wk = wk.merge(med[med["size"] >= 5][["trade_date", "ind", "ind_pe"]],
              on=["trade_date", "ind"], how="left")
wk["pe_ind"] = wk["pe"] / wk["ind_pe"]

grp = (wk.groupby(["trade_date", "ind"]).agg(gr=("ret60", "mean"), nm=("ret60", "size"))
         .reset_index())
grp = grp[grp["nm"] >= 5]
grp["gd"] = grp.groupby("trade_date")["gr"].transform(lambda s: s.rank(pct=True))
wk = wk.merge(grp[["trade_date", "ind", "gd"]], on=["trade_date", "ind"], how="inner")
wk["rk"] = wk.groupby(["trade_date", "ind"])["ret60"].rank(ascending=False)
led = wk[(wk["gd"] >= 0.9) & (wk["rk"] <= 3)].copy()
led["net"] = (led["cl126"] / led["close"] - 1) * 100 - COST
led["x2"] = led["hi126"] / led["close"] - 1 >= 1.0
led["year"] = led["trade_date"].dt.year

BANDS = {
    "pe_ind < 0.35 (UFLEX/MMFL zone)": led["pe_ind"] < 0.35,
    "pe_ind 0.35-0.70": led["pe_ind"].between(0.35, 0.70),
    "pe_ind 0.70-1.00": led["pe_ind"].between(0.70, 1.00),
    "pe_ind < 1 (shipped-bar cell)": led["pe_ind"] < 1,
    "pe_ind 1-1.5": led["pe_ind"].between(1.0, 1.5),
}
print(f"\n{'BAND':<32s}| era      | trades | tr/yr | mean/tr | med/tr | win% | P2x | worst coh")
for name, m in BANDS.items():
    V = led[m.fillna(False)]
    for era, em in [("disc<=22", V["year"] <= 2022), ("conf>=23", V["year"] >= 2023)]:
        S = V[em]
        if len(S) < 60:
            print(f"{name:<32s}| {era} | {len(S):>6,} | TOO THIN"); continue
        coh = S.groupby("trade_date")["net"].mean()
        yrs = max((S["trade_date"].max() - S["trade_date"].min()).days / 365.25, 0.1)
        print(f"{name:<32s}| {era} | {len(S):>6,} | {len(S)/yrs:>5.0f} | {S['net'].mean():>+6.2f}% | "
              f"{S['net'].median():>+6.2f}% | {(S['net']>0).mean()*100:>4.1f} | {S['x2'].mean()*100:>4.1f} | "
              f"{coh.min():>+7.1f}%", flush=True)
print("\nDEEP-CHEAP BANDS COMPLETE", flush=True)
