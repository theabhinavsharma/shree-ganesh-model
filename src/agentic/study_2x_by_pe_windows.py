"""2x-ers x starting valuation x window (21/42/63/126 td ~ 30/60/90/180 days).
Registered 2026-09-08 (study_2x_by_pe_multiwindow). Universe-wide law applies.

View A: P(2x within W) by valuation state at signal date.
View B: reverse autopsy — the starting-PE mix of realized 2x-ers vs universe.
States: LOSS (ttm<=0) · no-PE-data · deep-cheap pe_ind<0.7 · cheap 0.7-1 ·
        fair 1-1.5 · rich >1.5.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")
WINDOWS = {"30d~21td": 21, "60d~42td": 42, "90d~63td": 63, "180d~126td": 126}

print("ttm/pe…", flush=True)
q = pd.read_parquet(ROOT / "data/derived/pnl_quarterly.parquet").dropna(subset=["eps_basic", "filing_dt"])
q = q.sort_values("filing_dt").drop_duplicates(["symbol", "quarter_end", "basis"], keep="last")
lastq = q.groupby(["symbol", "basis"])["quarter_end"].max().unstack()
_o = pd.Timestamp("1900-01-01")
_c = lastq["con"] if "con" in lastq else pd.Series(_o, index=lastq.index)
_s = lastq["sa"] if "sa" in lastq else pd.Series(_o, index=lastq.index)
bmap = pd.Series(np.where(_c.fillna(_o) >= _s.fillna(_o), "con", "sa"), index=lastq.index)
q = q[q["basis"] == q["symbol"].map(bmap)].copy()
q["filing_dt"] = pd.to_datetime(q["filing_dt"]).dt.normalize()

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "close", "avg_traded_value_20d",
             "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")

fac = px[["symbol", "trade_date", "price_adjustment_factor_to_present"]].rename(
    columns={"trade_date": "filing_dt", "price_adjustment_factor_to_present": "fac"})
q = pd.merge_asof(q.sort_values("filing_dt"), fac.sort_values("filing_dt"), on="filing_dt",
                  by="symbol", direction="backward", tolerance=pd.Timedelta(days=30))
q["eps_adj"] = q["eps_basic"] * q["fac"].fillna(1.0)
q = q.sort_values(["symbol", "quarter_end"])
rows = []
for sym, gg in q.dropna(subset=["eps_adj"]).groupby("symbol"):
    qe = gg["quarter_end"].values; ea = gg["eps_adj"].values
    fd = pd.to_datetime(gg["filing_dt"]).values
    for i in range(3, len(gg)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) > 380:
            continue
        rows.append((sym, max(fd[i - 3:i + 1]), ea[i - 3:i + 1].sum()))
ttm = pd.DataFrame(rows, columns=["symbol", "known", "ttm"]).sort_values("known")

for lbl, n in WINDOWS.items():
    px[f"hi_{n}"] = g["high"].transform(
        lambda s, n=n: s.shift(-1)[::-1].rolling(n, min_periods=max(10, n // 2)).max()[::-1])

days = sorted(px["trade_date"].unique())
wk = px[(px["avg_traded_value_20d"] / 1e7 >= 5) & (px["close"] > 50)
        & px["trade_date"].isin(set(days[::5])) & (px["trade_date"] >= "2018-06-01")].copy()
wk = pd.merge_asof(wk.sort_values("trade_date"),
                   ttm.rename(columns={"known": "trade_date"}).sort_values("trade_date"),
                   on="trade_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=200))
wk["pe"] = np.where(wk["ttm"] > 0, wk["close"] / wk["ttm"], np.nan)
ah = pd.read_parquet(ROOT / "data/derived/announcements_historical.parquet",
                     columns=["symbol", "smIndustry"])
imap = (ah.dropna(subset=["smIndustry"]).groupby("symbol")["smIndustry"]
          .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None).dropna())
wk["ind"] = wk["symbol"].map(imap)
med = (wk.dropna(subset=["pe", "ind"]).groupby(["trade_date", "ind"])["pe"]
         .agg(["median", "size"]).reset_index().rename(columns={"median": "ind_pe"}))
wk = wk.merge(med[med["size"] >= 5][["trade_date", "ind", "ind_pe"]],
              on=["trade_date", "ind"], how="left")
wk["pe_ind"] = wk["pe"] / wk["ind_pe"]
wk["year"] = wk["trade_date"].dt.year

def state(r):
    if pd.notna(r["ttm"]) and r["ttm"] <= 0:
        return "LOSS-maker"
    x = r["pe_ind"]
    if pd.isna(x):
        return "no-PE-data"
    if x < 0.7:
        return "deep-cheap <0.7x"
    if x < 1.0:
        return "cheap 0.7-1x"
    if x < 1.5:
        return "fair 1-1.5x"
    return "rich >1.5x"

wk["state"] = wk.apply(state, axis=1)
ORDER = ["deep-cheap <0.7x", "cheap 0.7-1x", "fair 1-1.5x", "rich >1.5x", "LOSS-maker", "no-PE-data"]
print(f"grid {len(wk):,}; state mix: " +
      " · ".join(f"{s} {(wk['state']==s).mean()*100:.0f}%" for s in ORDER), flush=True)

for lbl, n in WINDOWS.items():
    wk[f"x2_{n}"] = wk[f"hi_{n}"] / wk["close"] - 1 >= 1.0

print("\n===== VIEW A: P(2x within window) by starting valuation =====", flush=True)
hdr = f"{'STATE':<18s}| era      |      n |" + "".join(f" {l:>10s} |" for l in WINDOWS)
print(hdr, flush=True)
for era, em in [("disc<=22", wk["year"] <= 2022), ("conf>=23", wk["year"] >= 2023)]:
    E = wk[em]
    for s in ["ALL"] + ORDER:
        S = E if s == "ALL" else E[E["state"] == s]
        if len(S) < 300:
            print(f"{s:<18s}| {era} | {len(S):>6,} | too few"); continue
        cells = "".join(f"     {S[f'x2_{n}'].mean()*100:5.2f}% |" for n in WINDOWS.values())
        print(f"{s:<18s}| {era} | {len(S):>6,} |{cells}", flush=True)

print("\n===== VIEW B: autopsy — starting-PE mix of realized 2x-ers =====", flush=True)
for era, em in [("disc<=22", wk["year"] <= 2022), ("conf>=23", wk["year"] >= 2023)]:
    E = wk[em]
    base_mix = E["state"].value_counts(normalize=True)
    for lbl, n in WINDOWS.items():
        X = E[E[f"x2_{n}"]]
        if len(X) < 60:
            print(f"[{era}] {lbl}: only {len(X)} 2x-ers — skipping mix"); continue
        mix = X["state"].value_counts(normalize=True)
        parts = " · ".join(
            f"{s}: {mix.get(s,0)*100:.0f}% ({mix.get(s,0)/max(base_mix.get(s,1e-9),1e-9):.1f}x)"
            for s in ORDER if base_mix.get(s, 0) > 0.01)
        med_pe = X["pe_ind"].median()
        print(f"[{era}] {lbl} (n={len(X):,}, med pe_ind {med_pe:.2f}): {parts}", flush=True)

print("\n2X-BY-PE WINDOWS COMPLETE", flush=True)
