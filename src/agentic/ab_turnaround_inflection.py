"""TURNAROUND INFLECTION — registered 2026-09-08 (follow-up to 2x-by-PE study).

Cell: loss-maker (TTM EPS <= 0) whose LATEST quarterly EPS flipped positive
after >= 2 consecutive negative quarters (point-in-time, filing-date keyed).
Bar: P(2x/126td) >= 1.5x plain LOSS-maker, n >= 300, BOTH eras.

Controls: plain loss-maker (no inflection) · still-negative latest quarter ·
deepening losses (latest q worse than prior) · inflection with fresh print
(<= 30d since flip filing) vs stale. Outcomes: P(2x/126), P(+50/126),
weekly-cohort mean end-126, median trough. Universe-wide law applies.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("quarters…", flush=True)
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

# per-quarter state, point-in-time: ttm sign, latest-q sign, prior-2 signs
rows = []
for sym, gg in q.groupby("symbol"):
    qe = gg["quarter_end"].values
    eps = gg["eps_basic"].values
    fd = pd.to_datetime(gg["filing_dt"]).values
    for i in range(3, len(gg)):
        if (qe[i] - qe[i - 3]).astype("timedelta64[D]").astype(int) > 380:
            continue
        ttm = eps[i - 3:i + 1].sum()
        infl = eps[i] > 0 and eps[i - 1] <= 0 and eps[i - 2] <= 0
        deepening = eps[i] <= 0 and eps[i] < eps[i - 1] <= 0
        rows.append((sym, max(fd[i - 3:i + 1]), ttm, eps[i] > 0, infl, deepening))
st = pd.DataFrame(rows, columns=["symbol", "known", "ttm", "q_pos", "inflection", "deepening"])
st = st.sort_values("known")
print(f"  state rows {len(st):,} · inflections {st['inflection'].sum():,}", flush=True)

print("panel…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "high", "low", "close", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["hi126"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=60).max()[::-1])
px["lo126"] = g["low"].transform(lambda s: s.shift(-1)[::-1].rolling(126, min_periods=60).min()[::-1])
px["cl126"] = g["close"].transform(lambda s: s.shift(-126))

days = sorted(px["trade_date"].unique())
wk = px[(px["avg_traded_value_20d"] / 1e7 >= 5) & (px["close"] > 50)
        & px["trade_date"].isin(set(days[::5])) & (px["trade_date"] >= "2018-06-01")
        & px["hi126"].notna()].copy()
wk = pd.merge_asof(wk.sort_values("trade_date"),
                   st.rename(columns={"known": "trade_date"}).sort_values("trade_date"),
                   on="trade_date", by="symbol", direction="backward",
                   tolerance=pd.Timedelta(days=200))
# days since the state became known (for fresh-vs-stale inflection)
kn = st.rename(columns={"known": "kdt"})[["symbol", "kdt"]].sort_values("kdt")
kn["trade_date"] = kn["kdt"]
wk = pd.merge_asof(wk.sort_values("trade_date"), kn, on="trade_date", by="symbol",
                   direction="backward", tolerance=pd.Timedelta(days=200))
wk["days_since"] = (wk["trade_date"] - wk["kdt"]).dt.days

wk["x2"] = wk["hi126"] / wk["close"] - 1 >= 1.0
wk["x50"] = wk["hi126"] / wk["close"] - 1 >= 0.50
wk["end126"] = (wk["cl126"] / wk["close"] - 1) * 100
wk["trough"] = (wk["lo126"] / wk["close"] - 1) * 100
wk["year"] = wk["trade_date"].dt.year
wk["week"] = wk["trade_date"].dt.to_period("W")

loss = wk["ttm"] <= 0
CELLS = {
    "UNIVERSE": pd.Series(True, index=wk.index),
    "LOSS plain (no inflection)": loss & ~wk["inflection"].fillna(False),
    "LOSS deepening (q worse)": loss & wk["deepening"].fillna(False),
    "TURNAROUND INFLECTION": loss & wk["inflection"].fillna(False),
    "INFLECTION fresh (<=30d)": loss & wk["inflection"].fillna(False) & (wk["days_since"] <= 30),
    "INFLECTION stale (>30d)": loss & wk["inflection"].fillna(False) & (wk["days_since"] > 30),
}
print(f"\n{'CELL':<28s}| era      |      n | P2x/126 | P50/126 | wk-end126 | medTr | lift-vs-LOSS")
for era, em in [("disc<=22", wk["year"] <= 2022), ("conf>=23", wk["year"] >= 2023)]:
    E = wk[em]
    base_loss = E[loss[em.index] if False else (E["ttm"] <= 0) & ~E["inflection"].fillna(False)]["x2"].mean()
    for name, m in CELLS.items():
        S = wk[m.fillna(False) & em]
        if len(S) < 60:
            print(f"{name:<28s}| {era} | {len(S):>6,} | too few"); continue
        w = S.groupby("week")["end126"].mean()
        lift = S["x2"].mean() / base_loss if base_loss > 0 else np.nan
        print(f"{name:<28s}| {era} | {len(S):>6,} |  {S['x2'].mean()*100:5.2f}% |  {S['x50'].mean()*100:5.1f}% |"
              f"   {w.mean():+6.2f}% | {S['trough'].median():+5.1f}% | {lift:5.2f}x", flush=True)
print("\nTURNAROUND INFLECTION COMPLETE", flush=True)
