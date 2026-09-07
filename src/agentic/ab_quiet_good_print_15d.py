"""M1 — QUIET GOOD PRINT overlay on the 15d z-band pool. Registered 2026-09-07
(ab_text_fundamentals_15d_REGISTERED): EPS acceleration top-tercile whose print
the market SHRUGGED (|reaction| < 3%, vol < 2x) within 60d before signal.
First principles: unpriced fundamental improvement inside a quiet tape = coiled
spring WITH a reason. Bar: +4pp pool touch in BOTH eras.

Surprise metric: SUE_price = (eps_q - eps_{q-4}) / price_at_filing (per-share,
CA-factor adjusted), terciled within each filing-quarter cohort (point-in-time).
Controls also reported: quiet-BAD-print (bottom tercile, muted) and loud prints.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("eps + surprises…", flush=True)
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
    columns=["symbol", "trade_date", "close", "high", "return_1d", "rsi_14_daily", "return_20d",
             "volume_vs_20d", "avg_traded_value_20d", "delivery_pct", "avg_delivery_pct_20d",
             "price_adjustment_factor_to_present"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")

# factor-adjust EPS; price + reaction at/after filing
fac = px[["symbol", "trade_date", "price_adjustment_factor_to_present", "close",
          "return_1d", "volume_vs_20d"]].rename(columns={"trade_date": "filing_dt"})
q = pd.merge_asof(q.sort_values("filing_dt"), fac.sort_values("filing_dt"), on="filing_dt",
                  by="symbol", direction="forward", tolerance=pd.Timedelta(days=5))
q["eps_adj"] = q["eps_basic"] * q["price_adjustment_factor_to_present"].fillna(1.0)
q = q.sort_values(["symbol", "quarter_end"])
q["eps_yoy_diff"] = q.groupby("symbol")["eps_adj"].diff(4)
q["sue"] = q["eps_yoy_diff"] / q["close"]          # surprise scaled by price at filing
q = q.dropna(subset=["sue", "return_1d"])
q["cohort"] = q["filing_dt"].dt.to_period("Q")
q["sue_ter"] = q.groupby("cohort")["sue"].transform(lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
q["muted"] = (q["return_1d"].abs() < 0.03) & (q["volume_vs_20d"] < 2)
q["loud_up"] = (q["return_1d"] >= 0.05) & (q["volume_vs_20d"] >= 2)
print(f"  prints with SUE: {len(q):,} · muted {q['muted'].mean()*100:.0f}% · "
      f"quiet-good {(q['muted'] & (q['sue_ter']==2)).mean()*100:.1f}%", flush=True)

print("pool…", flush=True)
px["ret5"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
mu = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
sd = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
rsi_z = (px["rsi_14_daily"] - mu) / sd
r20_z = px["return_20d"] / (px["dvol"] * np.sqrt(20))
r5_z = px["ret5"] / (px["dvol"] * np.sqrt(5))
px["hi15"] = g["high"].transform(lambda s: s.shift(-1)[::-1].rolling(15, min_periods=8).max()[::-1])
days = sorted(px["trade_date"].unique())
Z = (rsi_z.between(-1.0, 0.25) & r20_z.between(-1.0, 0.75) & r5_z.between(-1.0, 1.0)
     & (px["avg_traded_value_20d"] / 1e7 >= 5) & (px["close"] > 50)
     & (px["volume_vs_20d"] < 2) & px["trade_date"].isin(set(days[::5]))
     & (px["trade_date"] >= "2018-06-01") & px["hi15"].notna())
INV = ((px["avg_traded_value_20d"] / 1e7 >= 5) & (px["close"] > 50)
       & px["trade_date"].isin(set(days[::5])) & (px["trade_date"] >= "2018-06-01")
       & px["hi15"].notna())
uni = px[INV][["symbol", "trade_date", "close", "hi15"]].copy()
pool = px[Z][["symbol", "trade_date", "close", "hi15"]].copy()
pool["touch"] = pool["hi15"] / pool["close"] - 1 >= 0.05
pool["year"] = pool["trade_date"].dt.year

# join: most recent print within 60d before signal
pr = q[["symbol", "filing_dt", "sue_ter", "muted", "loud_up"]].rename(
    columns={"filing_dt": "trade_date"}).sort_values("trade_date")
pool = pd.merge_asof(pool.sort_values("trade_date"), pr, on="trade_date", by="symbol",
                     direction="backward", tolerance=pd.Timedelta(days=60))

CELLS = {
    "POOL (all)": pd.Series(True, index=pool.index),
    "no print in 60d": pool["sue_ter"].isna(),
    "QUIET GOOD (top-ter SUE, muted rx)": (pool["sue_ter"] == 2) & pool["muted"],
    "QUIET BAD (bot-ter SUE, muted rx)": (pool["sue_ter"] == 0) & pool["muted"],
    "LOUD GOOD (top-ter, rx>=5% 2xvol)": (pool["sue_ter"] == 2) & pool["loud_up"],
    "any print, mid SUE": pool["sue_ter"] == 1,
}
print(f"\n{'CELL':<38s}| era      |     n | touch  | vs pool")
for era, em in [("disc<=22", pool["year"] <= 2022), ("conf>=23", pool["year"] >= 2023)]:
    base = pool[em]["touch"].mean()
    for name, m in CELLS.items():
        s = pool[m.fillna(False) & em]
        if len(s) < 80:
            print(f"{name:<38s}| {era} | {len(s):>5,} | too few"); continue
        print(f"{name:<38s}| {era} | {len(s):>5,} | {s['touch'].mean()*100:5.1f}% | "
              f"{(s['touch'].mean()-base)*100:+.1f}pp", flush=True)

# ---- recall accounting + universe-level QGP gate (amendment 2026-09-08) ----
uni["touch"] = uni["hi15"] / uni["close"] - 1 >= 0.05
uni["big"] = uni["hi15"] / uni["close"] - 1 >= 0.10
uni["year"] = uni["trade_date"].dt.year
uni = pd.merge_asof(uni.sort_values("trade_date"), pr, on="trade_date", by="symbol",
                    direction="backward", tolerance=pd.Timedelta(days=60))
pool_keys = set(map(tuple, pool[["symbol", "trade_date"]].astype(str).values))
uni["in_pool"] = [tuple(x) in pool_keys for x in uni[["symbol", "trade_date"]].astype(str).values]

print("\n===== RECALL: where do the +10%/15d movers live? =====")
for era, em in [("disc<=22", uni["year"] <= 2022), ("conf>=23", uni["year"] >= 2023)]:
    movers = uni[em & uni["big"]]
    print(f"[{era}] +10% movers/wk: {len(movers)/max(movers['trade_date'].nunique(),1):.0f} "
          f"of {len(uni[em])/max(uni[em]['trade_date'].nunique(),1):.0f} investable "
          f"| % of movers inside z-band pool: {movers['in_pool'].mean()*100:.1f}%", flush=True)

print("\n===== M1 as INDEPENDENT gate on full universe =====")
UCELLS = {
    "UNIVERSE base": pd.Series(True, index=uni.index),
    "U QUIET GOOD (top-ter, muted)": (uni["sue_ter"] == 2) & uni["muted"],
    "U QUIET GOOD + not in pool": (uni["sue_ter"] == 2) & uni["muted"] & ~uni["in_pool"],
    "U LOUD GOOD (top-ter, rx>=5%)": (uni["sue_ter"] == 2) & uni["loud_up"],
    "U QUIET BAD (bot-ter, muted)": (uni["sue_ter"] == 0) & uni["muted"],
}
print(f"{'CELL':<32s}| era      |      n | touch5 | touch10 | vs base10")
for era, em in [("disc<=22", uni["year"] <= 2022), ("conf>=23", uni["year"] >= 2023)]:
    b10 = uni[em]["big"].mean()
    for name, m in UCELLS.items():
        s2 = uni[m.fillna(False) & em]
        if len(s2) < 80:
            print(f"{name:<32s}| {era} | {len(s2):>6,} | too few"); continue
        print(f"{name:<32s}| {era} | {len(s2):>6,} | {s2['touch'].mean()*100:5.1f}% | "
              f"{s2['big'].mean()*100:6.1f}% | {s2['big'].mean()/b10:4.2f}x", flush=True)
print("\nM1 QUIET GOOD PRINT COMPLETE", flush=True)
