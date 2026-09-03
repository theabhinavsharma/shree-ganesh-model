"""A/B — market-VOLATILITY exposure gates on the 15d contract. Pre-registered 2026-09-03.

Motivation: selection sees momentum (5d/20d market ret) and breadth, but NO market-vol
input — a vol-regime turn is invisible until it damages the lagged return windows.

Frame (identical to the shipped z-band A/B): weekly grid 2016-06→2026-08, Z-band pool
(rsi_z/r20_z/r5_z + vol<2, ADV>=5cr, >Rs50), top-8/week by delivery ratio, day-by-day
C2 exits from next open (vol-scaled SL, half@+5%, trail, 15td timeout), 0.30% costs.

Gates (all thresholds trailing-distribution based — no fitted parameters):
  G0 BASELINE   : full exposure every week
  G1 VIX-LEVEL  : half exposure when India VIX close > trailing-252d 80th percentile
  G2 VIX-ACCEL  : half exposure when VIX is +15% over its level 5 sessions ago
  G3 DISPERSION : half exposure when 5d mean cross-sectional std of daily returns
                  (investable universe) sits > 1 trailing-z vs its own 252d history
  G4 SKIP-ACCEL : ZERO exposure on G2 weeks (the aggressive variant)

Metrics per era (disc<=2022 / conf>=2023): mean weekly net, compounded CAGR, max
drawdown of the weekly-compounded path, years positive. Ship bar: a gate must beat
baseline CAGR AND maxDD in BOTH eras.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")

print("loading…", flush=True)
px = pd.read_parquet(ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet",
    columns=["symbol", "trade_date", "open", "high", "low", "close", "return_1d", "rsi_14_daily",
             "return_20d", "volume_vs_20d", "delivery_pct", "avg_delivery_pct_20d", "avg_traded_value_20d"])
px["trade_date"] = pd.to_datetime(px["trade_date"])
px = px.sort_values(["symbol", "trade_date"])
g = px.groupby("symbol")
px["ret_5d"] = g["close"].pct_change(5)
px["dvol"] = g["return_1d"].transform(lambda s: s.rolling(20).std())
px["rsi_mu"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).mean())
px["rsi_sd"] = g["rsi_14_daily"].transform(lambda s: s.rolling(252, min_periods=120).std())
px["adv"] = px["avg_traded_value_20d"] / 1e7
px["dlv"] = px["delivery_pct"] / px["avg_delivery_pct_20d"]
px["rsi_z"] = (px["rsi_14_daily"] - px["rsi_mu"]) / px["rsi_sd"]
px["r20_z"] = px["return_20d"] / (px["dvol"] * np.sqrt(20))
px["r5_z"] = px["ret_5d"] / (px["dvol"] * np.sqrt(5))

days = sorted(px["trade_date"].unique())
weekly = sorted(set(days[::5]))
Z = (px["rsi_z"].between(-1.0, 0.25) & px["r20_z"].between(-1.0, 0.75) & px["r5_z"].between(-1.0, 1.0)
     & (px["adv"] >= 5) & (px["close"] > 50) & (px["volume_vs_20d"] < 2)
     & px["trade_date"].isin(weekly) & (px["trade_date"] >= "2016-06-01"))
pool = px[Z]

print("market-vol series…", flush=True)
inv = px[(px["adv"] >= 5) & (px["close"] > 50)]
disp_daily = inv.groupby("trade_date")["return_1d"].std().rename("disp")
disp = disp_daily.rolling(5).mean()
disp_z = (disp - disp.rolling(252).mean()) / disp.rolling(252).std()

vix = pd.read_parquet(ROOT / "data/derived/india_vix.parquet").set_index("date")["vix_close"].sort_index()
vix_pct80 = vix.rolling(252).quantile(0.80)
vix_chg5 = vix / vix.shift(5) - 1

def asof(series, d):
    s = series.loc[:d]
    return float(s.iloc[-1]) if len(s) and pd.notna(s.iloc[-1]) else np.nan

sf = {s: gg.reset_index(drop=True) for s, gg in px.groupby("symbol")}
def c2(sym, d):
    gg = sf[sym]
    sig = gg[gg["trade_date"] == d]; fut = gg[gg["trade_date"] > d].head(15)
    if len(sig) == 0 or len(fut) < 8 or pd.isna(fut.iloc[0]["open"]): return None
    ep = float(fut.iloc[0]["open"]); dv = float(sig.iloc[0]["dvol"]) if pd.notna(sig.iloc[0]["dvol"]) else 0.02
    tgt = ep * 1.05; sl = ep * (1 - np.clip(3 * dv, 0.03, 0.12)); half = False; trail = None
    for i in range(len(fut)):
        r = fut.iloc[i]; lo, hi = r["low"], r["high"]
        if pd.isna(lo): continue
        if not half:
            if lo <= sl: return (sl / ep - 1) * 100
            if pd.notna(hi) and hi >= tgt: half = True; trail = ep * 1.025; continue
        else:
            if lo <= trail: return ((0.05 + (trail / ep - 1)) / 2) * 100
    last = float(fut.iloc[-1]["close"])
    return ((0.05 + (last / ep - 1)) / 2) * 100 if half else (last / ep - 1) * 100

print("weekly baskets + C2 sim…", flush=True)
weeks = []
picks = pool.sort_values(["trade_date", "dlv"], ascending=[True, False]).groupby("trade_date").head(8)
for d, wk in picks.groupby("trade_date"):
    rets = [v for v in (c2(r["symbol"], d) for _, r in wk.iterrows()) if v is not None]
    if len(rets) < 4: continue
    weeks.append(dict(d=d, ret=np.mean(rets) - 0.30,
                      vix=asof(vix, d), vix80=asof(vix_pct80, d),
                      vchg=asof(vix_chg5, d), dz=asof(disp_z, d)))
W = pd.DataFrame(weeks).dropna(subset=["ret"])
W["year"] = W["d"].dt.year
print(f"weeks simulated: {len(W)}", flush=True)

GATES = {
    "G0 baseline (always full)": np.ones(len(W)),
    "G1 half when VIX>80th pct": np.where(W["vix"] > W["vix80"], 0.5, 1.0),
    "G2 half when VIX +15%/5d": np.where(W["vchg"] > 0.15, 0.5, 1.0),
    "G3 half when dispersion z>1": np.where(W["dz"] > 1.0, 0.5, 1.0),
    "G4 SKIP when VIX +15%/5d": np.where(W["vchg"] > 0.15, 0.0, 1.0),
}
print(f"\n{'GATE':<30s}| era      | active% | mean wk | CAGR | maxDD | yrs+")
for name, exp in GATES.items():
    W["_r"] = W["ret"] / 100 * exp
    for era, m in [("disc<=22", W["year"] <= 2022), ("conf>=23", W["year"] >= 2023)]:
        s = W[m]
        path = (1 + s["_r"]).cumprod()
        yrs = (s["d"].max() - s["d"].min()).days / 365.25
        cagr = path.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
        dd = (path / path.cummax() - 1).min()
        yr = s.groupby("year")["_r"].mean()
        gated = (exp[m.values] < 1).mean() * 100
        print(f"{name:<30s}| {era} | {gated:>6.1f}% | {s['_r'].mean()*100:>+6.2f}% | {cagr*100:>+5.1f}% | {dd*100:>+5.1f}% | {(yr>0).mean()*100:>3.0f}%", flush=True)
print("VOL GATE A/B COMPLETE", flush=True)
