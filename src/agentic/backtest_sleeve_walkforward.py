"""SLEEVE WALK-FORWARD A/B — pre-registered 2026-08-29, run BEFORE any sleeve ships.

Arms (weekly grid, investable = ADV>=5cr & close>50):
  HC10 : top-10 by calibrated P(touch +20% in 30td) — hc-replica: LGB on the engine's
         feature set, EXPANDING window retrained quarterly, isotonic on trailing year,
         training labels' 30td windows complete before the scoring quarter starts.
  C6   : fresh-IPO break events (age<=252td, ret1d>=8%, vol>=3x), one per symbol/60td.
  UNION: HC10 ∪ C6 per week.

Exits (day-by-day from next open, 0.30% round-trip cost, stop-first on both-touch):
  HOLD100 : fixed 100td hold.
  TRAIL18 : -18% below running peak-high; cap +100%; timeout 100td.
  TRAIL15 : -15% variant.

Metrics per arm×exit×era (disc<=2022 / conf>=2023): n, net/trade, weekly-cohort mean,
P(2x touch), P(+25% touch), median trough, years positive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src/agentic"))
import find_multibagger_today as mbmod  # reuse the engines' exact panel builder

print("building feature panel (engine-identical)…", flush=True)
panel = mbmod.build_panel()
FEATS = mbmod.BASE_FEATS
panel = panel.dropna(subset=FEATS).reset_index(drop=True)
panel["trade_date"] = pd.to_datetime(panel["trade_date"])

print("labels + event flags…", flush=True)
g = panel.groupby("symbol", sort=False)
def fwd_max(col, n):
    return g[col].transform(lambda s: s.shift(-1)[::-1].rolling(n, min_periods=n).max()[::-1])
panel["fwd_hi30"] = fwd_max("high", 30)
panel["y20_30"] = (panel["fwd_hi30"] / panel["close"] - 1 >= 0.20).astype(int)
panel.loc[panel["fwd_hi30"].isna(), "y20_30"] = -1
panel["age_td"] = g.cumcount()
panel["c6_event"] = (panel["age_td"] <= 252) & (panel["return_1d"] >= 0.08) & (panel["volume_vs_20d"] >= 3)
panel["investable"] = (panel["adv_20d_cr"] >= 5) & (panel["close"] > 50)

days = np.array(sorted(panel["trade_date"].unique()))
weekly = set(days[::5])
sf = {s: gg[["trade_date", "open", "high", "low", "close"]].reset_index(drop=True)
      for s, gg in panel.groupby("symbol")}

def simulate(sym, d, mode):
    gg = sf[sym]
    idx = gg.index[gg["trade_date"] == d]
    if not len(idx): return None
    fut = gg.iloc[idx[0] + 1: idx[0] + 101]
    if len(fut) < 20 or pd.isna(fut.iloc[0]["open"]): return None
    ep = float(fut.iloc[0]["open"])
    if mode == "HOLD100":
        r = (float(fut.iloc[-1]["close"]) / ep - 1) * 100
    else:
        stop_pct = 0.18 if mode == "TRAIL18" else 0.15
        peak = ep; r = None
        for _, row in fut.iterrows():
            if pd.isna(row["low"]): continue
            stop = peak * (1 - stop_pct)
            if row["low"] <= stop:                       # stop-first
                r = (stop / ep - 1) * 100; break
            if row["high"] >= ep * 2.0:                  # +100% cap
                r = 100.0; break
            peak = max(peak, row["high"])
        if r is None:
            r = (float(fut.iloc[-1]["close"]) / ep - 1) * 100
    hi = fut["high"].max(); lo = fut["low"].min()
    return dict(ret=r - 0.30, t2x=hi / ep - 1 >= 1.0, t25=hi / ep - 1 >= 0.25,
                trough=(lo / ep - 1) * 100)

# ---- HC10: quarterly expanding-window training, weekly scoring ----
print("walk-forward HC10 scoring (quarterly retrain)…", flush=True)
labeled = panel[(panel["y20_30"] != -1) & (panel["adv_20d_cr"] >= 1.0)]
quarters = pd.period_range("2016Q1", "2026Q3", freq="Q")
hc_picks = []   # (trade_date, symbol)
for q in quarters:
    q_start, q_end = q.start_time, q.end_time
    cutoff = q_start - pd.Timedelta(days=75)   # 30td label window + buffer fully complete
    tr = labeled[labeled["trade_date"] <= cutoff]
    if len(tr) < 100_000: continue
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
    model.fit(tr[FEATS], tr["y20_30"])
    cal = tr[tr["trade_date"] >= cutoff - pd.Timedelta(days=365)]
    cal = cal.sample(min(50_000, len(cal)), random_state=42)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(model.predict_proba(cal[FEATS])[:, 1], cal["y20_30"])
    sc = panel[(panel["trade_date"] >= q_start) & (panel["trade_date"] <= q_end)
               & panel["trade_date"].isin(weekly) & panel["investable"]]
    if sc.empty: continue
    sc = sc.copy()
    sc["p"] = iso.transform(model.predict_proba(sc[FEATS])[:, 1])
    top = sc.sort_values("p", ascending=False).groupby("trade_date").head(10)
    hc_picks += list(zip(top["trade_date"], top["symbol"]))
    print(f"  {q}: train {len(tr):,} → picks {len(top)}", flush=True)

# ---- C6 events (one per symbol per 60td) ----
c6 = panel[panel["c6_event"] & panel["investable"]][["trade_date", "symbol"]]
keep, last = [], {}
for _, r in c6.sort_values("trade_date").iterrows():
    if r["symbol"] not in last or (r["trade_date"] - last[r["symbol"]]).days > 90:
        keep.append((r["trade_date"], r["symbol"])); last[r["symbol"]] = r["trade_date"]
c6_picks = keep
union = sorted(set(hc_picks) | set(c6_picks))
print(f"picks — HC10 {len(hc_picks):,} · C6 {len(c6_picks):,} · UNION {len(union):,}", flush=True)

# ---- simulate all arms × exits ----
print(f"\n{'ARM×EXIT':<22s}| era      |     n | net/trade | wk-mean | P(2x) | P(+25%) | medTrough | yrs+")
for arm, picks in [("HC10", hc_picks), ("C6", c6_picks), ("UNION", union)]:
    for mode in ["HOLD100", "TRAIL18", "TRAIL15"]:
        rows = []
        for d, s in picks:
            out = simulate(s, d, mode)
            if out: rows.append({**out, "d": d})
        df = pd.DataFrame(rows)
        if df.empty: continue
        df["year"] = df["d"].dt.year; df["week"] = df["d"].dt.to_period("W")
        for era, m in [("disc<=22", df["year"] <= 2022), ("conf>=23", df["year"] >= 2023)]:
            sdf = df[m]
            if len(sdf) < 30: continue
            wk = sdf.groupby("week")["ret"].mean(); yr = sdf.groupby("year")["ret"].mean()
            print(f"{arm+'×'+mode:<22s}| {era} | {len(sdf):>5,} | {sdf['ret'].mean():>+8.2f}% | {wk.mean():>+6.2f}% | {sdf['t2x'].mean()*100:>4.1f}% | {sdf['t25'].mean()*100:>6.1f}% | {sdf['trough'].median():>+8.1f}% | {(yr>0).mean()*100:>3.0f}%", flush=True)
print("SLEEVE A/B COMPLETE", flush=True)
