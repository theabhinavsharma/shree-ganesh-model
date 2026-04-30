"""
v3 ensemble: v2 features (40) + catalyst features (~17) = ~57 features.

Walk-forward 2024 + 2025 only (catalyst features are populated post-2026-02-28; pre-2026 rows just see zeros, which the model treats as 'no event').
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom/tmp/from_scratch_7d_run")
OUT = ROOT
print("== v3 ensemble (v2 + catalyst features) ==")

# load v2 OOF and reuse for parts; rebuild full feature set from price + catalyst parquets
df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
df = df[df["trade_date"] >= "2018-01-01"]

# label: max intraday-high over next 7 trading days, ratio to today's close (≥5% = winner)
H = 7
shifts = pd.concat([df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)], axis=1)
df["fwd_high_max"] = shifts.max(axis=1)
df["forward_return_7td"] = df["fwd_high_max"] / df["close"] - 1
df["winner_5pct_7td"] = (df["forward_return_7td"] >= 0.05).astype(int)
# require ALL 7 forward bars present (drops tail where label is partial)
df["_label_complete"] = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
df.loc[~df["_label_complete"], ["forward_return_7td", "winner_5pct_7td"]] = pd.NA

# v2 features (subset using parquet's pre-derived cols)
df["dist_sma20"] = df["close"]/df["sma_20"] - 1
df["dist_sma50"] = df["close"]/df["sma_50"] - 1
df["dist_sma200"] = df["close"]/df["sma_200"] - 1
df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7

# sector mapping
sm = pd.read_parquet(ROOT / "alt2" / "sector_index_members.parquet")
SECT_PRIORITY = ["NIFTY IT","NIFTY BANK","NIFTY AUTO","NIFTY METAL","NIFTY PHARMA","NIFTY FMCG",
                 "NIFTY REALTY","NIFTY ENERGY","NIFTY MEDIA","NIFTY PSE","NIFTY PVT BANK",
                 "NIFTY FINANCIAL SERVICES","NIFTY CONSUMER DURABLES","NIFTY OIL & GAS","NIFTY INFRA",
                 "NIFTY 50","NIFTY NEXT 50","NIFTY MIDCAP 100","NIFTY MIDCAP 150",
                 "NIFTY SMALLCAP 100","NIFTY SMALLCAP 250","NIFTY 500","NIFTY MICROCAP 250"]
sm["pri"] = sm["index_name"].map({n:i for i,n in enumerate(SECT_PRIORITY)}).fillna(99)
sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol","index_name"]].rename(columns={"index_name":"sector"})
df = df.merge(sec_map, on="symbol", how="left")
df["sector"] = df["sector"].fillna("OTHER")

sec_d = df.groupby(["trade_date","sector"])["return_1d"].median().reset_index().rename(columns={"return_1d":"sec_ret_1d"})
sec_d = sec_d.sort_values(["sector","trade_date"])
sec_d["sector_5d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(5).sum())
sec_d["sector_20d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(20).sum())
sec_d["sector_60d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(60).sum())
df = df.merge(sec_d[["trade_date","sector","sector_5d_ret","sector_20d_ret","sector_60d_ret"]], on=["trade_date","sector"], how="left")

liq = df[df["adv_20d_cr"] >= 1.0]
mkt = liq.groupby("trade_date").agg(
    market_1d_ret=("return_1d","median"),
    market_breadth_50dma=("above_50dma","mean"),
    market_breadth_200dma=("above_200dma","mean"),
).reset_index().sort_values("trade_date")
mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
df = df.merge(mkt, on="trade_date", how="left")

df["rel_strength_20d"] = df["return_20d"] - df["sector_20d_ret"]

# merge catalyst features
cat = pd.read_parquet("data/derived/catalyst_features.parquet")
cat["trade_date"] = pd.to_datetime(cat["trade_date"])
df = df.merge(cat, on=["symbol", "trade_date"], how="left")
CATALYST_FEATS = [c for c in cat.columns if c not in ("symbol", "trade_date")]
for c in CATALYST_FEATS:
    df[c] = df[c].fillna(0.0)

V2_FEATS = ["return_1d","return_20d",
            "dist_sma20","dist_sma50","dist_sma200",
            "above_50dma","above_200dma",
            "rsi_14_daily","rsi_14_weekly","rsi_14_monthly",
            "volume_vs_20d","traded_value_vs_20d","delivery_pct",
            "realized_vol_20d","adv_20d_cr",
            "sector_5d_ret","sector_20d_ret","sector_60d_ret",
            "market_5d_ret","market_20d_ret",
            "market_breadth_50dma","market_breadth_200dma","rel_strength_20d"]

ALL_FEATS = V2_FEATS + CATALYST_FEATS
# Keep ALL rows that have feature data; only drop label NaN when training.
df = df.dropna(subset=V2_FEATS)
df["year"] = df["trade_date"].dt.year
labeled = df[df["winner_5pct_7td"].notna() & df["forward_return_7td"].notna()].copy()
print(f"rows: {len(df):,}, feats: {len(ALL_FEATS)}")
print(f"  V2 base: {len(V2_FEATS)} + catalyst: {len(CATALYST_FEATS)}")

def topk_p(d, score_col, k):
    g = d.groupby("trade_date").apply(lambda x: x.nlargest(k, score_col), include_groups=False)
    return g["winner_5pct_7td"].mean(), g["forward_return_7td"].mean()

def run_fold(year, feats, label):
    tr = labeled[labeled["year"] < year]
    te = labeled[labeled["year"] == year].copy()
    if len(tr) == 0 or len(te) == 0: return None
    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                              min_child_samples=200, feature_fraction=0.85,
                              bagging_fraction=0.85, bagging_freq=5,
                              random_state=42, verbose=-1, n_jobs=-1)
    lgbm.fit(tr[feats], tr["winner_5pct_7td"])
    xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                              subsample=0.85, colsample_bytree=0.85,
                              random_state=42, verbosity=0, n_jobs=-1, tree_method="hist",
                              eval_metric="logloss")
    xgbm.fit(tr[feats], tr["winner_5pct_7td"])
    p_lgb = lgbm.predict_proba(te[feats])[:,1]
    p_xgb = xgbm.predict_proba(te[feats])[:,1]
    te["score"] = 0.5*p_lgb + 0.5*p_xgb
    rows = []
    for k in [1, 3, 5, 10]:
        p, r = topk_p(te, "score", k)
        rows.append({"label":label, "year":year, "k":k, "precision":p, "mean_ret":r})
    return rows, te

print("\n=== A/B: v2 vs v3 on 2024 + 2025 fold ===")
all_rows = []
v3_oof = []
for label, feats in [("v2", V2_FEATS), ("v3", ALL_FEATS)]:
    print(f"\n>> {label} ({len(feats)} feats)")
    for yr in [2024, 2025]:
        out = run_fold(yr, feats, label)
        if out is None: continue
        rows, te = out
        all_rows.extend(rows)
        for r in rows:
            print(f"  {label} {yr} k={r['k']}: prec={r['precision']:.3f} ret={r['mean_ret']:.4f}")
        if label == "v3":
            v3_oof.append(te[["trade_date","symbol","score","winner_5pct_7td","forward_return_7td"]])

# aggregate
metrics = pd.DataFrame(all_rows)
agg = metrics.groupby(["label","k"]).agg(p=("precision","mean"), r=("mean_ret","mean")).reset_index()
print("\n=== AVG across 2024-2025 ===")
print(agg.to_string(index=False))
metrics.to_csv(OUT/"v3_fold_metrics.csv", index=False)

# OOF + isotonic
v3_oof_df = pd.concat(v3_oof, ignore_index=True)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(v3_oof_df["score"], v3_oof_df["winner_5pct_7td"])
v3_oof_df["score_calibrated"] = iso.transform(v3_oof_df["score"])
v3_oof_df.to_parquet(OUT/"v3_oof.parquet", index=False)

# threshold scan
print("\n=== v3 threshold scan ===")
for t in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    sub = v3_oof_df[v3_oof_df["score"] >= t]
    if len(sub) == 0: continue
    print(f"  t={t:.2f}: n={len(sub):>6} prec={sub['winner_5pct_7td'].mean():.3f} ret={sub['forward_return_7td'].mean():.4f}")

# final fit on all data + live for 2026-04-27
print("\n=== final fit + live prediction ===")
final_lgb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
final_xgb = xgb.XGBClassifier(n_estimators=600, learning_rate=0.04, max_depth=7,
                               subsample=0.85, colsample_bytree=0.85,
                               random_state=42, verbosity=0, n_jobs=-1, tree_method="hist",
                               eval_metric="logloss")
final_lgb.fit(labeled[ALL_FEATS], labeled["winner_5pct_7td"])
final_xgb.fit(labeled[ALL_FEATS], labeled["winner_5pct_7td"])

# live = latest trade_date in the FULL feature frame (may not have a label yet)
latest = df["trade_date"].max()
live = df[df["trade_date"] == latest].copy()
live["score_lgb"] = final_lgb.predict_proba(live[ALL_FEATS])[:,1]
live["score_xgb"] = final_xgb.predict_proba(live[ALL_FEATS])[:,1]
live["score_ens"] = 0.5 * live["score_lgb"] + 0.5 * live["score_xgb"]
live["score_calibrated"] = iso.transform(live["score_ens"])

# sanity filter: liquid
live = live[live["adv_20d_cr"] >= 1.0]
# write FULL universe (every liquid name with a score) + a top-100 view
live.sort_values("score_ens", ascending=False).to_csv(OUT/"v3_live_full.csv", index=False)
top = live.sort_values("score_ens", ascending=False).head(100)
top.to_csv(OUT/"v3_live_top100.csv", index=False)
print(f"live full ({len(live):,}) + top100 written ({latest:%Y-%m-%d})")

summary = {
    "as_of_trade_date": str(latest.date()),
    "feature_count": len(ALL_FEATS),
    "v2_features": len(V2_FEATS),
    "catalyst_features": len(CATALYST_FEATS),
    "comparison": agg.to_dict(orient="records"),
}
(OUT/"v3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
