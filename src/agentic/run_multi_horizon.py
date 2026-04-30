"""Multi-horizon ensemble: 1d, 7d, 21d models stacked.

Concept: a 7d model alone has 80% top-5 precision. When the 1d model AND the
7d model AND the 21d model all agree on a name, the conditional precision
should jump because uncorrelated horizon biases get filtered out. This is
"signal triangulation" — narrow universe, much higher edge.

Output: tmp/from_scratch_7d_run/multi_horizon_top.csv with score_h1, score_h7,
score_h21, score_consensus (geometric mean), and a `triangulated` boolean
flag for names where all three are above the 70th percentile of their distribution.
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
print("== multi-horizon ensemble (1d / 7d / 21d) ==")

df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
df = df[df["trade_date"] >= "2018-01-01"]

# build labels for each horizon
HORIZONS = [(1, 0.02), (7, 0.05), (21, 0.10)]  # (lookahead_days, +x% threshold)
for H, thr in HORIZONS:
    shifts = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    fwd_max = shifts.max(axis=1)
    df[f"label_h{H}"] = ((fwd_max / df["close"] - 1) >= thr).astype("Int64")
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, f"label_h{H}"] = pd.NA

# v2 features (subset using parquet's pre-derived cols)
df["dist_sma20"] = df["close"] / df["sma_20"] - 1
df["dist_sma50"] = df["close"] / df["sma_50"] - 1
df["dist_sma200"] = df["close"] / df["sma_200"] - 1
df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7

# sector mapping
sm = pd.read_parquet(ROOT / "alt2" / "sector_index_members.parquet")
SECT_PRIORITY = ["NIFTY IT", "NIFTY BANK", "NIFTY AUTO", "NIFTY METAL", "NIFTY PHARMA",
                 "NIFTY FMCG", "NIFTY REALTY", "NIFTY ENERGY", "NIFTY MEDIA", "NIFTY PSE",
                 "NIFTY PVT BANK", "NIFTY FINANCIAL SERVICES", "NIFTY CONSUMER DURABLES",
                 "NIFTY OIL & GAS", "NIFTY INFRA", "NIFTY 50", "NIFTY NEXT 50",
                 "NIFTY MIDCAP 100", "NIFTY MIDCAP 150", "NIFTY SMALLCAP 100",
                 "NIFTY SMALLCAP 250", "NIFTY 500", "NIFTY MICROCAP 250"]
sm["pri"] = sm["index_name"].map({n: i for i, n in enumerate(SECT_PRIORITY)}).fillna(99)
sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol", "index_name"]].rename(
    columns={"index_name": "sector"})
df = df.merge(sec_map, on="symbol", how="left")
df["sector"] = df["sector"].fillna("OTHER")

sec_d = df.groupby(["trade_date", "sector"])["return_1d"].median().reset_index().rename(
    columns={"return_1d": "sec_ret_1d"})
sec_d = sec_d.sort_values(["sector", "trade_date"])
sec_d["sector_5d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(
    lambda s: s.rolling(5).sum())
sec_d["sector_20d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(
    lambda s: s.rolling(20).sum())
sec_d["sector_60d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(
    lambda s: s.rolling(60).sum())
df = df.merge(sec_d[["trade_date", "sector", "sector_5d_ret", "sector_20d_ret", "sector_60d_ret"]],
              on=["trade_date", "sector"], how="left")

liq = df[df["adv_20d_cr"] >= 1.0]
mkt = liq.groupby("trade_date").agg(
    market_1d_ret=("return_1d", "median"),
    market_breadth_50dma=("above_50dma", "mean"),
    market_breadth_200dma=("above_200dma", "mean"),
).reset_index().sort_values("trade_date")
mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
df = df.merge(mkt, on="trade_date", how="left")
df["rel_strength_20d"] = df["return_20d"] - df["sector_20d_ret"]

# catalyst features (already look-ahead-safe after the Three-Sins fix)
cat = pd.read_parquet("data/derived/catalyst_features.parquet")
cat["trade_date"] = pd.to_datetime(cat["trade_date"])
df = df.merge(cat, on=["symbol", "trade_date"], how="left")
CATALYST_FEATS = [c for c in cat.columns if c not in ("symbol", "trade_date")]
for c in CATALYST_FEATS:
    df[c] = df[c].fillna(0.0)

V2_FEATS = ["return_1d", "return_20d",
            "dist_sma20", "dist_sma50", "dist_sma200",
            "above_50dma", "above_200dma",
            "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
            "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
            "realized_vol_20d", "adv_20d_cr",
            "sector_5d_ret", "sector_20d_ret", "sector_60d_ret",
            "market_5d_ret", "market_20d_ret",
            "market_breadth_50dma", "market_breadth_200dma", "rel_strength_20d"]
ALL_FEATS = V2_FEATS + CATALYST_FEATS

df = df.dropna(subset=V2_FEATS).copy()
df["year"] = df["trade_date"].dt.year
print(f"rows: {len(df):,}, feats: {len(ALL_FEATS)}")


def fit_horizon(label_col, train_yrs, oof_yrs):
    train = df[df["year"].isin(train_yrs) & df[label_col].notna()].copy()
    oof = df[df["year"].isin(oof_yrs) & df[label_col].notna()].copy()
    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                              min_child_samples=200, feature_fraction=0.85,
                              bagging_fraction=0.85, bagging_freq=5,
                              random_state=42, verbose=-1, n_jobs=-1)
    xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                             subsample=0.85, colsample_bytree=0.85, random_state=42,
                             verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    y = train[label_col].astype(int)
    lgbm.fit(train[ALL_FEATS], y)
    xgbm.fit(train[ALL_FEATS], y)
    p_oof = 0.5 * lgbm.predict_proba(oof[ALL_FEATS])[:, 1] + 0.5 * xgbm.predict_proba(oof[ALL_FEATS])[:, 1]
    return lgbm, xgbm, oof.assign(score=p_oof)


per_horizon = {}
for H, _ in HORIZONS:
    print(f"\n>> horizon {H}d")
    lgbm, xgbm, oof = fit_horizon(f"label_h{H}", train_yrs=[2020, 2021, 2022, 2023], oof_yrs=[2024, 2025])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["score"], oof[f"label_h{H}"].astype(int))
    oof[f"score_h{H}_cal"] = iso.transform(oof["score"])

    # report top-5 precision per day
    top5 = oof.groupby("trade_date").apply(lambda x: x.nlargest(5, "score"), include_groups=False)
    p = top5[f"label_h{H}"].astype(int).mean()
    print(f"   top-5 precision (oof): {p:.3f}")

    # final fit on all-yrs
    full = df[df[f"label_h{H}"].notna()].copy()
    lgbm.fit(full[ALL_FEATS], full[f"label_h{H}"].astype(int))
    xgbm.fit(full[ALL_FEATS], full[f"label_h{H}"].astype(int))
    per_horizon[H] = (lgbm, xgbm, iso, oof)

# live scoring on latest date
latest = df["trade_date"].max()
live = df[df["trade_date"] == latest].copy()
live = live[live["adv_20d_cr"] >= 1.0]
print(f"\nlive scoring on {latest:%Y-%m-%d} ({len(live)} symbols)")

for H, (lgbm, xgbm, iso, _) in per_horizon.items():
    p = 0.5 * lgbm.predict_proba(live[ALL_FEATS])[:, 1] + 0.5 * xgbm.predict_proba(live[ALL_FEATS])[:, 1]
    live[f"score_h{H}"] = p
    live[f"score_h{H}_cal"] = iso.transform(p)

# consensus: geometric mean of the three calibrated scores
live["consensus"] = (live["score_h1_cal"] * live["score_h7_cal"] * live["score_h21_cal"]) ** (1 / 3)

# triangulated: all three above 75th percentile of their oof distribution
thresh_h1 = per_horizon[1][3]["score_h1_cal"].quantile(0.75)
thresh_h7 = per_horizon[7][3]["score_h7_cal"].quantile(0.75)
thresh_h21 = per_horizon[21][3]["score_h21_cal"].quantile(0.75)
live["triangulated"] = ((live["score_h1_cal"] >= thresh_h1) &
                        (live["score_h7_cal"] >= thresh_h7) &
                        (live["score_h21_cal"] >= thresh_h21))
print(f"  triangulated names today: {live['triangulated'].sum()}")

cols = ["symbol", "sector", "close", "score_h1", "score_h7", "score_h21",
        "score_h1_cal", "score_h7_cal", "score_h21_cal", "consensus", "triangulated"]
# write FULL universe + top-50 view
live.sort_values("consensus", ascending=False)[cols].to_csv(ROOT / "multi_horizon_full.csv", index=False)
top = live.sort_values("consensus", ascending=False).head(50)
top[cols].to_csv(ROOT / "multi_horizon_top.csv", index=False)
print(f"multi-horizon full ({len(live):,}) + top50 written")
print(f"\nwrote {ROOT/'multi_horizon_top.csv'}")
print("\ntop 15 by consensus:")
print(top[cols].head(15).to_string(index=False))
