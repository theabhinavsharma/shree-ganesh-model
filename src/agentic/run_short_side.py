"""Short-side model: predict P(stock falls >= 5% in any of next 7 trading days,
measured as min_intraday_low / close - 1 <= -0.05).

Architecture mirrors the long model (LGB + XGB ensemble + isotonic calibration).
Use: pair this with the long model. Trade only when long+short are both confident
in OPPOSITE names → long-short pairs (sector-neutral) or, for unhedged shorts,
load up via stock futures.

Output:
  tmp/from_scratch_7d_run/short_oof.parquet      OOF predictions
  tmp/from_scratch_7d_run/short_live_top100.csv  today's short candidates
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom/tmp/from_scratch_7d_run")
print("== short-side model: P(low_7td <= -5%) ==")

df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
df["trade_date"] = pd.to_datetime(df["trade_date"])
df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
df = df[df["trade_date"] >= "2018-01-01"]

H = 7
shifts_low = pd.concat(
    [df.groupby("symbol", sort=False)["low"].shift(-k) for k in range(1, H + 1)],
    axis=1,
)
df["fwd_low_min"] = shifts_low.min(axis=1)
df["forward_drawdown_7td"] = df["fwd_low_min"] / df["close"] - 1
df["loser_5pct_7td"] = (df["forward_drawdown_7td"] <= -0.05).astype(int)
complete = df.groupby("symbol", sort=False)["low"].shift(-H).notna()
df.loc[~complete, ["forward_drawdown_7td", "loser_5pct_7td"]] = pd.NA

# rebuild same feature set as v3
df["dist_sma20"] = df["close"] / df["sma_20"] - 1
df["dist_sma50"] = df["close"] / df["sma_50"] - 1
df["dist_sma200"] = df["close"] / df["sma_200"] - 1
df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7

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
sec_d["sector_5d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(5).sum())
sec_d["sector_20d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(20).sum())
sec_d["sector_60d_ret"] = sec_d.groupby("sector")["sec_ret_1d"].transform(lambda s: s.rolling(60).sum())
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
labeled = df[df["loser_5pct_7td"].notna() & df["forward_drawdown_7td"].notna()].copy()
print(f"rows: {len(df):,}, labeled: {len(labeled):,}, base rate of -5% loser: {labeled['loser_5pct_7td'].mean():.3f}")


def topk_p(d, score_col, k):
    g = d.groupby("trade_date").apply(lambda x: x.nlargest(k, score_col), include_groups=False)
    return g["loser_5pct_7td"].mean(), g["forward_drawdown_7td"].mean()


print("\n=== short-side fold (2024 + 2025) ===")
oof_rows = []
for yr in [2024, 2025]:
    tr = labeled[labeled["year"] < yr]
    te = labeled[labeled["year"] == yr].copy()
    if len(tr) == 0 or len(te) == 0:
        continue
    lgbm = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=64,
                              min_child_samples=200, feature_fraction=0.85,
                              bagging_fraction=0.85, bagging_freq=5,
                              random_state=42, verbose=-1, n_jobs=-1)
    lgbm.fit(tr[ALL_FEATS], tr["loser_5pct_7td"].astype(int))
    xgbm = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=7,
                             subsample=0.85, colsample_bytree=0.85, random_state=42,
                             verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
    xgbm.fit(tr[ALL_FEATS], tr["loser_5pct_7td"].astype(int))
    p_lgb = lgbm.predict_proba(te[ALL_FEATS])[:, 1]
    p_xgb = xgbm.predict_proba(te[ALL_FEATS])[:, 1]
    te["score"] = 0.5 * p_lgb + 0.5 * p_xgb
    oof_rows.append(te[["trade_date", "symbol", "score", "loser_5pct_7td", "forward_drawdown_7td"]])
    for k in [5, 10, 20]:
        p, r = topk_p(te, "score", k)
        print(f"   {yr} short top-{k}: hit={p:.3f}  mean_drawdown={r:.4f}")

oof = pd.concat(oof_rows, ignore_index=True)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof["score"], oof["loser_5pct_7td"].astype(int))
oof["score_calibrated"] = iso.transform(oof["score"])
oof.to_parquet(ROOT / "short_oof.parquet", index=False)

# threshold scan
print("\nshort threshold scan:")
for t in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    sub = oof[oof["score"] >= t]
    if len(sub) == 0:
        continue
    print(f"   t={t:.2f}: n={len(sub):>6} hit={sub['loser_5pct_7td'].mean():.3f} "
          f"mean_dd={sub['forward_drawdown_7td'].mean():.4f}")

# final + live
print("\n=== final fit + live short candidates ===")
final_lgb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
final_xgb = xgb.XGBClassifier(n_estimators=600, learning_rate=0.04, max_depth=7,
                              subsample=0.85, colsample_bytree=0.85, random_state=42,
                              verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
final_lgb.fit(labeled[ALL_FEATS], labeled["loser_5pct_7td"].astype(int))
final_xgb.fit(labeled[ALL_FEATS], labeled["loser_5pct_7td"].astype(int))

latest = df["trade_date"].max()
live = df[df["trade_date"] == latest].copy()
live["score_lgb"] = final_lgb.predict_proba(live[ALL_FEATS])[:, 1]
live["score_xgb"] = final_xgb.predict_proba(live[ALL_FEATS])[:, 1]
live["score_ens"] = 0.5 * live["score_lgb"] + 0.5 * live["score_xgb"]
live["score_calibrated"] = iso.transform(live["score_ens"])
live = live[live["adv_20d_cr"] >= 1.0]
# write FULL universe + top-100 view
live.sort_values("score_ens", ascending=False).to_csv(ROOT / "short_live_full.csv", index=False)
short_top = live.sort_values("score_ens", ascending=False).head(100)
short_top.to_csv(ROOT / "short_live_top100.csv", index=False)
print(f"short live full ({len(live):,}) + top100 written ({latest:%Y-%m-%d})")
print(f"max calibrated short prob today: {short_top['score_calibrated'].max():.3f}")
