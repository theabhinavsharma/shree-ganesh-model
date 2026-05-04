"""9-year walk-forward backtest WITH macro-enriched feature set.

Mirrors backtest_10yr.py but joins extra_features.parquet (macro_*, sec_*, the
1 KEEP interaction macro_int_regimevix_x_rv20) into the training panel.

Output:
  data/derived/backtest_10yr_macro_oof.parquet — OOS predictions with macro
  reports/backtest_10yr_macro_summary.md

Usage:
  Run after feature_factory.py has produced extra_features.parquet.
  Then run backtest_event_driven.py on the macro OOF to measure CAGR delta.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
EXTRA = ROOT / "data/derived/extra_features.parquet"
OUT_OOF = ROOT / "data/derived/backtest_10yr_macro_oof.parquet"
OUT_REPORT = ROOT / "reports/backtest_10yr_macro_summary.md"

H = 7

# Same base features as backtest_10yr.py
BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]

# Macro features to add — only the ones with proven time-series IC + the 1 KEEP interaction
MACRO_FEATS_KEEP = [
    # Top-IC macro features (non-suspect, |ts_ic| >= 0.10):
    "macro_new_52w_highs",
    "macro_smid_lcap_breadth_diff",
    "eurinr_20d_chg",
    "macro_em_hy_oas",
    "macro_brent_60d_pct",
    "macro_gold_ppi",
    "macro_hy_oas",
    "macro_us_vix_z_60d",
    "macro_spx",
    "macro_us_3m",
    "macro_brent_5d_pct",
    "macro_breadth_50_5d_chg",
    "macro_ig_oas",
    "macro_cross_section_dispersion_20d",
    "macro_median_realized_vol_20d",
    # The 1 KEEP cross-sectional interaction:
    "macro_int_regimevix_x_rv20",
]


def build_panel() -> pd.DataFrame:
    """Same panel as backtest_10yr.py."""
    print("loading prices …")
    df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    shifts_high = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts_high.max(axis=1)
    df["forward_high_pct_7td"] = df["fwd_high_max"] / df["close"] - 1
    df["winner_5pct_7td"] = (df["forward_high_pct_7td"] >= 0.05).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, ["forward_high_pct_7td", "winner_5pct_7td"]] = pd.NA

    df["close_fwd_7"] = df.groupby("symbol")["close"].shift(-H)
    df["fwd_c2c_7"] = df["close_fwd_7"] / df["close"] - 1

    df["dist_sma20"] = df["close"] / df["sma_20"] - 1
    df["dist_sma50"] = df["close"] / df["sma_50"] - 1
    df["dist_sma200"] = df["close"] / df["sma_200"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7
    df["year"] = df["trade_date"].dt.year

    liq = df[df["adv_20d_cr"] >= 1.0]
    mkt = liq.groupby("trade_date").agg(
        market_1d_ret=("return_1d", "median"),
        market_breadth_50dma=("above_50dma", "mean"),
        market_breadth_200dma=("above_200dma", "mean"),
    ).reset_index().sort_values("trade_date")
    mkt["market_5d_ret"] = mkt["market_1d_ret"].rolling(5).sum()
    mkt["market_20d_ret"] = mkt["market_1d_ret"].rolling(20).sum()
    df = df.merge(mkt, on="trade_date", how="left")
    return df


def join_macro(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Join the macro/aggregate features from extra_features.parquet."""
    if not EXTRA.exists():
        print("  WARN: extra_features.parquet missing — running base-only")
        return df, []
    ex = pd.read_parquet(EXTRA)
    ex["trade_date"] = pd.to_datetime(ex["trade_date"])
    keep = [c for c in MACRO_FEATS_KEEP if c in ex.columns]
    print(f"  joining {len(keep)}/{len(MACRO_FEATS_KEEP)} macro features (others missing)")
    if not keep:
        return df, []
    sub = ex[["symbol", "trade_date"] + keep]
    df = df.merge(sub, on=["symbol", "trade_date"], how="left")
    # fill NaNs with median (defensive)
    for c in keep:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)
    return df, keep


def main() -> None:
    df = build_panel()
    df, MACRO_FEATS = join_macro(df)
    FEATS = BASE_FEATS + MACRO_FEATS
    print(f"  total features: {len(FEATS)} ({len(BASE_FEATS)} base + {len(MACRO_FEATS)} macro)")
    df = df.dropna(subset=FEATS).copy()
    labeled = df[df["winner_5pct_7td"].notna() & df["fwd_c2c_7"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"  labeled rows after liquid filter: {len(labeled):,}")

    all_oof = []
    for yr in range(2017, 2026):
        tr = labeled[labeled["year"] < yr]
        te = labeled[labeled["year"] == yr].copy()
        if len(tr) < 5000 or len(te) < 100:
            print(f"  skip {yr}: tr={len(tr)} te={len(te)}")
            continue
        print(f"\n=== fold {yr} (train n={len(tr):,}, test n={len(te):,}) ===")
        lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[FEATS], tr["winner_5pct_7td"].astype(int))
        xgbm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42,
                                  verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        xgbm.fit(tr[FEATS], tr["winner_5pct_7td"].astype(int))
        p_lgb = lgbm.predict_proba(te[FEATS])[:, 1]
        p_xgb = xgbm.predict_proba(te[FEATS])[:, 1]
        te["score"] = 0.5 * p_lgb + 0.5 * p_xgb
        if len(all_oof):
            prev = pd.concat(all_oof, ignore_index=True).tail(200000)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(prev["score"], prev["winner_5pct_7td"].astype(int))
            te["score_cal"] = iso.transform(te["score"])
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(te["score"], te["winner_5pct_7td"].astype(int))
            te["score_cal"] = iso.transform(te["score"])
        all_oof.append(te[["trade_date", "symbol", "score", "score_cal",
                            "winner_5pct_7td", "fwd_c2c_7"]].copy())
        # report top quantiles for QC
        print(f"  score_cal: max={te['score_cal'].max():.3f}  "
              f"#>=0.95={(te['score_cal']>=0.95).sum()}  "
              f"#>=0.85={(te['score_cal']>=0.85).sum()}  "
              f"#>=0.80={(te['score_cal']>=0.80).sum()}  "
              f"#>=0.70={(te['score_cal']>=0.70).sum()}")

    if not all_oof:
        print("no folds completed")
        return
    full = pd.concat(all_oof, ignore_index=True)
    full.to_parquet(OUT_OOF, index=False)
    print(f"\nwrote {OUT_OOF}: {len(full):,} OOF rows")
    # quick top-tail comparison
    n_high = (full['score_cal'] >= 0.95).sum()
    print(f"  full OOF tail: #>=0.95={n_high}  #>=0.85={(full['score_cal']>=0.85).sum()}")


if __name__ == "__main__":
    main()
