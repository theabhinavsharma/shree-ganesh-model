"""A/B 10-year walk-forward backtest WITH 5 KEEP factors added.

Tests whether the factors evaluator's KEEP verdicts translate to real
top-5 portfolio lift — not just IC lift in isolation.

Identical methodology to backtest_10yr.py, but training feature set adds:
  - alpha_volume_signed_revert  (IC=+0.023, IR=3.55)
  - amihud_20d                   (IC=-0.031, IR=2.27)
  - rv_60d                       (IC=-0.066, IR=1.82)
  - vol_of_vol_60d               (IC=-0.059, IR=1.69)
  - turnover_skew_20d            (IC=-0.024, IR=1.65)

Output:
  reports/backtest_10yr_with_factors_summary.md
  data/derived/backtest_10yr_with_factors_oof.parquet
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
OUT_OOF = ROOT / "data/derived/backtest_10yr_with_factors_oof.parquet"
OUT_BASKET = ROOT / "data/derived/backtest_10yr_with_factors_basket.parquet"
OUT_REPORT = ROOT / "reports/backtest_10yr_with_factors_summary.md"

H = 7

KEEP_FACTORS = [
    # Original 5 (now demoted to DROP_AB_FAIL — leaving for control comparison)
    "alpha_volume_signed_revert",
    "amihud_20d",
    "rv_60d",
    "vol_of_vol_60d",
    "turnover_skew_20d",
    # New IC_PASSED (Screener fundamentals — 2026-04-29):
    "scr_stock_price_cagr_3_years",
    "scr_stock_price_cagr_5_years",
    "scr_stock_price_cagr_1_year",
    "scr_compounded_profit_growth_5_years",
    "scr_compounded_profit_growth_3_years",
    "scr_roe",
    "scr_roce",
    "scr_price_to_book",
]


def build_panel() -> pd.DataFrame:
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

    # JOIN extra_features
    if EXTRA.exists():
        ex = pd.read_parquet(EXTRA, columns=["symbol", "trade_date"] + KEEP_FACTORS)
        ex["trade_date"] = pd.to_datetime(ex["trade_date"])
        df = df.merge(ex, on=["symbol", "trade_date"], how="left")
        print(f"  joined {len(KEEP_FACTORS)} extra factors")
    else:
        raise SystemExit(f"missing {EXTRA} — run feature_factory.py first")
    return df


BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]
ALL_FEATS = BASE_FEATS + KEEP_FACTORS


def main() -> None:
    df = build_panel()
    # extra_features only goes back ~2023-06; folds before that will have NaN in keep_factors
    # Strategy: drop those features pre-2024 by setting to median (so trees can still split)
    for f in KEEP_FACTORS:
        df[f] = df[f].fillna(df[f].median())

    df = df.dropna(subset=BASE_FEATS).copy()
    labeled = df[df["winner_5pct_7td"].notna() & df["fwd_c2c_7"].notna()].copy()
    print(f"labeled rows: {len(labeled):,}")
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"after liquid filter: {len(labeled):,}")

    all_oof: list[pd.DataFrame] = []
    test_years = list(range(2016, 2026))
    for yr in test_years:
        tr = labeled[labeled["year"] < yr]
        te = labeled[labeled["year"] == yr].copy()
        if len(tr) < 5000 or len(te) < 100:
            continue
        # Skip pre-2024 folds since extra_features data starts ~2023-06
        # The factors will be filled with median, providing no signal — meaningful A/B only for 2024+
        print(f"\n=== fold {yr} (train n={len(tr):,}, test n={len(te):,}) ===")
        lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[ALL_FEATS], tr["winner_5pct_7td"].astype(int))
        xgbm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42,
                                  verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        xgbm.fit(tr[ALL_FEATS], tr["winner_5pct_7td"].astype(int))
        p_lgb = lgbm.predict_proba(te[ALL_FEATS])[:, 1]
        p_xgb = xgbm.predict_proba(te[ALL_FEATS])[:, 1]
        te["score"] = 0.5 * p_lgb + 0.5 * p_xgb
        te["score_cal"] = te["score"]  # skip iso here (not the point of A/B)

        all_oof.append(te[["trade_date", "symbol", "score", "score_cal",
                            "winner_5pct_7td", "fwd_c2c_7"]].copy())
        top5 = te.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date").head(5)
        basket = top5.groupby("trade_date")["fwd_c2c_7"].mean()
        if len(basket) > 0:
            print(f"  top-5 basket: n_days={len(basket)} mean_7d={basket.mean()*100:+.2f}% "
                  f"days>=5%={int((basket>=0.05).sum())} ({(basket>=0.05).mean()*100:.0f}%)")

    if not all_oof:
        return
    full = pd.concat(all_oof, ignore_index=True)
    full.to_parquet(OUT_OOF, index=False)

    full = full.dropna(subset=["fwd_c2c_7"])
    full = full.sort_values(["trade_date", "score"], ascending=[True, False])
    top5_per_day = full.groupby("trade_date").head(5)
    basket = top5_per_day.groupby("trade_date").agg(
        ret7=("fwd_c2c_7", "mean"),
        avg_score=("score", "mean"),
    ).reset_index()
    basket["year"] = basket["trade_date"].dt.year
    basket.to_parquet(OUT_BASKET, index=False)

    summary = basket.groupby("year").agg(
        n_days=("ret7", "size"),
        mean_7d=("ret7", "mean"),
        median_7d=("ret7", "median"),
        days_5pct=("ret7", lambda s: int((s >= 0.05).sum())),
        days_negative=("ret7", lambda s: int((s < 0).sum())),
    ).reset_index()
    summary["pct_5pct_days"] = summary["days_5pct"] / summary["n_days"]

    md = ["# 10-year Backtest WITH 5 KEEP factors", "",
          f"Added factors: {', '.join(KEEP_FACTORS)}", "",
          "**Note:** factor data starts ~2023-06, so pre-2024 folds get median-fill (no real signal).",
          "Compare years 2024 + 2025 against baseline `backtest_10yr_summary.md` for the A/B verdict.", "",
          "## Per-year summary", "",
          "| Year | OOS days | mean 7d | median 7d | days ≥+5% (%) | days <0 (%) |",
          "|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        md.append(f"| {int(r['year'])} | {int(r['n_days'])} | {r['mean_7d']*100:+.2f}% | "
                  f"{r['median_7d']*100:+.2f}% | {r['pct_5pct_days']*100:.1f}% | "
                  f"{r['days_negative']/r['n_days']*100:.1f}% |")
    grand_mean = basket["ret7"].mean()
    md.append(f"| **all** | {len(basket)} | **{grand_mean*100:+.2f}%** | "
              f"{basket['ret7'].median()*100:+.2f}% | "
              f"{(basket['ret7']>=0.05).mean()*100:.1f}% | "
              f"{(basket['ret7']<0).mean()*100:.1f}% |")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
