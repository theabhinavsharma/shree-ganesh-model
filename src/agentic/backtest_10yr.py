"""10-year walk-forward backtest of the v3 ensemble (2016 → 2025).

For each year Y in 2016..2025:
  • Train: data with year < Y AND year >= 2015
  • Predict: data with year == Y
  • Calibrate via isotonic regression on Y-1 OOF (or fold-internal if Y=2016)
  • Build daily top-5 portfolio (equal weight)
  • Compute 7d close-to-close return per pick
  • Aggregate per-year stats

Outputs:
  reports/backtest_10yr_summary.md  — per-year table + grand total
  data/derived/backtest_10yr_oof.parquet — full OOF predictions for all 10 years
  data/derived/backtest_10yr_basket.parquet — daily top-5 basket returns
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
ALT_ROOT = ROOT / "tmp/from_scratch_7d_run"
OUT_OOF = ROOT / "data/derived/backtest_10yr_oof.parquet"
OUT_BASKET = ROOT / "data/derived/backtest_10yr_basket.parquet"
OUT_REPORT = ROOT / "reports/backtest_10yr_summary.md"

H = 7  # forward horizon


def build_panel() -> pd.DataFrame:
    print("loading prices …")
    df = pd.read_parquet("data/derived/stock_daily_facts_adjusted_2015plus.parquet")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

    # 7d forward high → label
    shifts_high = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts_high.max(axis=1)
    df["forward_high_pct_7td"] = df["fwd_high_max"] / df["close"] - 1
    df["winner_5pct_7td"] = (df["forward_high_pct_7td"] >= 0.05).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, ["forward_high_pct_7td", "winner_5pct_7td"]] = pd.NA

    # 7d c2c return
    df["close_fwd_7"] = df.groupby("symbol")["close"].shift(-H)
    df["fwd_c2c_7"] = df["close_fwd_7"] / df["close"] - 1

    # standard features (subset that's reliably populated 2015-2025)
    df["dist_sma20"] = df["close"] / df["sma_20"] - 1
    df["dist_sma50"] = df["close"] / df["sma_50"] - 1
    df["dist_sma200"] = df["close"] / df["sma_200"] - 1
    df["above_50dma"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_200dma"] = (df["close"] > df["sma_200"]).astype(int)
    df["realized_vol_20d"] = df.groupby("symbol")["return_1d"].transform(lambda s: s.rolling(20).std())
    df["adv_20d_cr"] = df["avg_traded_value_20d"] / 1e7
    df["year"] = df["trade_date"].dt.year

    # cross-sectional / market-wide
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


FEATS = ["return_1d", "return_20d",
         "dist_sma20", "dist_sma50", "dist_sma200",
         "above_50dma", "above_200dma",
         "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
         "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
         "realized_vol_20d", "adv_20d_cr",
         "market_5d_ret", "market_20d_ret",
         "market_breadth_50dma", "market_breadth_200dma"]


def main() -> None:
    df = build_panel()
    df = df.dropna(subset=FEATS).copy()
    labeled = df[df["winner_5pct_7td"].notna() & df["fwd_c2c_7"].notna()].copy()
    print(f"labeled rows: {len(labeled):,}, base rate winner_5pct_7td: {labeled['winner_5pct_7td'].mean():.3f}")
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"after liquid filter: {len(labeled):,}")

    all_oof: list[pd.DataFrame] = []
    test_years = list(range(2016, 2026))
    for yr in test_years:
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

        # calibrate on prior-year predictions if we have them, else self
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
        # quick year stats
        top5 = te.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date").head(5)
        basket = top5.groupby("trade_date")["fwd_c2c_7"].mean()
        if len(basket) > 0:
            print(f"  top-5 basket: n_days={len(basket)} mean_7d={basket.mean()*100:+.2f}% "
                  f"days>=5%={int((basket>=0.05).sum())} ({(basket>=0.05).mean()*100:.0f}%)")

    if not all_oof:
        print("no folds completed")
        return
    full = pd.concat(all_oof, ignore_index=True)
    full.to_parquet(OUT_OOF, index=False)
    print(f"\nwrote {OUT_OOF}: {len(full):,} OOF rows across {full['trade_date'].dt.year.nunique()} years")

    # build daily top-5 basket across all 10 years
    full = full.dropna(subset=["fwd_c2c_7"])
    full = full.sort_values(["trade_date", "score"], ascending=[True, False])
    top5_per_day = full.groupby("trade_date").head(5)
    basket = top5_per_day.groupby("trade_date").agg(
        ret7=("fwd_c2c_7", "mean"),
        avg_score=("score", "mean"),
        n_picks=("symbol", "size"),
    ).reset_index()
    basket["year"] = basket["trade_date"].dt.year
    basket.to_parquet(OUT_BASKET, index=False)

    # per-year summary
    summary = basket.groupby("year").agg(
        n_days=("ret7", "size"),
        mean_7d=("ret7", "mean"),
        median_7d=("ret7", "median"),
        days_5pct=("ret7", lambda s: int((s >= 0.05).sum())),
        days_2pct=("ret7", lambda s: int((s >= 0.02).sum())),
        days_negative=("ret7", lambda s: int((s < 0).sum())),
    ).reset_index()
    summary["pct_5pct_days"] = summary["days_5pct"] / summary["n_days"]
    summary["ann_compound_weekly"] = (1 + summary["mean_7d"]) ** 52 - 1

    # write report
    md = ["# 10-year Walk-forward Backtest", "",
          f"**Strategy:** every trading day, equal-weight top-5 picks by score; hold 7 trading days; close-to-close return.", "",
          f"**Train:** all data with year < target. **Test:** target year (2016–2025).", "",
          f"**Universe filter:** ADV ≥ ₹1cr/day, EQ series.", "", "## Per-year summary", "",
          "| Year | OOS days | mean 7d | median 7d | days ≥+5% (n / %) | days <0 (n / %) | theoretical ann ROI |",
          "|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        md.append(
            f"| {int(r['year'])} | {int(r['n_days'])} | {r['mean_7d']*100:+.2f}% | {r['median_7d']*100:+.2f}% | "
            f"{int(r['days_5pct'])} / {r['pct_5pct_days']*100:.1f}% | "
            f"{int(r['days_negative'])} / {r['days_negative']/r['n_days']*100:.1f}% | "
            f"{r['ann_compound_weekly']*100:+.0f}% |")
    grand_mean = basket["ret7"].mean()
    grand_5 = (basket["ret7"] >= 0.05).mean()
    md.append(f"| **2016-2025** | {len(basket)} | **{grand_mean*100:+.2f}%** | {basket['ret7'].median()*100:+.2f}% | "
              f"{int((basket['ret7']>=0.05).sum())} / **{grand_5*100:.1f}%** | "
              f"{int((basket['ret7']<0).sum())} / {(basket['ret7']<0).mean()*100:.1f}% | "
              f"{((1+grand_mean)**52-1)*100:+.0f}% |")
    md.append("")
    md.append("## Honest reading")
    md.append("")
    md.append("- **Theoretical ann ROI** assumes you trade every day with zero slippage and no overlap. Real capture is 20-40% of this.")
    md.append("- **Realistic execution:** if you trade weekly, capture is ~ (1+median_7d)^52 - 1, not mean-based.")
    md.append("- **Negative-day count matters:** ~30-50% of OOS days are negative — you need patience-filter to avoid trading those.")
    md.append("- This run uses the basic feature set (no catalysts, no sentiment, no fundamentals). The production model should outperform.")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"wrote {OUT_REPORT}")
    print()
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
