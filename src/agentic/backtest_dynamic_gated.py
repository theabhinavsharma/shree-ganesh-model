"""Dynamic-gated backtest — what the user actually wanted measured.

Difference from backtest_10yr.py:
  • Every day we check ALL stocks for the 0.95+ calibrated band
  • If 0+ names fire → trade them (top-5 of those)
  • If 0 names fire → SIT IN CASH for 7% annualised
  • 2018, 2019 should look very different now: most days the model didn't
    fire 0.95+, so we mostly sat in cash, not in losing baskets

Walk-forward years 2016-2025. For each year:
  - n_days_with_fire: how often the model fires 0.95+
  - basket_return_when_firing
  - blended ann (fire days at conditional return + cash days at 7%/365)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / "reports/dynamic_gated_backtest.md"
OUT_PARQUET = ROOT / "data/derived/dynamic_gated_backtest.parquet"

H = 7  # 7-day forward
THRESHOLD = 0.05  # +5% within 7 days
GATE = 0.95  # only trade when calibrated score >= 0.95
CASH_ANN = 0.07  # LIQUIDPLUS yield
CASH_DAILY = (1 + CASH_ANN) ** (1/252) - 1  # daily

FEATS = ["return_1d", "return_20d",
         "dist_sma20", "dist_sma50", "dist_sma200",
         "above_50dma", "above_200dma",
         "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
         "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
         "realized_vol_20d", "adv_20d_cr",
         "market_5d_ret", "market_20d_ret",
         "market_breadth_50dma", "market_breadth_200dma"]


def build_panel() -> pd.DataFrame:
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2014-01-01"]

    shifts = pd.concat(
        [df.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, H + 1)],
        axis=1,
    )
    df["fwd_high_max"] = shifts.max(axis=1)
    df["winner"] = (df["fwd_high_max"] / df["close"] - 1 >= THRESHOLD).astype("Int64")
    complete = df.groupby("symbol", sort=False)["high"].shift(-H).notna()
    df.loc[~complete, "winner"] = pd.NA

    # close-to-close 7d (for actual return on hit)
    df["close_fwd"] = df.groupby("symbol")["close"].shift(-H)
    df["fwd_c2c"] = df["close_fwd"] / df["close"] - 1

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
        market_breadth_50dma=("above_50dma", "mean"),
        market_breadth_200dma=("above_200dma", "mean"),
    ).reset_index().sort_values("trade_date")
    df = df.merge(mkt, on="trade_date", how="left")
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())
    return df


def main() -> None:
    print("== backtest_dynamic_gated ==")
    df = build_panel()
    df = df.dropna(subset=FEATS).copy()
    labeled = df[df["winner"].notna() & df["fwd_c2c"].notna()].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]
    print(f"  labeled rows: {len(labeled):,}")

    # Walk-forward: train on years <= yr-2, calibrate on yr-1, test on yr
    rows = []
    for yr in range(2017, 2026):
        tr = labeled[labeled["year"] <= yr - 2]
        cal = labeled[labeled["year"] == yr - 1]
        te = labeled[labeled["year"] == yr].copy()
        if len(tr) < 5000 or len(cal) < 1000 or len(te) < 100:
            continue

        lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                                   min_child_samples=200, feature_fraction=0.85,
                                   bagging_fraction=0.85, bagging_freq=5,
                                   random_state=42, verbose=-1, n_jobs=-1)
        lgbm.fit(tr[FEATS], tr["winner"].astype(int))
        xgbm = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=7,
                                  subsample=0.85, colsample_bytree=0.85, random_state=42,
                                  verbosity=0, n_jobs=-1, tree_method="hist", eval_metric="logloss")
        xgbm.fit(tr[FEATS], tr["winner"].astype(int))

        p_cal = 0.5 * lgbm.predict_proba(cal[FEATS])[:, 1] + 0.5 * xgbm.predict_proba(cal[FEATS])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, cal["winner"].astype(int))

        p_te = 0.5 * lgbm.predict_proba(te[FEATS])[:, 1] + 0.5 * xgbm.predict_proba(te[FEATS])[:, 1]
        te["score_cal"] = iso.transform(p_te)

        # Daily aggregate: how often does ANY name fire at 0.95+?
        daily = te.groupby("trade_date").agg(
            n_total=("symbol", "size"),
            n_at_095=("score_cal", lambda s: (s >= GATE).sum()),
            max_score=("score_cal", "max"),
        ).reset_index()
        n_total_days = len(daily)
        n_fire_days = (daily["n_at_095"] >= 1).sum()

        # When fire happens, take top-5 (by score_cal) and compute basket close-to-close
        fire_te = te[te["score_cal"] >= GATE].copy()
        fire_basket = (fire_te.sort_values(["trade_date", "score_cal"], ascending=[True, False])
                              .groupby("trade_date").head(5))
        basket_per_fire_day = fire_basket.groupby("trade_date")["fwd_c2c"].mean()

        # Compute blended ann ROI
        # Each fire day → realize basket return over next 7 days (no overlap assumption: simplified)
        # Each cash day → CASH_DAILY × 1
        if len(basket_per_fire_day) > 0:
            mean_per_fire = basket_per_fire_day.mean()
            median_per_fire = basket_per_fire_day.median()
            hit_rate = (basket_per_fire_day > 0).mean()
        else:
            mean_per_fire = 0
            median_per_fire = 0
            hit_rate = 0

        # Blended: assume fire-day positions held 7 days (1 turn = 7d) - so n_fire_days * (return per fire) +
        #          cash for the rest
        fire_compounded = (1 + mean_per_fire) ** n_fire_days  # naive
        cash_days = n_total_days - n_fire_days
        cash_compounded = (1 + CASH_DAILY) ** (cash_days * 1)
        blended_year = fire_compounded * cash_compounded - 1
        # this is OVERSTATED because we assumed each fire day independently compounds
        # honest: blend daily with weights
        avg_per_day_fire = mean_per_fire * n_fire_days / n_total_days  # weighted contribution
        avg_per_day_cash = CASH_DAILY * cash_days  # contribution
        # actually compute weighted-average daily return:
        weighted_daily = (n_fire_days * mean_per_fire / 7 + cash_days * CASH_DAILY) / n_total_days
        blended_ann = (1 + weighted_daily) ** 252 - 1

        rows.append({
            "year": yr,
            "n_total_days": n_total_days,
            "n_fire_days": int(n_fire_days),
            "fire_pct_days": float(n_fire_days / n_total_days),
            "mean_basket_when_fires": float(mean_per_fire),
            "median_basket_when_fires": float(median_per_fire),
            "hit_rate_when_fires": float(hit_rate),
            "max_score_year": float(daily["max_score"].max()),
            "blended_ann_roi": float(blended_ann),
        })
        print(f"  {yr}: fire days {n_fire_days}/{n_total_days} ({n_fire_days/n_total_days*100:.0f}%) · "
              f"mean basket on fire {mean_per_fire*100:+.2f}% · "
              f"hit rate {hit_rate*100:.0f}% · "
              f"BLENDED ANN {blended_ann*100:+.0f}%")

    res = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT_PARQUET, index=False)

    # report
    md = ["# Dynamic-Gated Backtest — does sitting in cash on no-fire days fix 2018-19?", "",
          "Method: every day, model scores all stocks. If any name has calibrated score ≥ 0.95, "
          "take top-5 of them (basket). Otherwise → sit in cash @ 7%/yr.", "",
          "Walk-forward: train on years ≤ yr-2, calibrate on yr-1, test on yr (strictly prospective).", "",
          "## Per-year results", "",
          "| Year | Trading days | Fire days (0.95+) | Fire % of days | Basket return when fires | Hit rate | **Blended ann ROI** |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        md.append(f"| {int(r['year'])} | {int(r['n_total_days'])} | "
                  f"{int(r['n_fire_days'])} | {r['fire_pct_days']*100:.0f}% | "
                  f"{r['mean_basket_when_fires']*100:+.2f}% | "
                  f"{r['hit_rate_when_fires']*100:.0f}% | "
                  f"**{r['blended_ann_roi']*100:+.0f}%** |")
    md.append("")
    md.append(f"**Strategy: GATE = {GATE} calibrated; trade only on fire days; cash on others.**")
    md.append("")
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
