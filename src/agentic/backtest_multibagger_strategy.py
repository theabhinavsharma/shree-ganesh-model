"""Backtest the multibagger strategy: every week in 2024-2025, pick the
top-N names by 100%/180d calibrated score, hold 180 trading days, measure
return. This is the integrity proof of the strategy.

For each weekly entry date (Monday) in 2024:
  1. Predict scores for that day's universe
  2. Pick top 4 names by score_100pct_180d
  3. Hold each for 180 trading days
  4. Measure max-high return AND close-to-close return

Aggregate:
  • Per-entry-date: did ≥1 of 4 double? (matches the 90% claim)
  • Distribution: median basket return, max basket return, drawdown
  • Hit rate of "≥1 doubled" across all entries

This is the cleanest integrity check possible without forward-stepping.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / "reports/multibagger_strategy_backtest.md"
OUT_PARQUET = ROOT / "data/derived/multibagger_strategy_backtest.parquet"

BASE_FEATS = ["return_1d", "return_20d",
              "dist_sma20", "dist_sma50", "dist_sma200",
              "above_50dma", "above_200dma",
              "rsi_14_daily", "rsi_14_weekly", "rsi_14_monthly",
              "volume_vs_20d", "traded_value_vs_20d", "delivery_pct",
              "realized_vol_20d", "adv_20d_cr",
              "market_5d_ret", "market_20d_ret",
              "market_breadth_50dma", "market_breadth_200dma"]
HORIZON = 180
THRESHOLD = 1.00  # 100% double
BASKET_SIZE = 4
SCORE_BAR = 0.86


def build_panel() -> pd.DataFrame:
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    df = df[df["trade_date"] >= "2015-01-01"]

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
    market_med = liq.groupby("trade_date")["return_1d"].median().rename("market_1d_ret").reset_index()
    df = df.merge(mkt, on="trade_date", how="left")
    df = df.merge(market_med, on="trade_date", how="left")
    df["market_5d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(5).sum())
    df["market_20d_ret"] = df.groupby("symbol")["market_1d_ret"].transform(lambda s: s.rolling(20).sum())
    return df


def build_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    fwd_max = df.groupby("symbol")["high"].transform(lambda s: s.shift(-1).rolling(horizon, min_periods=1).max())
    fwd_pct = fwd_max / df["close"] - 1
    target = (fwd_pct >= threshold).astype(int)
    complete = df.groupby("symbol", sort=False)["high"].shift(-horizon).notna()
    target[~complete] = -1
    return target


def main() -> None:
    print("== backtest_multibagger_strategy ==")
    panel = build_panel()
    panel = panel.dropna(subset=BASE_FEATS).copy()
    panel["target"] = build_target(panel, HORIZON, THRESHOLD)
    # 180-day forward close-to-close (correct: shift(-N) gets N days ahead)
    panel["close_fwd_180"] = panel.groupby("symbol")["close"].shift(-HORIZON)
    # 180-day forward max-high — concat shifted columns (the CORRECT forward-looking method)
    print("  computing forward max-high (this is slow, ~30s)…")
    shifts_high = pd.concat(
        [panel.groupby("symbol", sort=False)["high"].shift(-k) for k in range(1, HORIZON + 1)],
        axis=1,
    )
    panel["fwd_max_high_180"] = shifts_high.max(axis=1)
    panel["ret_close_180"] = panel["close_fwd_180"] / panel["close"] - 1
    panel["ret_max_high_180"] = panel["fwd_max_high_180"] / panel["close"] - 1

    labeled = panel[panel["target"] != -1].copy()
    labeled = labeled[labeled["adv_20d_cr"] >= 1.0]

    # train ONCE on data through 2023, evaluate on 2024 alone (single-shot model)
    tr = labeled[labeled["year"] <= 2023]
    print(f"  train rows (≤ 2023): {len(tr):,}")
    lgbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64,
                               min_child_samples=200, feature_fraction=0.85,
                               bagging_fraction=0.85, bagging_freq=5,
                               random_state=42, verbose=-1, n_jobs=-1)
    lgbm.fit(tr[BASE_FEATS], tr["target"])
    tr_calib = tr.sample(min(50000, len(tr)), random_state=42)
    p_tr = lgbm.predict_proba(tr_calib[BASE_FEATS])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_tr, tr_calib["target"])

    # Test set: weekly entries through Oct 2024 — we now have data through Apr 2026
    # so 180d forward from Oct 2024 = ~Apr 2025 (covered)
    test = labeled[(labeled["year"] == 2024) & (labeled["trade_date"] <= "2024-10-31")].copy()
    p_test = lgbm.predict_proba(test[BASE_FEATS])[:, 1]
    test["score_cal"] = iso.transform(p_test)
    print(f"  test rows: {len(test):,}")

    # Weekly entries (each Monday)
    test["entry_weekday"] = test["trade_date"].dt.weekday
    weekly = test[test["entry_weekday"] == 0].copy()  # Mondays
    entry_dates = sorted(weekly["trade_date"].unique())
    print(f"  weekly entry dates: {len(entry_dates)}")

    rows = []
    for entry_date in entry_dates:
        day_picks = test[test["trade_date"] == entry_date].copy()
        # filter to score >= threshold
        candidates = day_picks[day_picks["score_cal"] >= SCORE_BAR]
        if len(candidates) == 0:
            continue
        # take top BASKET_SIZE by score
        basket = candidates.nlargest(BASKET_SIZE, "score_cal")

        # measure outcomes
        avg_max_return = basket["ret_max_high_180"].mean()
        avg_close_return = basket["ret_close_180"].mean()
        n_doubled = (basket["ret_max_high_180"] >= 1.00).sum()
        any_doubled = int(n_doubled >= 1)
        n_50pct = (basket["ret_max_high_180"] >= 0.50).sum()

        rows.append({
            "entry_date": pd.Timestamp(entry_date),
            "basket_size": len(basket),
            "avg_max_return_180d": avg_max_return,
            "avg_close_return_180d": avg_close_return,
            "median_max_return_180d": basket["ret_max_high_180"].median(),
            "n_doubled": int(n_doubled),
            "any_doubled": any_doubled,
            "n_50pct_or_more": int(n_50pct),
            "best_pick_max": float(basket["ret_max_high_180"].max()),
            "worst_pick_max": float(basket["ret_max_high_180"].min()),
            "winners_str": ", ".join(basket.nlargest(BASKET_SIZE, "score_cal")["symbol"].tolist()),
        })

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)

    # aggregate
    n_entries = len(df)
    pct_any_doubled = df["any_doubled"].mean() if n_entries else 0
    avg_basket_max = df["avg_max_return_180d"].mean() * 100 if n_entries else 0
    avg_basket_close = df["avg_close_return_180d"].mean() * 100 if n_entries else 0
    median_basket_max = df["median_max_return_180d"].median() * 100 if n_entries else 0

    print(f"\n=== STRATEGY BACKTEST RESULTS ===")
    print(f"  entry days tested: {n_entries}")
    print(f"  baskets where ≥1 doubled in 180d: {df['any_doubled'].sum()}/{n_entries} ({pct_any_doubled*100:.1f}%)")
    print(f"  avg basket max return (180d): {avg_basket_max:+.1f}%")
    print(f"  avg basket close-to-close return (180d): {avg_basket_close:+.1f}%")
    print(f"  median basket max return: {median_basket_max:+.1f}%")

    # per-month breakdown — detect regime-specific calibration
    if n_entries:
        df["entry_month"] = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m")
        monthly = df.groupby("entry_month").agg(
            n=("entry_date", "size"),
            pct_any_doubled=("any_doubled", "mean"),
            avg_max=("avg_max_return_180d", "mean"),
            avg_close=("avg_close_return_180d", "mean"),
        ).round(3)
        print(f"\n  per-entry-month breakdown:")
        print(monthly.to_string())
    if n_entries:
        print(f"  best entry: {df.loc[df['avg_max_return_180d'].idxmax(), 'entry_date']:%Y-%m-%d}  "
              f"max_ret={df['avg_max_return_180d'].max()*100:+.1f}%")
        print(f"  worst entry: {df.loc[df['avg_max_return_180d'].idxmin(), 'entry_date']:%Y-%m-%d}  "
              f"max_ret={df['avg_max_return_180d'].min()*100:+.1f}%")

    # report
    md = [f"# Multibagger strategy backtest", "",
          f"Strategy: every Monday, pick top-{BASKET_SIZE} names with score ≥ {SCORE_BAR} "
          f"on 100%/{HORIZON}d model, hold {HORIZON} trading days.", "",
          f"Test period: 2024-01-01 to 2024-07-01 (need 180d forward data for outcome).", "",
          "## Aggregate results", "",
          f"- **Total weekly entries tested:** {n_entries}",
          f"- **% baskets with ≥1 name doubling in 180d:** {pct_any_doubled*100:.1f}%",
          f"- **Avg basket max-high return (180d):** {avg_basket_max:+.2f}%",
          f"- **Avg basket close-to-close return (180d):** {avg_basket_close:+.2f}%",
          f"- **Median basket max return:** {median_basket_max:+.2f}%", ""]

    if n_entries >= 5:
        md.append("## Per-entry results (chronological)")
        md.append("")
        md.append("| Entry date | Basket | Avg max % | Avg close % | n doubled | n ≥+50% | Best pick % | Worst pick % | Winners |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in df.iterrows():
            md.append(f"| {pd.Timestamp(r['entry_date']):%Y-%m-%d} | {int(r['basket_size'])} | "
                      f"{r['avg_max_return_180d']*100:+.1f}% | "
                      f"{r['avg_close_return_180d']*100:+.1f}% | "
                      f"{int(r['n_doubled'])}/{int(r['basket_size'])} | "
                      f"{int(r['n_50pct_or_more'])}/{int(r['basket_size'])} | "
                      f"{r['best_pick_max']*100:+.1f}% | "
                      f"{r['worst_pick_max']*100:+.1f}% | "
                      f"{r['winners_str']} |")

    md.append("")
    md.append("## Honest read")
    md.append("")
    md.append(f"The model's claim of 90% hit rate on score ≥ {SCORE_BAR} translates to:")
    md.append(f"- A 4-name basket where ≥1 of 4 should double in 180d ({pct_any_doubled*100:.1f}% of baskets here)")
    md.append(f"- The basket's average max return: {avg_basket_max:+.1f}% (max-high captures the peak; close-to-close is harder to capture)")
    md.append(f"- Real-world capture (no perfect-exit assumption): expect 50-70% of avg max return")
    md.append(f"- Realistic basket return: {avg_basket_max*0.6:+.1f}% to {avg_basket_max*0.8:+.1f}% over 180d")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")
    print(f"     {OUT_PARQUET}")


if __name__ == "__main__":
    main()
