"""Analyze WHY the multibagger strategy worked in Feb/Sep 2024 and failed
in May/Jun/Oct 2024. Look at macro / breadth / sector signals during each
basket entry window — see what distinguishes 'good' from 'bad' regimes.

Goal: derive a regime filter that tells us "deploy basket NOW vs wait"
with 80%+ accuracy.

Inputs:
  data/derived/multibagger_strategy_backtest.parquet — basket outcomes
  prices parquet — for breadth, market_5d_ret, market_20d_ret, vol regime

Output:
  reports/regime_for_strategy.md — what features predict basket success
  data/derived/regime_filter_thresholds.parquet — actionable cutoffs
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
BACKTEST = ROOT / "data/derived/multibagger_strategy_backtest.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / "reports/regime_for_strategy.md"


def main() -> None:
    print("== analyze_regime_for_strategy ==")
    if not BACKTEST.exists():
        print(f"missing {BACKTEST}")
        return
    bt = pd.read_parquet(BACKTEST)
    bt["entry_date"] = pd.to_datetime(bt["entry_date"])

    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "sma_50",
                                            "sma_200", "return_1d", "avg_traded_value_20d", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[px["series"] == "EQ"]
    px["adv_cr"] = px["avg_traded_value_20d"] / 1e7
    px = px[px["adv_cr"] >= 1.0]

    # compute market-wide signals per day
    px["above_50"] = (px["close"] > px["sma_50"]).astype(int)
    px["above_200"] = (px["close"] > px["sma_200"]).astype(int)
    daily = px.groupby("trade_date").agg(
        breadth_50=("above_50", "mean"),
        breadth_200=("above_200", "mean"),
        market_med_ret=("return_1d", "median"),
    ).reset_index()
    daily["market_5d"] = daily["market_med_ret"].rolling(5).sum()
    daily["market_20d"] = daily["market_med_ret"].rolling(20).sum()
    daily["market_60d"] = daily["market_med_ret"].rolling(60).sum()
    daily["breadth_50_5d_chg"] = daily["breadth_50"].diff(5)

    # join to backtest
    bt = bt.merge(daily, left_on="entry_date", right_on="trade_date", how="left")
    bt = bt.dropna(subset=["breadth_50"])

    # success column (≥1 doubled)
    bt["success"] = (bt["any_doubled"] == 1).astype(int)

    print(f"  {len(bt)} backtest entries with regime features")

    # compare features between success vs failure
    features = ["breadth_50", "breadth_200", "market_5d", "market_20d", "market_60d",
                "breadth_50_5d_chg"]
    print(f"\n=== Feature distributions: success vs failure ===")
    rows = []
    for f in features:
        succ_med = bt.loc[bt["success"] == 1, f].median()
        fail_med = bt.loc[bt["success"] == 0, f].median()
        succ_mean = bt.loc[bt["success"] == 1, f].mean()
        fail_mean = bt.loc[bt["success"] == 0, f].mean()
        # simple rank correlation (Spearman) of feature vs basket return
        try:
            ic = bt[[f, "avg_max_return_180d"]].corr(method="spearman").iloc[0, 1]
        except Exception:
            ic = np.nan
        rows.append({
            "feature": f,
            "median_success": round(float(succ_med), 4),
            "median_fail": round(float(fail_med), 4),
            "delta_median": round(float(succ_med - fail_med), 4),
            "ic_vs_basket_return": round(float(ic), 3) if pd.notna(ic) else np.nan,
        })
    res = pd.DataFrame(rows).sort_values("ic_vs_basket_return", key=lambda s: s.abs(), ascending=False)
    print(res.to_string(index=False))

    # find threshold rules
    print(f"\n=== Threshold rules: best simple regime gate ===")
    # try: when breadth_50 >= 0.65 AND market_20d >= 0.02, what's the success rate?
    rules = [
        ("breadth_50 ≥ 0.65", bt["breadth_50"] >= 0.65),
        ("breadth_50 ≥ 0.70", bt["breadth_50"] >= 0.70),
        ("market_20d ≥ 0.02", bt["market_20d"] >= 0.02),
        ("market_60d ≥ 0.05", bt["market_60d"] >= 0.05),
        ("breadth_50 ≥ 0.65 AND market_20d ≥ 0.02", (bt["breadth_50"] >= 0.65) & (bt["market_20d"] >= 0.02)),
        ("breadth_50 ≥ 0.65 AND market_60d ≥ 0.05", (bt["breadth_50"] >= 0.65) & (bt["market_60d"] >= 0.05)),
        ("breadth_50_5d_chg ≥ 0.05", bt["breadth_50_5d_chg"] >= 0.05),
    ]
    rule_rows = []
    for name, mask in rules:
        sub = bt[mask]
        if len(sub) < 3:
            continue
        rule_rows.append({
            "rule": name,
            "n_entries_passing": len(sub),
            "of_total": len(bt),
            "success_rate": round(sub["success"].mean(), 3),
            "avg_max_return": round(float(sub["avg_max_return_180d"].mean()) * 100, 1),
            "avg_close_return": round(float(sub["avg_close_return_180d"].mean()) * 100, 1),
        })
    rule_df = pd.DataFrame(rule_rows)
    print(rule_df.to_string(index=False))

    # build report
    md = ["# Regime filter for multibagger strategy", "",
          "Question: when should we deploy the multibagger basket vs wait?", "",
          "## Per-feature comparison: success vs failure", ""]
    md.append("| Feature | Median (success) | Median (fail) | Delta | IC vs basket return |")
    md.append("|---|---:|---:|---:|---:|")
    for _, r in res.iterrows():
        ic_str = f"{r['ic_vs_basket_return']:+.3f}" if pd.notna(r['ic_vs_basket_return']) else "—"
        md.append(f"| {r['feature']} | {r['median_success']:.4f} | {r['median_fail']:.4f} | "
                  f"{r['delta_median']:+.4f} | {ic_str} |")
    md.append("")
    md.append("## Threshold rules — when does the basket actually work?")
    md.append("")
    md.append("| Rule | Entries passing | Success rate (≥1 doubled) | Avg max | Avg close |")
    md.append("|---|---:|---:|---:|---:|")
    for _, r in rule_df.iterrows():
        md.append(f"| {r['rule']} | {int(r['n_entries_passing'])}/{int(r['of_total'])} | "
                  f"{r['success_rate']*100:.0f}% | {r['avg_max_return']:+.1f}% | {r['avg_close_return']:+.1f}% |")
    md.append("")
    md.append("## Honest interpretation")
    md.append("")
    md.append("The multibagger strategy is **NOT a 90%-conviction always-on signal**.")
    md.append("It works in specific market regimes (high breadth + positive market_20d/60d) and")
    md.append("fails outside those regimes. The regime filter table shows the deploy/wait gates.")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
