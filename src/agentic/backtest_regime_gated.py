"""Test if a regime gate improves the multibagger strategy. Compare:
  ALL-IN: deploy basket every Monday regardless of regime
  GATED: deploy only when regime matches Feb/Sep success pattern

Regime gate (derived from regime analysis):
  market_20d <= -0.02 (recent dip)  OR  breadth_50_5d_chg <= -0.03 (breadth narrowing)
  AND breadth_50 between 0.55 and 0.75 (not too narrow, not too broad)

Output: comparison table — does the gate improve risk-adjusted returns?
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
BACKTEST = ROOT / "data/derived/multibagger_strategy_backtest.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / "reports/regime_gated_backtest.md"


def main() -> None:
    if not BACKTEST.exists():
        print(f"missing {BACKTEST}")
        return
    bt = pd.read_parquet(BACKTEST)
    bt["entry_date"] = pd.to_datetime(bt["entry_date"])

    # rebuild regime signals
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "sma_50",
                                            "return_1d", "avg_traded_value_20d", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[(px["series"] == "EQ") & (px["avg_traded_value_20d"] / 1e7 >= 1.0)]
    px["above_50"] = (px["close"] > px["sma_50"]).astype(int)
    daily = px.groupby("trade_date").agg(
        breadth_50=("above_50", "mean"),
        market_med=("return_1d", "median"),
    ).reset_index()
    daily["market_5d"] = daily["market_med"].rolling(5).sum()
    daily["market_20d"] = daily["market_med"].rolling(20).sum()
    daily["breadth_50_5d_chg"] = daily["breadth_50"].diff(5)

    bt = bt.merge(daily, left_on="entry_date", right_on="trade_date", how="left").dropna(subset=["breadth_50"])

    # define regime gate (multiple variants to test)
    bt["gate_v1"] = (
        (bt["market_20d"] <= -0.02) &
        (bt["breadth_50"].between(0.50, 0.75))
    )
    bt["gate_v2"] = (
        (bt["breadth_50_5d_chg"] <= -0.03) &
        (bt["market_5d"] <= 0)
    )
    bt["gate_v3"] = (
        (bt["market_20d"] <= 0) &
        (bt["breadth_50_5d_chg"] <= -0.02)
    )

    bt["success"] = (bt["any_doubled"] == 1).astype(int)

    print("=== Regime gate comparison (44 weekly entries in 2024) ===\n")
    rows = []
    rows.append({
        "strategy": "ALL-IN (no gate)",
        "n_deploys": len(bt),
        "of_total": len(bt),
        "success_rate": float(bt["success"].mean()),
        "avg_max": float(bt["avg_max_return_180d"].mean()),
        "avg_close": float(bt["avg_close_return_180d"].mean()),
    })
    for v in ["gate_v1", "gate_v2", "gate_v3"]:
        sub = bt[bt[v]]
        if len(sub) == 0:
            continue
        rows.append({
            "strategy": f"GATED ({v})",
            "n_deploys": len(sub),
            "of_total": len(bt),
            "success_rate": float(sub["success"].mean()),
            "avg_max": float(sub["avg_max_return_180d"].mean()),
            "avg_close": float(sub["avg_close_return_180d"].mean()),
        })
    res = pd.DataFrame(rows)
    print(res.round(3).to_string(index=False))

    # also show what % of the year you'd be deployed under each gate
    print("\nDeployment ratio (deploys / total weeks):")
    for v in ["gate_v1", "gate_v2", "gate_v3"]:
        ratio = bt[v].mean()
        print(f"  {v}: {ratio*100:.0f}% of weeks")

    # compute realistic ann ROI assuming non-overlapping baskets only
    # if you deploy 50% of weeks, you get ~52*0.5 = 26 deployment opportunities
    # but baskets last 180d (~26 weeks), so you can only have 1-2 active at a time
    # → realistic: 2-3 baskets per year
    md = ["# Regime-gated multibagger strategy backtest", "",
          "## Comparison: gated vs all-in", "",
          "| Strategy | n deploys | Coverage | Success rate | Avg max % | Avg close % |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in res.iterrows():
        md.append(f"| {r['strategy']} | {int(r['n_deploys'])}/{int(r['of_total'])} | "
                  f"{r['n_deploys']/r['of_total']*100:.0f}% | "
                  f"{r['success_rate']*100:.0f}% | "
                  f"{r['avg_max']*100:+.1f}% | "
                  f"{r['avg_close']*100:+.1f}% |")
    md.append("")
    md.append("## Honest interpretation")
    md.append("")
    md.append("If the GATED rows show meaningfully higher success rate and avg returns, the regime filter improves the strategy.")
    md.append("If they don't, the model has no clean regime gate and the all-in baseline is the honest expectation.")
    md.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
