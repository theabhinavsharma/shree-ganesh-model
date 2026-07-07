"""Today's actionable event-driven picks under the new gate config.

Reads:
  data/derived/high_conviction_predictions.parquet — today's calibrated scores
Applies:
  GATE = 0.65, MAX_POS = 10, HOLD = 7 trading days
  TARGET = +5%, SL = -3%
Outputs:
  reports/event_driven_today_<DATE>.md — picks + entry/SL/target/exit-by
  data/derived/event_driven_today.parquet — same structured

Backed by:
  reports/event_driven_backtest_backtest_10yr_macro_oof.md — Top-10 @ 0.65
  produced backtested CAGR +144%, Sharpe 6.17, MaxDD -21.6% over 9-yr OOS.
"""
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
HC = ROOT / "data/derived/high_conviction_predictions.parquet"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT_REPORT = ROOT / f"reports/event_driven_today_{date.today():%Y%m%d}.md"
OUT_PARQUET = ROOT / "data/derived/event_driven_today.parquet"

GATE = 0.65
MAX_POS = 10
HOLD_DAYS = 7
TARGET_PCT = 0.05
SL_PCT = -0.03
MIN_ADV_CR = 1.0


def main() -> None:
    if not HC.exists():
        print(f"missing {HC}")
        return
    pred = pd.read_parquet(HC)
    pred["best"] = pred[["score_5pct_7d_cal", "score_10pct_15d_cal", "score_20pct_30d_cal"]].max(axis=1)

    # liquidity filter
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close",
                                            "avg_traded_value_20d", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[px["series"] == "EQ"]
    px["adv_20d_cr"] = px["avg_traded_value_20d"] / 1e7
    last = px["trade_date"].max()
    today_px = px[px["trade_date"] == last]

    pred = pred.merge(today_px[["symbol", "adv_20d_cr"]], on="symbol", how="left")
    qualified = pred[(pred["best"] >= GATE) & (pred["adv_20d_cr"] >= MIN_ADV_CR)].copy()
    qualified = qualified.sort_values("best", ascending=False).head(MAX_POS)

    # date math: exit-by = entry_date + HOLD_DAYS trading days
    today_dt = pd.Timestamp(date.today())
    # approximate exit date as +10 calendar days for 7 trading
    exit_by = today_dt + pd.tseries.offsets.BDay(HOLD_DAYS)

    # build action table
    qualified["entry_price"] = qualified["close"]
    qualified["target_price"] = qualified["close"] * (1 + TARGET_PCT)
    qualified["stop_loss"] = qualified["close"] * (1 + SL_PCT)
    qualified["entry_date"] = today_dt.strftime("%Y-%m-%d")
    qualified["exit_by_date"] = exit_by.strftime("%Y-%m-%d")
    qualified["alloc_pct_of_capital"] = round(100.0 / MAX_POS, 1)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    qualified.to_parquet(OUT_PARQUET, index=False)

    # markdown report
    md = [f"# Event-driven picks — {today_dt:%Y-%m-%d}", ""]
    md.append(f"**Strategy:** Top-{MAX_POS} basket @ score_cal ≥ {GATE} on any of (5%/7d, 10%/15d, 20%/30d).")
    md.append(f"")
    md.append(f"**Backtest evidence (9-yr walk-forward OOS, BASELINE model — no macro):**")
    md.append(f"- CAGR +154.8% · Sharpe 6.18 · Max drawdown -20.1%")
    md.append(f"- Hit rate 56% per trade · ~575 trades/yr · ~2 trades/trading-day with MAX_POS=10")
    md.append(f"- Source: `reports/event_driven_backtest_backtest_10yr_oof.md`")
    md.append(f"- IMPORTANT: macro features tested in F round did NOT help on average. Baseline model is honest deployment.")
    md.append(f"")
    md.append(f"**Caveats:** Sweep bias (top of 20 configs); Sharpe 6.17 likely overstated by 30-40% after slippage; "
              f"performance concentrated in 2018 + 2024 folds; backtest is OOS-walk-forward but config was selected from sweep — "
              f"forward paper-trade for genuine validation before scaling capital.")
    md.append(f"")
    md.append(f"**Sizing:** ₹{1.0/MAX_POS*100:.0f}% per name. SL: -3%. Target: +5%. Hold: ≤{HOLD_DAYS} trading days.")
    md.append(f"")
    md.append(f"## Today's qualified picks ({len(qualified)} / {MAX_POS} slots)")
    md.append(f"")
    if len(qualified) == 0:
        md.append(f"🔴 **No names cross {GATE} bar today.** All capital → LIQUIDPLUS / CASHIETF (~7% ann).")
        md.append(f"")
        md.append(f"Top-10 by score (still BELOW {GATE} bar):")
        all_top = pred.sort_values("best", ascending=False).head(10)
        md.append(f"")
        md.append(f"| # | Symbol | Close | best score | 5%/7d | 10%/15d | 20%/30d |")
        md.append(f"|---|---|---:|---:|---:|---:|---:|")
        for i, (_, r) in enumerate(all_top.iterrows(), 1):
            md.append(f"| {i} | **{r['symbol']}** | ₹{r['close']:.2f} | "
                      f"{r['best']:.3f} | {r['score_5pct_7d_cal']:.3f} | "
                      f"{r['score_10pct_15d_cal']:.3f} | {r['score_20pct_30d_cal']:.3f} |")
    else:
        md.append(f"| # | Symbol | Close | Score (best) | Entry | Target +5% | SL -3% | Exit by | Alloc |")
        md.append(f"|---|---|---:|---:|---:|---:|---:|---|---:|")
        for i, (_, r) in enumerate(qualified.iterrows(), 1):
            md.append(f"| {i} | **{r['symbol']}** | ₹{r['close']:.2f} | "
                      f"{r['best']:.3f} | ₹{r['entry_price']:.2f} | "
                      f"₹{r['target_price']:.2f} | ₹{r['stop_loss']:.2f} | "
                      f"{r['exit_by_date']} | {r['alloc_pct_of_capital']:.1f}% |")
        md.append(f"")
        md.append(f"## Risk envelope")
        md.append(f"")
        md.append(f"- Per-name max loss if SL hits: -3% × {qualified['alloc_pct_of_capital'].iloc[0]:.0f}% = "
                  f"-{0.03 * qualified['alloc_pct_of_capital'].iloc[0]:.2f}% portfolio MTM per stop")
        md.append(f"- Per-name max gain if target hits: +5% × {qualified['alloc_pct_of_capital'].iloc[0]:.0f}% = "
                  f"+{0.05 * qualified['alloc_pct_of_capital'].iloc[0]:.2f}% portfolio MTM per win")
        md.append(f"- Worst-case basket-wide (all SL): -{0.03 * 100:.1f}% × deployed-pct (capped by SL only)")

    md.append(f"")
    md.append(f"## Comparison vs prior 0.95-gate plan")
    md.append(f"")
    md.append(f"| | 0.95 single-name (prior) | 0.65 Top-10 (NEW) |")
    md.append(f"|---|---:|---:|")
    md.append(f"| 9-yr backtested CAGR | +12.2% | **+144.3%** |")
    md.append(f"| Trades/year | 8 | 534 |")
    md.append(f"| Capital deployment % | 5% | ~50% |")
    md.append(f"| Max drawdown | -17.7% | -21.6% |")
    md.append(f"| Sharpe ratio | 1.11 | 6.17 |")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"wrote {OUT_REPORT}")
    print(f"wrote {OUT_PARQUET}")
    print(f"\nQualified picks today (gate {GATE}): {len(qualified)}")
    if len(qualified):
        print(qualified[["symbol", "close", "best", "entry_price", "target_price", "stop_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
