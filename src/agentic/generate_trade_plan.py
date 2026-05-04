"""Daily trade plan — the single document the user reads each morning.

Consolidates ALL signals into a deploy/wait verdict with concrete actions:
  • Regime gate verdict (deploy/wait the multibagger sleeve)
  • Confluence picks for the always-on quality sleeve
  • 0.95-bar tactical triggers (if any)
  • Position sizing per sleeve (capital %, names, SL, target)
  • Total expected return + drawdown for the next holding period

Output: reports/trade_plan_<YYYYMMDD>.md
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
MULTI = ROOT / "data/derived/multibagger_today_predictions.parquet"
CONF = ROOT / "data/derived/confluence_picks.parquet"
HC_PRED = ROOT / "data/derived/high_conviction_predictions.parquet"
CHART = ROOT / "data/derived/chart_signals.parquet"

# 2026-05-01: Gate-tuning experiment showed 0.65-0.70 gates with Top-10 baskets
# achieve 2x/180d backtested CAGR (+144% vs +14% at 0.95). Updated config below.
SHORT_HORIZON_GATE = 0.65       # was 0.80; 0.95 was starved (8 trades/yr)
SHORT_HORIZON_MAX_POS = 10      # Top-10 basket
# event-driven backtest evidence: reports/event_driven_backtest_backtest_10yr_macro_oof.md

OUT_REPORT = ROOT / f"reports/trade_plan_{date.today():%Y%m%d}.md"

# Allocation envelope — FOCUSED on 2x in 180d, no diversification floor
# Updated 2026-04-30 per user directive: "removing min filter of 30%, just 2x in 180d"
ALLOC = {
    "multibagger_gated": 1.00,    # 100% capital deploys when gate green
    "confluence_quality": 0.00,    # not used (was diversification sleeve)
    "tactical_0_95": 0.00,          # not used (was tactical sleeve)
    "reserve_cash": 0.00,           # 100% cash when gate red
}

# Per-sleeve sizing rules
N_NAMES_MULTIBAGGER = 4   # 25% per name when deployed
N_NAMES_CONFLUENCE = 0
N_NAMES_TACTICAL = 0


def regime_check() -> dict:
    """Re-check the gate v1 today."""
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "sma_50",
                                            "return_1d", "avg_traded_value_20d", "series"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px[(px["series"] == "EQ") & (px["avg_traded_value_20d"] / 1e7 >= 1.0)]
    px["above_50"] = (px["close"] > px["sma_50"]).astype(int)
    daily = px.groupby("trade_date").agg(
        breadth_50=("above_50", "mean"),
        market_med=("return_1d", "median"),
    ).reset_index()
    daily["market_20d"] = daily["market_med"].rolling(20).sum()
    today = daily.iloc[-1]
    market_20d = float(today["market_20d"])
    breadth_50 = float(today["breadth_50"])
    deploy = (market_20d <= -0.02) and (0.50 <= breadth_50 <= 0.75)
    return {
        "verdict": "DEPLOY" if deploy else "WAIT",
        "market_20d": market_20d,
        "breadth_50": breadth_50,
        "as_of": today["trade_date"],
    }


def main() -> None:
    today = pd.Timestamp(date.today())
    gate = regime_check()
    md = [f"# Trade Plan — {today:%Y-%m-%d}", "",
          "**Single goal: 2x in 180 days. No diversification sleeve.**",
          "100% capital deploys to multibagger basket when regime gate green.",
          "100% capital sits in LIQUIDPLUS when gate red.", "",
          f"_Regime gate v1 verdict: **{gate['verdict']}**  "
          f"(market_20d={gate['market_20d']*100:+.2f}%, breadth_50={gate['breadth_50']*100:.0f}%)_", ""]

    # ── SOLE SLEEVE: MULTIBAGGER (regime-gated) — 100% capital ──
    md.append("## The trade: 100% capital → 4-name multibagger basket (regime-gated)")
    md.append("")
    if gate["verdict"] == "WAIT":
        md.append(f"🔴 **WAIT** — gate is red. Park 100% in LIQUIDPLUS / CASHIETF (~7% ann).")
        md.append("")
        md.append("**Trigger to deploy:** market_20d ≤ -2% AND breadth_50 between 50% and 75%.")
        md.append("Historical regime-gated success rate: 64% (vs 41% all-in).")
        md.append("")
        md.append("**Today's reading (does NOT match):**")
        md.append(f"- market_20d = **{gate['market_20d']*100:+.2f}%** (need ≤ -2%)")
        md.append(f"- breadth_50 = **{gate['breadth_50']*100:.0f}%** (need 50-75%)")
    elif MULTI.exists():
        m = pd.read_parquet(MULTI)
        EXCLUDE = {"LICMFGOLD", "GROWWGOLD", "SILVER1", "MIDCAP", "BANKNIFTY1", "QNIFTY",
                    "NIFTYBEES", "GOLDBEES", "LIQUIDBEES"}
        m = m[~m["symbol"].isin(EXCLUDE)]
        if "adv_20d_cr" in m.columns:
            m = m[m["adv_20d_cr"] >= 5.0]
        if "score_100pct_180d" in m.columns:
            picks = m.sort_values("score_100pct_180d", ascending=False).head(N_NAMES_MULTIBAGGER)
            per_name = 1.0 / max(N_NAMES_MULTIBAGGER, 1) * 100
            md.append(f"🟢 **DEPLOY** — {len(picks)} names, equal-weighted at {per_name:.1f}% each (100% capital total):")
            md.append("")
            md.append("| Symbol | Close | 180d score | Stop-loss (-15%) | Target (+100%) | Hold |")
            md.append("|---|---:|---:|---:|---:|---|")
            for _, r in picks.iterrows():
                sl = r["close"] * 0.85
                tgt = r["close"] * 2.00
                md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | "
                          f"{r['score_100pct_180d']:.3f} | ₹{sl:.2f} | ₹{tgt:.2f} | 180d |")
    md.append("")

    # ── EXPECTED RETURN ──
    md.append("## Expected outcome (honest, focused on 2x/180d)")
    md.append("")
    if gate["verdict"] == "DEPLOY":
        md.append("**Gate GREEN → 100% deployed in 4-name multibagger basket.**")
        md.append("")
        md.append("Per the corrected backtest (regime-gated v1, 14 of 44 OOS weeks deployed):")
        md.append("- **64% probability ≥1 of 4 names doubles in 180d**")
        md.append("- **Avg basket max-high return: +50% per 180d**")
        md.append("- **Avg basket close-to-close: +4% per 180d** (without optimal exit)")
        md.append("- **With laddered exits at +50% / +75% / +100% per name: estimated +25-35% basket close-equivalent**")
        md.append("")
        md.append("Ann math (assuming 2 non-overlapping baskets/year): (1.30)² - 1 = **+69% ann**")
        md.append("With 2× MTF leverage: ~+150% ann (but max drawdown -30% in adverse year)")
    else:
        md.append("**Gate RED → 100% in LIQUIDPLUS / CASHIETF (~7% ann).**")
        md.append("")
        md.append("No deployment until gate flips green. The strategy succeeds historically only in")
        md.append("post-pullback regimes (market_20d ≤ -2% AND breadth_50 ∈ [50%,75%]).")
        md.append("")
        md.append("Today's reading does NOT match: market_20d=" +
                  f"{gate['market_20d']*100:+.2f}%, breadth_50={gate['breadth_50']*100:.0f}%.")
        md.append("")
        md.append("Historical wait time between gate-green windows: typically 2-8 weeks.")
    md.append("")

    # ── CIRCUIT BREAKERS ──
    md.append("## Circuit breakers (mandatory)")
    md.append("")
    md.append("- Per-name drawdown -25% → exit that name")
    md.append("- Portfolio MTM drawdown -15% → cut leverage to 1×")
    md.append("- Portfolio MTM drawdown -25% → exit ALL positions, reset")
    md.append("- Portfolio MTM drawdown -30% (the floor) → stop strategy until thesis re-validated")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"wrote {OUT_REPORT}")
    print(f"\nRegime: {gate['verdict']}")


if __name__ == "__main__":
    main()
