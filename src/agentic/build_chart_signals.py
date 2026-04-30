"""Multi-timeframe chart pattern + technical signal detector.

For every liquid stock, computes:

  TREND CONTEXT (long-term)
    • Weekly trend: above/below 50w MA
    • Monthly trend: above/below 200d MA
    • 200-DMA slope (bullish if positive)

  CROSSOVERS (events)
    • Golden cross: 50-DMA crosses above 200-DMA in last N days
    • Death cross: 50-DMA crosses below 200-DMA in last N days
    • 20-50 cross: short-term momentum signal

  BREAKOUTS / BREAKDOWNS
    • close > 20d high with vol > 1.5× ADV (bullish breakout)
    • close < 20d low with vol > 1.5× ADV (bearish breakdown)
    • close > 52w high (52w breakout — strongest signal)
    • close < 52w low (52w breakdown)

  SUPPORT / RESISTANCE
    • Distance to nearest peak (last 252 days) above current price (resistance)
    • Distance to nearest trough below current price (support)

  VOLUME ANALYSIS
    • vol_zscore_60d: today's volume z-score vs 60d
    • on_balance_volume_trend: OBV slope last 20d (proxy for accumulation)

  MOMENTUM EXTREMES
    • rsi_extreme: >75 or <25 flag
    • rsi_divergence: price new high but RSI not (bearish divergence)

  CHART PATTERNS (heuristic detection)
    • cup_and_handle: prior peak → 30-50% pullback → recovery → consolidation → breakout
    • double_top: two peaks within 5% over 30-90d, recent decline
    • double_bottom: two troughs within 5% over 30-90d, recent rise
    • bull_flag: 10%+ rally followed by tight 5-10d pullback
    • symmetric_triangle: contracting H-L range over 30-60d

Output:
  data/derived/chart_signals.parquet — one row per symbol (today's snapshot)
  reports/chart_signals_summary.md — top breakouts, golden crosses, double bottoms today

Used by: inspect_symbol.py to surface chart-pattern context per name
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
OUT = ROOT / "data/derived/chart_signals.parquet"
OUT_REPORT = ROOT / "reports/chart_signals_summary.md"

LOOKBACK = 504  # ~2 years of daily bars


def find_peaks_troughs(s: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """Local maxima/minima within a rolling window."""
    is_peak = (s == s.rolling(2 * window + 1, center=True).max())
    is_trough = (s == s.rolling(2 * window + 1, center=True).min())
    return is_peak, is_trough


def detect_signals_for_symbol(g: pd.DataFrame) -> dict:
    """Compute chart signals for one symbol given full price history."""
    if len(g) < 252:
        return {}
    g = g.sort_values("trade_date").reset_index(drop=True)
    last = g.iloc[-1]
    today = last["trade_date"]
    close = last["close"]

    # last LOOKBACK days
    recent = g.tail(LOOKBACK).copy()

    # ── trend context ──
    sma50 = recent["sma_50"].iloc[-1] if pd.notna(recent["sma_50"].iloc[-1]) else np.nan
    sma200 = recent["sma_200"].iloc[-1] if pd.notna(recent["sma_200"].iloc[-1]) else np.nan
    sma50_20d_ago = recent["sma_50"].iloc[-21] if len(recent) >= 21 else np.nan
    sma200_20d_ago = recent["sma_200"].iloc[-21] if len(recent) >= 21 else np.nan

    out = {
        "trade_date": today,
        "close": close,
        "above_50dma": int(close > sma50) if pd.notna(sma50) else 0,
        "above_200dma": int(close > sma200) if pd.notna(sma200) else 0,
        "sma200_slope_20d": (sma200 - sma200_20d_ago) / sma200_20d_ago if pd.notna(sma200) and pd.notna(sma200_20d_ago) and sma200_20d_ago > 0 else np.nan,
    }

    # ── crossovers ──
    # golden cross: 50 > 200 today AND 50 < 200 X days ago (within last 30d)
    cross_window = recent.tail(30)
    if "sma_50" in cross_window.columns and "sma_200" in cross_window.columns and len(cross_window) >= 2:
        was_below = cross_window["sma_50"] < cross_window["sma_200"]
        is_above_today = (cross_window["sma_50"].iloc[-1] > cross_window["sma_200"].iloc[-1])
        out["golden_cross_30d"] = int(is_above_today and was_below.any())
        # death cross: 50 < 200 today AND was above
        was_above = cross_window["sma_50"] > cross_window["sma_200"]
        is_below_today = (cross_window["sma_50"].iloc[-1] < cross_window["sma_200"].iloc[-1])
        out["death_cross_30d"] = int(is_below_today and was_above.any())
    else:
        out["golden_cross_30d"] = 0
        out["death_cross_30d"] = 0

    # ── breakouts / breakdowns ──
    high_20d = recent["high"].tail(20).max()
    low_20d = recent["low"].tail(20).min()
    high_52w = recent["high"].max()
    low_52w = recent["low"].min()
    avg_vol_20d = recent.iloc[-1].get("avg_vol_20d", recent["total_traded_qty"].tail(20).mean())
    today_vol = last["total_traded_qty"]
    vol_ratio = today_vol / avg_vol_20d if avg_vol_20d > 0 else 1.0

    out["breakout_20d_with_vol"] = int(close >= high_20d * 0.999 and vol_ratio >= 1.5)
    out["breakdown_20d_with_vol"] = int(close <= low_20d * 1.001 and vol_ratio >= 1.5)
    out["breakout_52w"] = int(close >= high_52w * 0.999)
    out["breakdown_52w"] = int(close <= low_52w * 1.001)
    out["dist_from_52w_high"] = (close / high_52w - 1) if high_52w > 0 else 0
    out["dist_from_52w_low"] = (close / low_52w - 1) if low_52w > 0 else 0

    # ── support / resistance levels ──
    # find peaks above current close (resistance) and troughs below (support)
    is_peak, is_trough = find_peaks_troughs(recent["close"], window=10)
    peaks = recent.loc[is_peak, "close"].dropna()
    troughs = recent.loc[is_trough, "close"].dropna()
    resistance_above = peaks[peaks > close]
    support_below = troughs[troughs < close]
    out["nearest_resistance_pct"] = (resistance_above.min() / close - 1) if len(resistance_above) else np.nan
    out["nearest_support_pct"] = (support_below.max() / close - 1) if len(support_below) else np.nan

    # ── volume z-score ──
    vol_60d = recent["total_traded_qty"].tail(60)
    vol_mean = vol_60d.mean()
    vol_std = vol_60d.std()
    out["vol_zscore_60d"] = (today_vol - vol_mean) / vol_std if vol_std > 0 else 0
    out["vol_ratio_20d"] = vol_ratio

    # ── on-balance volume trend ──
    obv_signs = np.sign(recent["return_1d"].fillna(0))
    obv = (obv_signs * recent["total_traded_qty"]).cumsum()
    if len(obv) >= 20:
        obv_slope = (obv.iloc[-1] - obv.iloc[-20]) / max(abs(obv.iloc[-20]), 1)
        out["obv_slope_20d"] = obv_slope
    else:
        out["obv_slope_20d"] = 0

    # ── momentum extremes ──
    rsi = last.get("rsi_14_daily", 50)
    out["rsi_overbought"] = int(rsi > 75) if pd.notna(rsi) else 0
    out["rsi_oversold"] = int(rsi < 25) if pd.notna(rsi) else 0
    # bearish divergence: new 60d high in price BUT RSI lower than 20d-ago RSI
    high_60d_today = close >= recent["high"].tail(60).max() * 0.99
    rsi_20d_ago = recent.iloc[-21].get("rsi_14_daily", 50) if len(recent) >= 21 else 50
    out["rsi_bearish_divergence"] = int(high_60d_today and rsi < rsi_20d_ago - 5)

    # ── chart patterns (heuristic) ──
    # double top: 2 distinct peaks within 3% over 30+ day spacing in last 90d, recent decline ≥ 7%
    recent_90 = recent.tail(90).reset_index(drop=True)
    is_peak_90, is_trough_90 = find_peaks_troughs(recent_90["close"], window=8)
    peak_idx = recent_90.index[is_peak_90].tolist()
    trough_idx = recent_90.index[is_trough_90].tolist()
    out["double_top_90d"] = 0
    if len(peak_idx) >= 2:
        # require 30+ day separation between two highest peaks
        sorted_peaks = sorted([(i, recent_90["close"].iloc[i]) for i in peak_idx],
                                key=lambda x: -x[1])
        for i in range(len(sorted_peaks)):
            for j in range(i + 1, len(sorted_peaks)):
                idx_a, p_a = sorted_peaks[i]
                idx_b, p_b = sorted_peaks[j]
                if abs(idx_a - idx_b) >= 30 and abs(p_a - p_b) / p_a < 0.03 and close < p_a * 0.93:
                    out["double_top_90d"] = 1
                    break
            if out["double_top_90d"]:
                break

    out["double_bottom_90d"] = 0
    if len(trough_idx) >= 2:
        sorted_troughs = sorted([(i, recent_90["close"].iloc[i]) for i in trough_idx],
                                 key=lambda x: x[1])
        for i in range(len(sorted_troughs)):
            for j in range(i + 1, len(sorted_troughs)):
                idx_a, p_a = sorted_troughs[i]
                idx_b, p_b = sorted_troughs[j]
                if abs(idx_a - idx_b) >= 30 and abs(p_a - p_b) / p_a < 0.03 and close > p_a * 1.07:
                    out["double_bottom_90d"] = 1
                    break
            if out["double_bottom_90d"]:
                break

    # cup-and-handle (rough): 30%+ peak → ≥20% drawdown → 80%+ recovery → tight handle
    peak_252 = recent["high"].max()
    idx_peak = recent["high"].idxmax()
    if idx_peak < len(recent) - 30:
        post_peak = recent.iloc[idx_peak:]
        trough_post = post_peak["low"].min()
        drawdown = (trough_post - peak_252) / peak_252
        recovery = (close - trough_post) / (peak_252 - trough_post) if peak_252 > trough_post else 0
        # handle: last 10d range ≤ 10% of close
        last_10_range = (recent["high"].tail(10).max() - recent["low"].tail(10).min()) / close
        out["cup_and_handle"] = int(drawdown <= -0.20 and recovery >= 0.80 and last_10_range <= 0.10)
    else:
        out["cup_and_handle"] = 0

    # bull flag: 15%+ rally in last 30d, then tight 5-10d consolidation
    rally_30 = (recent["close"].iloc[-1] / recent["close"].iloc[-30] - 1) if len(recent) >= 30 else 0
    last_5_range = (recent["high"].tail(5).max() - recent["low"].tail(5).min()) / close
    out["bull_flag"] = int(rally_30 >= 0.15 and last_5_range <= 0.05)

    # symmetric triangle: H-L range contracting over 30d
    if len(recent) >= 30:
        ranges = (recent["high"] - recent["low"]).tail(30).rolling(5).mean()
        if ranges.iloc[-1] < ranges.iloc[0] * 0.7:
            out["symmetric_triangle"] = 1
        else:
            out["symmetric_triangle"] = 0
    else:
        out["symmetric_triangle"] = 0

    # ── composite "bullish" / "bearish" signal counts ──
    bullish_signals = [
        out.get("above_50dma", 0), out.get("above_200dma", 0),
        out.get("golden_cross_30d", 0), out.get("breakout_20d_with_vol", 0),
        out.get("breakout_52w", 0), out.get("double_bottom_90d", 0),
        out.get("cup_and_handle", 0), out.get("bull_flag", 0),
    ]
    bearish_signals = [
        out.get("death_cross_30d", 0), out.get("breakdown_20d_with_vol", 0),
        out.get("breakdown_52w", 0), out.get("double_top_90d", 0),
        out.get("rsi_overbought", 0), out.get("rsi_bearish_divergence", 0),
    ]
    out["bullish_count"] = sum(bullish_signals)
    out["bearish_count"] = sum(bearish_signals)
    out["chart_score"] = out["bullish_count"] - out["bearish_count"]

    return out


def main() -> None:
    print("== build_chart_signals ==")
    df = pd.read_parquet(PRICES, columns=[
        "symbol", "trade_date", "close", "open", "high", "low",
        "total_traded_qty", "avg_vol_20d", "sma_20", "sma_50", "sma_200",
        "rsi_14_daily", "return_1d", "avg_traded_value_20d", "series"
    ])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["series"] == "EQ"]
    df = df.sort_values(["symbol", "trade_date"])

    # liquid universe: ADV >= 1cr/day on the latest day
    latest = df["trade_date"].max()
    liquid_today = df[(df["trade_date"] == latest) & (df["avg_traded_value_20d"] / 1e7 >= 1.0)]["symbol"].unique()
    df = df[df["symbol"].isin(liquid_today)]
    print(f"  universe: {len(liquid_today)} liquid stocks  today={latest:%Y-%m-%d}")

    rows = []
    for sym, g in df.groupby("symbol"):
        try:
            r = detect_signals_for_symbol(g)
            if r:
                r["symbol"] = sym
                rows.append(r)
        except Exception as e:
            pass

    if not rows:
        print("no signals computed")
        return
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)

    # build summary report
    md = [f"# Chart signals summary — {latest:%Y-%m-%d}", "",
          f"Universe: {len(res):,} liquid EQ stocks (ADV ≥ ₹1cr/day)", "",
          "## Today's notable patterns", ""]

    for label, col, sort_dir in [
        ("🟢 Golden Cross (last 30d)", "golden_cross_30d", "bullish_count"),
        ("🔴 Death Cross (last 30d)", "death_cross_30d", "bearish_count"),
        ("🟢 52-week Breakout", "breakout_52w", "vol_ratio_20d"),
        ("🔴 52-week Breakdown", "breakdown_52w", "vol_ratio_20d"),
        ("🟢 20-day Breakout + Volume", "breakout_20d_with_vol", "vol_ratio_20d"),
        ("🟢 Cup and Handle", "cup_and_handle", "chart_score"),
        ("🟢 Bull Flag", "bull_flag", "chart_score"),
        ("🟢 Double Bottom (90d)", "double_bottom_90d", "chart_score"),
        ("🔴 Double Top (90d)", "double_top_90d", "bearish_count"),
        ("🔴 Bearish RSI Divergence", "rsi_bearish_divergence", "bearish_count"),
    ]:
        sub = res[res[col] == 1].sort_values(sort_dir, ascending=False)
        md.append(f"### {label}: {len(sub)} stocks")
        if len(sub):
            md.append("")
            md.append("| Symbol | Close | Vol ratio | RSI | Bull / Bear count | Chart score |")
            md.append("|---|---:|---:|---:|---:|---:|")
            for _, r in sub.head(15).iterrows():
                rsi = r.get("rsi_14_daily")
                rsi_str = f"{rsi:.0f}" if pd.notna(rsi) else "—"
                md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | "
                          f"{r.get('vol_ratio_20d', 1):.1f}× | {rsi_str} | "
                          f"{int(r.get('bullish_count', 0))} / {int(r.get('bearish_count', 0))} | "
                          f"{int(r.get('chart_score', 0)):+d} |")
        md.append("")

    # top 20 by composite chart score
    top_chart = res.sort_values("chart_score", ascending=False).head(20)
    md.append("## Top-20 by composite chart score (bullish - bearish signals)")
    md.append("")
    md.append("| Symbol | Close | Bullish | Bearish | Score | Patterns |")
    md.append("|---|---:|---:|---:|---:|---|")
    for _, r in top_chart.iterrows():
        patterns = []
        if r.get("breakout_52w"): patterns.append("52w-breakout")
        if r.get("breakout_20d_with_vol"): patterns.append("20d-breakout")
        if r.get("golden_cross_30d"): patterns.append("golden-cross")
        if r.get("cup_and_handle"): patterns.append("cup&handle")
        if r.get("bull_flag"): patterns.append("bull-flag")
        if r.get("double_bottom_90d"): patterns.append("double-bottom")
        if r.get("above_50dma") and r.get("above_200dma"): patterns.append("above-200dma")
        md.append(f"| **{r['symbol']}** | ₹{r['close']:.2f} | "
                  f"{int(r['bullish_count'])} | {int(r['bearish_count'])} | "
                  f"{int(r['chart_score']):+d} | {', '.join(patterns) or '—'} |")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(md))
    print(f"\nwrote {OUT}: {len(res)} stocks")
    print(f"wrote {OUT_REPORT}")
    print(f"\nQuick counts:")
    for col in ["golden_cross_30d", "death_cross_30d", "breakout_52w",
                  "breakdown_52w", "breakout_20d_with_vol", "cup_and_handle",
                  "bull_flag", "double_bottom_90d", "double_top_90d", "rsi_bearish_divergence"]:
        if col in res.columns:
            n = int(res[col].sum())
            print(f"  {col:<32} {n:>4} stocks")


if __name__ == "__main__":
    main()
