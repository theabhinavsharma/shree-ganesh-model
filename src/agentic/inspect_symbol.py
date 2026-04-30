"""Per-symbol inspector — given a ticker, dump everything the system knows.

Usage:
  python src/agentic/inspect_symbol.py WEBELSOLAR
  python src/agentic/inspect_symbol.py OFSS
  python src/agentic/inspect_symbol.py RELIANCE TCS HDFCBANK   # multiple

Pulls together: price + technicals, fundamentals, catalysts, sentiment,
Wikipedia, all model scores, sector context, recent paper-trade outcomes.
Identifies whether the system has a buy/sell/hold signal RIGHT NOW.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CAT = ROOT / "data/derived/catalyst_features.parquet"
FUND = ROOT / "data/derived/fundamentals_snapshot.parquet"
NEWS = ROOT / "data/derived/news_features.parquet"
MACRO = ROOT / "data/derived/macro_sentiment.parquet"
WIKI = ROOT / "data/derived/wiki_pageviews.parquet"
LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
LIVE_SHORT = ROOT / "tmp/from_scratch_7d_run/short_live_top100.csv"
MH = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"
SWS = ROOT / "tmp/from_scratch_7d_run/sector_weak_shorts.csv"
LEDGER = ROOT / "data/derived/paper_trading_ledger.parquet"
CHART_SIGNALS = ROOT / "data/derived/chart_signals.parquet"

LINE = "─" * 80


def section(title: str) -> str:
    return f"\n{LINE}\n{title}\n{LINE}"


def inspect_one(sym: str) -> str:
    out = []
    out.append(section(f"  📊 {sym}"))

    # 1. price + technicals
    df = pd.read_parquet(PRICES)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    sym_df = df[df["symbol"] == sym]
    if sym_df.empty:
        return f"\n{sym}: NOT FOUND in price parquet\n"
    sym_df = sym_df.sort_values("trade_date")
    last = sym_df.iloc[-1]
    out.append(section("PRICE / TECHNICAL"))
    out.append(f"  As of:      {last['trade_date']:%Y-%m-%d}  (close ₹{last['close']:.2f})")
    out.append(f"  1-day:      {(last.get('return_1d',0) or 0)*100:+.2f}%")
    out.append(f"  20-day:     {(last.get('return_20d',0) or 0)*100:+.2f}%")
    if pd.notna(last.get("sma_20")):
        out.append(f"  vs SMA20:   {(last['close']/last['sma_20']-1)*100:+.2f}%")
    if pd.notna(last.get("sma_50")):
        out.append(f"  vs SMA50:   {(last['close']/last['sma_50']-1)*100:+.2f}%")
    if pd.notna(last.get("sma_200")):
        out.append(f"  vs SMA200:  {(last['close']/last['sma_200']-1)*100:+.2f}%")
    out.append(f"  RSI daily:  {last.get('rsi_14_daily', '—')}")
    out.append(f"  RSI weekly: {last.get('rsi_14_weekly', '—')}")
    out.append(f"  Vol vs 20d: {last.get('volume_vs_20d','—'):.2f}×" if pd.notna(last.get('volume_vs_20d')) else "  Vol vs 20d: —")
    out.append(f"  Delivery%:  {last.get('delivery_pct','—')}")
    if pd.notna(last.get("avg_traded_value_20d")):
        out.append(f"  ADV 20d:    ₹{last['avg_traded_value_20d']/1e7:.1f}cr/day")

    # 2. fundamentals
    if FUND.exists():
        f = pd.read_parquet(FUND)
        sub = f[f["symbol"] == sym]
        if not sub.empty:
            r = sub.sort_values("fetch_date").iloc[-1]
            out.append(section("FUNDAMENTAL"))
            out.append(f"  PE:         {r.get('pe','—')}")
            out.append(f"  Sector PE:  {r.get('sector_pe','—')}")
            if pd.notna(r.get('pe')) and pd.notna(r.get('sector_pe')):
                out.append(f"  Vs sector:  {(r['pe']/r['sector_pe']-1)*100:+.0f}%")
            out.append(f"  52w high:   ₹{r.get('week52_high','—')}  ({r.get('dist_from_52w_high_pct','—')}% off)")
            out.append(f"  Last Q rev: ₹{r.get('last_q_revenue','—')} cr")
            out.append(f"  Last Q PAT: ₹{r.get('last_q_pat','—')} cr")
            out.append(f"  QoQ rev:    {r.get('qoq_revenue_growth','—')}%")
            out.append(f"  QoQ PAT:    {r.get('qoq_pat_growth','—')}%")

    # 3. catalysts
    if CAT.exists():
        c = pd.read_parquet(CAT)
        c["trade_date"] = pd.to_datetime(c["trade_date"])
        sub = c[c["symbol"] == sym].sort_values("trade_date")
        if not sub.empty:
            r = sub.iloc[-1]
            out.append(section("CATALYSTS (latest row)"))
            out.append(f"  Date:       {r['trade_date']:%Y-%m-%d}")
            out.append(f"  ann_5d:     {r.get('ann_5d_count','—')}   ann_30d: {r.get('ann_30d_count','—')}")
            out.append(f"  orders 5d:  {r.get('ann_order_5d','—')}   capex 30d: {r.get('ann_capex_30d','—')}")
            out.append(f"  results 5d: {r.get('ann_result_5d','—')}   buyback 30d: {r.get('ann_buyback_30d','—')}")
            ins = r.get('insider_net_60d_inr', 0) or 0
            blk = r.get('block_buy_5d_inr', 0) or 0
            out.append(f"  Insider net 60d: ₹{ins/1e5:+.1f}L")
            out.append(f"  Block buy  5d:   ₹{blk/1e5:+.1f}L")

    # 4. sentiment + news
    if NEWS.exists():
        n = pd.read_parquet(NEWS)
        sub = n[n["symbol"] == sym]
        if not sub.empty:
            r = sub.sort_values("as_of").iloc[-1]
            out.append(section("SENTIMENT (last 5d)"))
            out.append(f"  News mentions:    {r.get('news_count_5d',0):.0f}   sent: {r.get('news_sentiment_5d',0):+.2f}")
            out.append(f"  Reddit mentions:  {r.get('reddit_mentions_5d',0):.0f}   sent: {r.get('reddit_sentiment_5d',0):+.2f}")
            out.append(f"  YouTube mentions: {r.get('youtube_mentions_5d',0):.0f}   sent: {r.get('youtube_sentiment_5d',0):+.2f}")

    # 5. Wikipedia
    if WIKI.exists():
        w = pd.read_parquet(WIKI)
        sub = w[w["symbol"] == sym]
        if not sub.empty:
            r = sub.sort_values("trade_date").iloc[-1]
            out.append(section("WIKIPEDIA (retail attention)"))
            out.append(f"  Daily views:      {r.get('wiki_views',0):.0f}")
            out.append(f"  7d mean:          {r.get('wiki_views_7d_mean',0):.0f}")
            out.append(f"  30d-baseline z:   {r.get('wiki_views_z',0):+.2f}")

    # 6. macro context
    if MACRO.exists():
        m = pd.read_parquet(MACRO).sort_values("as_of").iloc[-1].to_dict()
        gms = m.get("global_macro_sent", 0) or 0
        dms = m.get("domestic_macro_sent", 0) or 0
        overall = (gms + dms) / 2
        regime = "🔴 RISK_OFF" if overall <= -0.3 else ("🟢 RISK_ON" if overall >= 0.3 else "◯ NEUTRAL")
        out.append(section("MACRO CONTEXT (today)"))
        out.append(f"  Regime: {regime}  (score {overall:+.2f})")
        out.append(f"  Global: {gms:+.2f}   Domestic: {dms:+.2f}")

    # 7. model scores
    out.append(section("MODEL SIGNALS"))
    if LIVE_LONG.exists():
        ll = pd.read_csv(LIVE_LONG)
        sub = ll[ll["symbol"] == sym]
        if not sub.empty:
            r = sub.iloc[0]
            out.append(f"  LONG  v3 score_ens: {r.get('score_ens', 0):.3f}   score_cal: {r.get('score_calibrated', 0):.3f}")
        else:
            out.append(f"  LONG  v3:  not in top-100 today")
    if LIVE_SHORT.exists():
        ls = pd.read_csv(LIVE_SHORT)
        sub = ls[ls["symbol"] == sym]
        if not sub.empty:
            r = sub.iloc[0]
            out.append(f"  SHORT model score_ens: {r.get('score_ens', 0):.3f}   score_cal: {r.get('score_calibrated', 0):.3f}")
        else:
            out.append(f"  SHORT model: not in top-100 today")
    if MH.exists():
        mh = pd.read_csv(MH)
        sub = mh[mh["symbol"] == sym]
        if not sub.empty:
            r = sub.iloc[0]
            out.append(f"  MULTI-HORIZON consensus: {r.get('consensus', 0):.3f}   triangulated: {r.get('triangulated', False)}")
            out.append(f"    h1_cal={r.get('score_h1_cal',0):.2f}  h7_cal={r.get('score_h7_cal',0):.2f}  h21_cal={r.get('score_h21_cal',0):.2f}")
    if SWS.exists():
        sw = pd.read_csv(SWS)
        sub = sw[sw["symbol"] == sym]
        if not sub.empty:
            r = sub.iloc[0]
            out.append(f"  SECTOR-WEAK SHORT overlay: in list  (sector {r['sector']} 5d {r['sector_5d_ret_pct']:+.1f}%)")
            out.append(f"    reason: {r.get('reason','—')}")

    # 7b. chart patterns (multi-timeframe technical analysis)
    if CHART_SIGNALS.exists():
        cs = pd.read_parquet(CHART_SIGNALS)
        sub = cs[cs["symbol"] == sym]
        if not sub.empty:
            r = sub.iloc[0]
            out.append(section("CHART PATTERNS (multi-timeframe technical)"))
            patterns_active = []
            if r.get("breakout_52w"): patterns_active.append("🟢 52-week BREAKOUT")
            if r.get("breakdown_52w"): patterns_active.append("🔴 52-week BREAKDOWN")
            if r.get("breakout_20d_with_vol"): patterns_active.append("🟢 20-day breakout + volume")
            if r.get("breakdown_20d_with_vol"): patterns_active.append("🔴 20-day breakdown + volume")
            if r.get("golden_cross_30d"): patterns_active.append("🟢 Golden Cross (50 over 200)")
            if r.get("death_cross_30d"): patterns_active.append("🔴 Death Cross (50 under 200)")
            if r.get("cup_and_handle"): patterns_active.append("🟢 Cup and Handle")
            if r.get("bull_flag"): patterns_active.append("🟢 Bull Flag")
            if r.get("double_bottom_90d"): patterns_active.append("🟢 Double Bottom (90d)")
            if r.get("double_top_90d"): patterns_active.append("🔴 Double Top (90d)")
            if r.get("symmetric_triangle"): patterns_active.append("◯ Symmetric Triangle (consolidation)")
            if r.get("rsi_overbought"): patterns_active.append("⚠️  RSI overbought (>75)")
            if r.get("rsi_oversold"): patterns_active.append("⚠️  RSI oversold (<25)")
            if r.get("rsi_bearish_divergence"): patterns_active.append("🔴 RSI Bearish Divergence")
            if patterns_active:
                for p in patterns_active:
                    out.append(f"  {p}")
            else:
                out.append("  No notable patterns detected today.")
            out.append("")
            out.append(f"  Trend context:")
            out.append(f"    Above 50-DMA:  {'✓' if r.get('above_50dma') else '✗'}")
            out.append(f"    Above 200-DMA: {'✓' if r.get('above_200dma') else '✗'}")
            sma200_slope = r.get("sma200_slope_20d", 0) or 0
            out.append(f"    200-DMA 20d slope: {sma200_slope*100:+.2f}% (bullish if positive)")
            out.append(f"    Distance from 52w high: {(r.get('dist_from_52w_high', 0) or 0)*100:+.1f}%")
            out.append(f"    Distance from 52w low:  {(r.get('dist_from_52w_low', 0) or 0)*100:+.1f}%")
            out.append("")
            out.append(f"  Support / Resistance:")
            res_pct = r.get("nearest_resistance_pct")
            sup_pct = r.get("nearest_support_pct")
            if pd.notna(res_pct):
                out.append(f"    Nearest resistance:  {res_pct*100:+.1f}% above (peak in last 2y)")
            if pd.notna(sup_pct):
                out.append(f"    Nearest support:     {sup_pct*100:+.1f}% below (trough in last 2y)")
            out.append("")
            out.append(f"  Volume:")
            out.append(f"    Today's vol z-score (60d): {r.get('vol_zscore_60d', 0):+.2f}")
            out.append(f"    Today's vol vs 20d ADV: {r.get('vol_ratio_20d', 1):.2f}×")
            out.append(f"    OBV slope (20d): {r.get('obv_slope_20d', 0)*100:+.1f}%")
            out.append("")
            out.append(f"  Composite chart score: "
                       f"{int(r.get('bullish_count', 0))} bull / {int(r.get('bearish_count', 0))} bear = "
                       f"{int(r.get('chart_score', 0)):+d}")

    # 8. paper-trade ledger
    if LEDGER.exists():
        led = pd.read_parquet(LEDGER)
        sub = led[led["symbol"] == sym]
        if not sub.empty:
            out.append(section("PAPER TRADING LEDGER"))
            for _, r in sub.tail(5).iterrows():
                out.append(f"  {pd.Timestamp(r['snapshot_date']):%Y-%m-%d}: status={r['status']:6} entry=₹{r['entry_close']:.2f}  "
                          f"outcome={r.get('outcome','open')}  pwin_cal={r.get('pwin_cal',0):.2f}")

    # 9. honest verdict
    out.append(section("SYSTEM VERDICT"))
    long_score = 0
    short_score = 0
    if LIVE_LONG.exists():
        sub = pd.read_csv(LIVE_LONG)
        s = sub[sub["symbol"] == sym]
        long_score = float(s["score_calibrated"].iloc[0]) if not s.empty else 0
    if LIVE_SHORT.exists():
        sub = pd.read_csv(LIVE_SHORT)
        s = sub[sub["symbol"] == sym]
        short_score = float(s["score_calibrated"].iloc[0]) if not s.empty else 0
    if long_score >= 0.80:
        out.append(f"  🟢 HIGH-CONVICTION LONG  (score_cal {long_score:.2f})")
    elif long_score >= 0.65:
        out.append(f"  🟡 weak long signal  (score_cal {long_score:.2f}) — below 0.75 RISK_OFF floor")
    if short_score >= 0.70:
        out.append(f"  🔴 SHORT signal  (score_cal {short_score:.2f})")
    if long_score < 0.65 and short_score < 0.50:
        out.append(f"  ◯ no signal — model not calling this name today")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+", help="Ticker(s) to inspect, e.g. WEBELSOLAR OFSS")
    args = ap.parse_args()
    for sym in args.symbols:
        print(inspect_one(sym.upper()))
        print()


if __name__ == "__main__":
    main()
