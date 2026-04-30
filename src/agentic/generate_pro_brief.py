"""Pro-format daily brief — every top pick gets:

  • Header (symbol, close, side, conf%, sizing)
  • Plain-English thesis (1 line)
  • Valuation vs sector peers (PE, sector PE, 52w distance, QoQ growth)
  • Technical state (ret20d, RSI, sma distances, vol)
  • Catalysts (counts + types from last 5/30d)
  • Expected return (E[+5% high] and E[close 7d]) from real OOS band stats
  • Bull / Base / Bear with calibrated probabilities
  • Risk acknowledged (slippage, liquidity, regime)
  • Position sizing math
  • Links (NSE / Screener / TradingView)

Modeled on The Claude Portfolio's per-position template (finbold trade-recap evidence).

Reads:
  tmp/from_scratch_7d_run/v3_live_top100.csv
  tmp/from_scratch_7d_run/multi_horizon_top.csv
  tmp/from_scratch_7d_run/short_live_top100.csv
  tmp/from_scratch_7d_run/score_band_stats.csv
  data/derived/fundamentals_snapshot.parquet (graceful if missing rows)
  data/derived/catalyst_features.parquet
  data/derived/stock_daily_facts_adjusted_2015plus.parquet

Writes:
  reports/daily_pro_brief_<YYYYMMDD>.md
"""
from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/abhinavs./Documents/Zoom")
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_LONG = ROOT / "tmp/from_scratch_7d_run/v3_live_top100.csv"
LIVE_SHORT = ROOT / "tmp/from_scratch_7d_run/short_live_top100.csv"
LIVE_MH = ROOT / "tmp/from_scratch_7d_run/multi_horizon_top.csv"
BAND_STATS = ROOT / "tmp/from_scratch_7d_run/score_band_stats.csv"
PRICES = ROOT / "data/derived/stock_daily_facts_adjusted_2015plus.parquet"
CATALYSTS = ROOT / "data/derived/catalyst_features.parquet"
FUNDAMENTALS = ROOT / "data/derived/fundamentals_snapshot.parquet"
NEWS_FEAT = ROOT / "data/derived/news_features.parquet"
MACRO_SENT = ROOT / "data/derived/macro_sentiment.parquet"


def _band_for(score_cal: float, bands: pd.DataFrame) -> pd.Series:
    """Return the OOS band stats row for a given calibrated score."""
    edges = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]
    for i, hi in enumerate(edges[1:], start=0):
        if score_cal <= hi:
            return bands.iloc[i]
    return bands.iloc[-1]


def _thesis(row: dict) -> str:
    """Compose one-line plain-English thesis from feature pattern."""
    parts = []
    if row.get("triangulated"):
        parts.append("triangulated 1d/7d/21d agreement")
    rsi = row.get("rsi_14_daily") or 50
    vol = row.get("volume_vs_20d") or 1
    ret20 = (row.get("return_20d") or 0) * 100
    if rsi >= 80:
        parts.append(f"RSI {rsi:.0f} extended (climax risk)")
    elif rsi <= 35:
        parts.append(f"RSI {rsi:.0f} oversold bounce setup")
    if vol >= 2.5:
        parts.append(f"vol {vol:.1f}× ADV momentum-confirmed")
    elif vol < 0.5:
        parts.append("silent accumulation (low volume)")
    if ret20 >= 30:
        parts.append(f"+{ret20:.0f}% in 20d trend")
    elif ret20 <= -10:
        parts.append(f"{ret20:.0f}% in 20d, mean-reversion candidate")
    cat_30 = row.get("ann_30d_count") or 0
    if cat_30 >= 3:
        parts.append(f"{cat_30:.0f} catalysts in 30d")
    return "; ".join(parts) if parts else "model-only signal, no strong feature pattern"


def _bull_base_bear(score_cal: float, band: pd.Series) -> dict:
    """Three-state outcome with OOS-grounded probabilities.
    Bull = +15% swing target hit
    Base = +5% high but no swing
    Bear = -5% low touched
    Probs may not sum to 1 (mutually exclusive paths can overlap)."""
    p_swing = float(band["p_swing15"])
    p_high = float(band["p_winner"])
    p_low5 = float(band["p_min_minus5"])
    return {
        "p_bull_15pct": round(p_swing, 3),
        "p_base_5pct": round(max(0.0, p_high - p_swing), 3),
        "p_bear_minus5": round(p_low5, 3),
        "expected_high_pct": round(float(band["mean_high"]) * 100, 2),
        "expected_close_pct": round(float(band["mean_close"]) * 100, 2),
        "p_close_pos": round(float(band["p_close_pos"]), 3),
    }


def _risk_notes(row: dict) -> list[str]:
    notes = []
    adv_cr = (row.get("avg_traded_value_20d") or 0) / 1e7
    if adv_cr < 5:
        notes.append(f"thin liquidity (ADV ₹{adv_cr:.1f}cr/day) — slippage on exit if size > ₹{adv_cr*5:.0f}L")
    rsi = row.get("rsi_14_daily") or 50
    if rsi >= 85:
        notes.append("RSI > 85 — single -3% candle can flush 60%+ of holders")
    sec = (row.get("sector") or "OTHER")
    if "MICROCAP" in sec:
        notes.append("MICROCAP — no F&O hedge available, full beta to sentiment")
    if (row.get("delivery_pct") or 0) < 20:
        notes.append(f"low delivery % ({row.get('delivery_pct',0):.0f}%) — momentum largely intraday")
    return notes


def _links(sym: str) -> dict:
    return {
        "nse": f"https://www.nseindia.com/get-quotes/equity?symbol={sym}",
        "screener": f"https://www.screener.in/company/{sym}/",
        "tradingview": f"https://www.tradingview.com/symbols/NSE-{sym}/",
    }


def _format_pos(row: dict, side: str, sizing_pct: float, capital: int = 1_000_000) -> str:
    sym = row["symbol"]
    close = row.get("close", 0)
    score_cal = row.get("score_calibrated") or row.get("consensus") or 0
    sl_pct = -0.05 if side == "LONG" else 0.05
    sl_price = close * (1 + sl_pct)
    t1_price = close * (1 + (0.05 if side == "LONG" else -0.05))
    t2_price = close * (1 + (0.15 if side == "LONG" else -0.15))
    inr = int(sizing_pct * capital)
    shares = int(inr / close) if close > 0 else 0

    bbb = row.get("_bbb", {})
    risk = row.get("_risk_notes", [])
    L = _links(sym)
    fund = row.get("_fund", {})

    pe = fund.get("pe")
    spe = fund.get("sector_pe")
    pe_str = f"{pe:.1f}× vs sector {spe:.1f}× ({(pe/spe-1)*100:+.0f}% peer)" if pe and spe else "n/a"
    qoq_rev = fund.get("qoq_revenue_growth")
    qoq_pat = fund.get("qoq_pat_growth")
    growth_str = f"rev QoQ {qoq_rev:+.1f}%, PAT QoQ {qoq_pat:+.1f}%" if qoq_rev is not None and qoq_pat is not None else "no quarterlies in snapshot"
    w52h = fund.get("week52_high")
    dist52 = fund.get("dist_from_52w_high_pct")
    w52_str = f"₹{w52h:.0f} 52w-hi, {dist52:+.1f}% off" if w52h else "no 52w in snapshot"

    return f"""
========================================================================
{sym}  •  ₹{close:.2f}  •  {side} ({sizing_pct*100:.1f}% sized)  •  Conf {score_cal*100:.0f}%
========================================================================

THESIS:
  {row.get('_thesis','')}

VALUATION (vs peers):
  PE: {pe_str}
  Growth: {growth_str}
  52w: {w52_str}

TECHNICAL:
  ret20d {(row.get('return_20d') or 0)*100:+.1f}%  RSI_d {row.get('rsi_14_daily') or 0:.0f}  RSI_w {row.get('rsi_14_weekly') or 0:.0f}
  vs sma20 {(row.get('close',0)/(row.get('sma_20') or row.get('close',1))-1)*100:+.1f}%  vs sma50 {(row.get('close',0)/(row.get('sma_50') or row.get('close',1))-1)*100:+.1f}%  vs sma200 {(row.get('close',0)/(row.get('sma_200') or row.get('close',1))-1)*100:+.1f}%
  vol_vs_20d {row.get('volume_vs_20d') or 0:.1f}×  delivery {row.get('delivery_pct') or 0:.0f}%  ADV ₹{(row.get('avg_traded_value_20d') or 0)/1e7:.1f}cr/day

CATALYSTS (last 30d):
  total ann={row.get('ann_30d_count',0):.0f}  orders5d={row.get('ann_order_5d',0):.0f}  results5d={row.get('ann_result_5d',0):.0f}  capex30d={row.get('ann_capex_30d',0):.0f}
  insider_net60d=₹{(row.get('insider_net_60d_inr',0) or 0)/1e5:+.1f}L  block_buy5d=₹{(row.get('block_buy_5d_inr',0) or 0)/1e5:.1f}L

SENTIMENT (last 5d, finance lexicon):
  news     n={row.get('news_count_5d',0) or 0:.0f}  score={(row.get('news_sentiment_5d') or 0):+.2f}    reddit   n={row.get('reddit_mentions_5d',0) or 0:.0f}  score={(row.get('reddit_sentiment_5d') or 0):+.2f}
  youtube  n={row.get('youtube_mentions_5d',0) or 0:.0f}  score={(row.get('youtube_sentiment_5d') or 0):+.2f}

EXPECTED RETURN (OOS-grounded, score_cal={score_cal:.2f}):
  E[+5% high in 7d]:  {bbb.get('expected_high_pct',0):+.1f}%   (P fires: {row.get('_band_p_winner',0)*100:.0f}%)
  E[close 7d]:        {bbb.get('expected_close_pct',0):+.1f}%   (P close>0: {bbb.get('p_close_pos',0)*100:.0f}%)

OUTCOME PROBABILITIES (OOS, same score band):
  BULL (+15% swing hit):     {bbb.get('p_bull_15pct',0)*100:>5.1f}%
  BASE (+5% to T1, no swing):{bbb.get('p_base_5pct',0)*100:>5.1f}%
  BEAR (-5% low touched):    {bbb.get('p_bear_minus5',0)*100:>5.1f}%   ← stop-out risk

RISK ACKNOWLEDGED:
{chr(10).join('  - ' + r for r in risk) if risk else '  - none beyond standard equity beta'}

POSITION SIZING (@ ₹{capital//100000}L):
  alloc {sizing_pct*100:.1f}% = ₹{inr:,}  ({shares:,} shares)
  SL  ₹{sl_price:.2f} ({sl_pct*100:+.1f}%)
  T1  ₹{t1_price:.2f} (sell 25%, raise SL to entry)
  T2  ₹{t2_price:.2f} (sell 50%, trail rest 7d-low)

LINKS:
  {L['nse']}
  {L['screener']}
  {L['tradingview']}
"""


def main(top_n_long: int = 10, top_n_short: int = 5, capital: int = 1_000_000) -> None:
    today = pd.Timestamp(date.today())

    # load all sources
    long_df = pd.read_csv(LIVE_LONG) if LIVE_LONG.exists() else pd.DataFrame()
    short_df = pd.read_csv(LIVE_SHORT) if LIVE_SHORT.exists() else pd.DataFrame()
    mh_df = pd.read_csv(LIVE_MH) if LIVE_MH.exists() else pd.DataFrame()
    bands = pd.read_csv(BAND_STATS, index_col=0)

    # latest price snapshot
    px = pd.read_parquet(PRICES, columns=["symbol", "trade_date", "close", "sma_20", "sma_50",
                                           "sma_200", "rsi_14_daily", "rsi_14_weekly",
                                           "return_1d", "return_20d", "volume_vs_20d",
                                           "delivery_pct", "avg_traded_value_20d"])
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    latest = px["trade_date"].max()
    snap = px[px["trade_date"] == latest].set_index("symbol")

    # catalysts
    cat = pd.read_parquet(CATALYSTS) if CATALYSTS.exists() else pd.DataFrame()
    if len(cat):
        cat["trade_date"] = pd.to_datetime(cat["trade_date"])
        cat = cat[cat["trade_date"] == latest].set_index("symbol")

    # fundamentals (graceful if missing)
    fund = pd.read_parquet(FUNDAMENTALS) if FUNDAMENTALS.exists() else pd.DataFrame()
    if len(fund):
        fund["fetch_date"] = pd.to_datetime(fund["fetch_date"])
        fund = fund.sort_values("fetch_date").groupby("symbol").tail(1).set_index("symbol")

    # news/social sentiment per symbol
    news_feat = pd.read_parquet(NEWS_FEAT) if NEWS_FEAT.exists() else pd.DataFrame()
    if len(news_feat):
        news_feat["as_of"] = pd.to_datetime(news_feat["as_of"]).dt.date
        news_feat = news_feat.sort_values("as_of").groupby("symbol").tail(1).set_index("symbol")

    # macro sentiment (one row, latest)
    macro_row = None
    if MACRO_SENT.exists():
        ms = pd.read_parquet(MACRO_SENT)
        ms["as_of"] = pd.to_datetime(ms["as_of"]).dt.date
        macro_row = ms.sort_values("as_of").iloc[-1].to_dict()

    # join multi-horizon for triangulated flag
    mh_idx = mh_df.set_index("symbol") if len(mh_df) else pd.DataFrame()

    def enrich(sym: str, base: dict) -> dict:
        r = dict(base)
        if sym in snap.index:
            for c in snap.columns:
                if c not in r:
                    r[c] = snap.at[sym, c]
        if sym in cat.index if len(cat) else False:
            for c in cat.columns:
                r[c] = cat.at[sym, c]
        if sym in mh_idx.index if len(mh_idx) else False:
            r["triangulated"] = bool(mh_idx.at[sym, "triangulated"])
            r["consensus"] = float(mh_idx.at[sym, "consensus"])
        if sym in fund.index if len(fund) else False:
            r["_fund"] = {c: fund.at[sym, c] for c in fund.columns}
        else:
            r["_fund"] = {}
        if len(news_feat) and sym in news_feat.index:
            for c in ["news_count_5d", "news_sentiment_5d",
                      "reddit_mentions_5d", "reddit_sentiment_5d",
                      "youtube_mentions_5d", "youtube_sentiment_5d"]:
                if c in news_feat.columns:
                    r[c] = news_feat.at[sym, c]
        r["_thesis"] = _thesis(r)
        score_cal = r.get("score_calibrated") or r.get("consensus") or 0
        band = _band_for(score_cal, bands)
        r["_bbb"] = _bull_base_bear(score_cal, band)
        r["_band_p_winner"] = float(band["p_winner"])
        r["_risk_notes"] = _risk_notes(r)
        return r

    # build report
    lines = [f"# Pro Brief — {today:%Y-%m-%d}", ""]
    lines.append(f"_Generated by `generate_pro_brief.py` from v3 ensemble + multi-horizon + short-side + fundamentals + catalysts_")
    lines.append("")

    # macro top
    breadth = (snap["close"] > snap["sma_50"]).mean() if len(snap) else None
    breadth_200 = (snap["close"] > snap["sma_200"]).mean() if len(snap) else None
    lines.append("## Regime")
    lines.append(f"- Breadth (above 50dma): {breadth:.0%}" if breadth is not None else "")
    lines.append(f"- Breadth (above 200dma): {breadth_200:.0%}" if breadth_200 is not None else "")
    if len(long_df):
        top_cal = long_df["score_calibrated"].max()
        lines.append(f"- Top long score_cal today: {top_cal:.3f}")
        if top_cal < 0.80:
            lines.append("- **PATIENCE FILTER**: no 0.80+ trigger → recommended to stay in cash / LIQUIDPLUS")

    # sector heat-map (5d / 20d returns by sector)
    sect_members = ROOT / "tmp/from_scratch_7d_run/alt2/sector_index_members.parquet"
    if sect_members.exists():
        sm = pd.read_parquet(sect_members)
        SECT_PRIORITY = ["NIFTY IT","NIFTY BANK","NIFTY AUTO","NIFTY METAL","NIFTY PHARMA",
                         "NIFTY FMCG","NIFTY REALTY","NIFTY ENERGY","NIFTY MEDIA","NIFTY PSE",
                         "NIFTY PVT BANK","NIFTY FINANCIAL SERVICES","NIFTY CONSUMER DURABLES",
                         "NIFTY OIL & GAS","NIFTY INFRA"]
        sm["pri"] = sm["index_name"].map({n:i for i,n in enumerate(SECT_PRIORITY)}).fillna(99)
        sec_map = sm.sort_values("pri").drop_duplicates("symbol")[["symbol","index_name"]].rename(columns={"index_name":"sector"})
        px2 = pd.read_parquet(PRICES, columns=["symbol","trade_date","return_1d"])
        px2["trade_date"] = pd.to_datetime(px2["trade_date"])
        px2 = px2.merge(sec_map, on="symbol", how="left")
        px2 = px2[px2["sector"].isin(SECT_PRIORITY)]
        latest_px2 = px2["trade_date"].max()
        d = px2.groupby(["trade_date","sector"])["return_1d"].median().reset_index()
        d = d.sort_values(["sector","trade_date"])
        d["s_5d"] = d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(5).sum())
        d["s_20d"] = d.groupby("sector")["return_1d"].transform(lambda s: s.rolling(20).sum())
        latest_sec = d[d["trade_date"]==latest_px2].sort_values("s_5d", ascending=False)
        if len(latest_sec):
            lines.append("")
            lines.append("### Sector heat-map (5d / 20d, ranked)")
            lines.append("")
            lines.append("| Sector | 5d % | 20d % | Read |")
            lines.append("|---|---:|---:|---|")
            for _, r in latest_sec.head(15).iterrows():
                read = "🟢 leader" if r["s_5d"] > 0.02 else ("🔴 lagger" if r["s_5d"] < -0.02 else "◯ flat")
                lines.append(f"| {r['sector']} | {r['s_5d']*100:+.2f}% | {r['s_20d']*100:+.2f}% | {read} |")

    # macro sentiment line
    if macro_row:
        gms = macro_row.get("global_macro_sent", 0) or 0
        dms = macro_row.get("domestic_macro_sent", 0) or 0
        rh = int(macro_row.get("rate_hawkish_score", 0) or 0)
        rd = int(macro_row.get("rate_dovish_score", 0) or 0)
        oil = macro_row.get("oil_sentiment", 0) or 0
        inr = macro_row.get("usdinr_sentiment", 0) or 0
        lines.append("")
        lines.append("### Macro sentiment (last 5d, finance lexicon over RSS+Reddit+YT)")
        lines.append(f"- **Global:** {gms:+.2f}  (hawkish/dovish rates: {rh}/{rd})")
        lines.append(f"- **Domestic India:** {dms:+.2f}")
        lines.append(f"- **Oil:** {oil:+.2f} (>0 = falling oil = +ve for India)")
        lines.append(f"- **USDINR:** {inr:+.2f} (>0 = INR strong)")
        macro_overall = (gms + dms) / 2
        if macro_overall < -0.3:
            lines.append("- ⚠️ **RISK-OFF macro** — size down 50%, prefer defensives / cash")
        elif macro_overall > 0.3:
            lines.append("- ✓ **RISK-ON macro** — model picks favored, size up to plan")
        else:
            lines.append("- **NEUTRAL macro** — execute model picks at planned size")
    lines.append("")

    # ════════════════════════════════════════════════════════════════
    # HARD RULE: respect the filter cascade. If it returned 0 names,
    # NO LONG TRADES are surfaced regardless of user pressure.
    # This rule was added after a real failure on 2026-04-28 where
    # the model said "no trade" but a Path B microcap stack was
    # surfaced anyway → user lost money the next day.
    # ════════════════════════════════════════════════════════════════
    cascade_csv = ROOT / "tmp/from_scratch_7d_run/actionable_today.csv"
    cascade_n = 0
    if cascade_csv.exists():
        try:
            cascade_n = len(pd.read_csv(cascade_csv))
        except Exception:
            cascade_n = 0
    macro_overall = 0.0
    if macro_row:
        macro_overall = ((macro_row.get("global_macro_sent", 0) or 0) + (macro_row.get("domestic_macro_sent", 0) or 0)) / 2
    floor = 0.75 if macro_overall <= -0.3 else 0.65
    if cascade_n == 0:
        actionable_long = pd.DataFrame()  # FORCE empty — don't surface alternatives
    else:
        actionable_long = long_df[long_df["score_calibrated"] >= floor] if len(long_df) else pd.DataFrame()

    if len(actionable_long) == 0:
        lines.append("## Long Picks")
        lines.append("")
        lines.append("# 🛑 NO TRADE TODAY")
        lines.append("")
        lines.append(f"The filter cascade returned **{cascade_n} actionable names**.")
        lines.append("")
        lines.append(f"- Top long score_cal today: {long_df['score_calibrated'].max():.3f}" if len(long_df) else "")
        lines.append(f"- Patience floor: {floor:.2f} ({'RISK-OFF tightened' if floor==0.75 else 'NEUTRAL'})")
        lines.append(f"- Macro: {macro_overall:+.2f}")
        lines.append("")
        lines.append("**Action: DO NOT TRADE LONGS TODAY. Park in LIQUIDPLUS / CASHIETF.**")
        lines.append("")
        lines.append("> _Hard rule encoded after 2026-04-28 forced-trade failure. The brief will not surface 'Path B' alternatives, microcap baskets, or 'if you really want to trade…' caveats when the cascade says zero. The next high-conviction signal will appear in tomorrow's brief if it shows up._")
        lines.append("")
    else:
        long_top = actionable_long.sort_values("score_calibrated", ascending=False).head(top_n_long)
        lines.append(f"## Long Picks (top {len(long_top)} clearing floor {floor:.2f})")
        lines.append("")
        for _, r in long_top.iterrows():
            row = enrich(r["symbol"], r.to_dict())
            lines.append(_format_pos(row, "LONG", sizing_pct=0.08, capital=capital))

    # SHORT picks
    if len(short_df):
        short_top = short_df.sort_values("score_calibrated", ascending=False).head(top_n_short)
        lines.append(f"\n## Short Picks — ML model (top {len(short_top)})")
        lines.append("")
        for _, r in short_top.iterrows():
            row = enrich(r["symbol"], r.to_dict())
            lines.append(_format_pos(row, "SHORT", sizing_pct=0.05, capital=capital))

    # Sector-weak large-cap short overlay (catches what ML misses)
    sws_path = ROOT / "tmp/from_scratch_7d_run/sector_weak_shorts.csv"
    if sws_path.exists():
        sws = pd.read_csv(sws_path)
        if len(sws):
            lines.append(f"\n## Short Picks — Macro overlay (sector-weak large-caps, futures)")
            lines.append("")
            lines.append("_When ML short model misses large-caps, this overlay catches them via 5d sector weakness × technical extension._")
            lines.append("")
            lines.append("| Symbol | Sector | Close | Sector 5d | 20d% | RSI | ADV cr/d | Reason |")
            lines.append("|---|---|---:|---:|---:|---:|---:|---|")
            for _, r in sws.head(8).iterrows():
                lines.append(f"| **{r['symbol']}** | {r['sector']} | ₹{r['close']:.2f} | {r['sector_5d_ret_pct']:+.1f}% | "
                             f"{r.get('return_20d_pct',0):+.1f}% | {r.get('rsi_14_daily','-')} | "
                             f"{r['adv_cr']:.0f} | {r['reason']} |")

    # band stats footer
    lines.append("\n## Score-band cheat-sheet (OOS 2024-2025)")
    lines.append("")
    lines.append("| score_cal | n | P(+5% high) | P(-5% low) | E[high] | E[close] | P(close>0) |")
    lines.append("|---|---|---|---|---|---|---|")
    for idx, b in bands.iterrows():
        lines.append(f"| {idx} | {int(b['n']):,} | {b['p_winner']*100:.0f}% | {b['p_min_minus5']*100:.0f}% | {b['mean_high']*100:+.1f}% | {b['mean_close']*100:+.1f}% | {b['p_close_pos']*100:.0f}% |")

    out = OUT_DIR / f"daily_pro_brief_{today:%Y%m%d}.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")
    print(f"  long_top={top_n_long}  short_top={top_n_short}  capital=₹{capital//100000}L")
    print(f"  fundamentals coverage: {len(fund) if len(fund) else 0} symbols")


if __name__ == "__main__":
    main()
