"""LLM-driven hypothesis agent — generates STRONG factor hypotheses
from market microstructure + behavioral finance theory + recent model evidence.

Unlike the static factor_registry which we hand-seeded with WQ-101 alphas,
this agent:

  1. Reads recent OOF predictions + actual outcomes
  2. Identifies systematic *failure modes* (where the model is wrong consistently)
  3. Reads recent news + earnings transcripts (when available)
  4. Frames a hypothesis explaining the gap, drawing from:
       - Behavioral finance (anchoring, loss aversion, herding, overreaction)
       - Market microstructure (order flow imbalance, liquidity premium)
       - Network effects (sector spillover, supply-chain co-movement)
       - Calendar / event windows (earnings drift, ex-dividend, FOMC)
       - Macro-conditional regimes (rate hike cycle, INR weakness, oil shock)
  5. Encodes it as a registry entry with computable formula

This file generates hypotheses programmatically (via templated heuristics);
the *actual* LLM call is intended as a follow-on integration with Claude API.

For now, this seeds the registry with 30+ strong, theory-driven hypotheses
beyond the WorldQuant-101 set.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
REGISTRY = ROOT / "data/derived/factor_registry.json"

NEW_HYPOTHESES = [
    # ════════════ BEHAVIORAL FINANCE ════════════
    {"id": "anchor_52w_high", "name": "52w-high anchoring",
     "category": "behavioral",
     "description": "Stocks within 5% of 52w-high overreact on bad news (anchoring + loss aversion)",
     "formula": "(close >= 0.95*week52_high) * news_sentiment_5d_negative",
     "data_needed": ["week52_high", "news_sentiment_5d"], "has_data": True},

    {"id": "post_earnings_drift", "name": "Post-earnings announcement drift (PEAD)",
     "category": "behavioral",
     "description": "Stocks beating estimates drift up for 60-90 days (slow-information processing)",
     "formula": "(qoq_pat_growth > 25) * days_since_results <= 60",
     "data_needed": ["qoq_pat_growth", "ann_result_5d"], "has_data": True},

    {"id": "round_number_bias", "name": "Round-number resistance",
     "category": "behavioral",
     "description": "Stocks pause at psychological levels (₹100, ₹500, ₹1000); breakout = momentum",
     "formula": "abs(close - round(close, -2)) / close < 0.01 * 5d_above_level",
     "data_needed": ["close"], "has_data": True},

    {"id": "herding_volume_spike", "name": "Retail herding via volume spike",
     "category": "behavioral",
     "description": "Volume > 5× ADV with low delivery% = retail crowd trade, mean reverts",
     "formula": "(volume_vs_20d > 5) * (delivery_pct < 25) * -1",
     "data_needed": ["volume_vs_20d", "delivery_pct"], "has_data": True},

    {"id": "overnight_overreaction", "name": "Overnight gap mean reversion",
     "category": "behavioral",
     "description": "Open >> previous close = retail FOMO; reverts intraday",
     "formula": "(open / prev_close - 1) > 0.05",
     "data_needed": ["open", "prev_close"], "has_data": True},

    # ════════════ MICROSTRUCTURE ════════════
    {"id": "high_low_range_pct", "name": "Intraday range expansion",
     "category": "microstructure",
     "description": "(high - low) / close >> 20d avg = volatility regime shift",
     "formula": "(high-low)/close vs rolling(20).mean()",
     "data_needed": ["high", "low", "close"], "has_data": True},

    {"id": "delivery_pct_z", "name": "Delivery % anomaly z-score",
     "category": "microstructure",
     "description": "High delivery% spike = institutional accumulation (vs intraday churn)",
     "formula": "(delivery_pct - rolling_mean_60d) / rolling_std_60d",
     "data_needed": ["delivery_pct"], "has_data": True},

    {"id": "vwap_deviation", "name": "Close vs VWAP residual",
     "category": "microstructure",
     "description": "Closing far from VWAP = end-of-day directional pressure",
     "formula": "(close - vwap) / vwap",
     "data_needed": ["close", "total_traded_value", "total_traded_qty"], "has_data": True},

    # ════════════ NETWORK / SECTOR ════════════
    {"id": "sector_dispersion", "name": "Within-sector return dispersion",
     "category": "network",
     "description": "When sector dispersion is low, leaders mean-revert; when high, momentum sustains",
     "formula": "std(member returns_5d) within sector",
     "data_needed": ["sector", "return_1d"], "has_data": True},

    {"id": "sector_leader_lag", "name": "Sector leader-lagger spread",
     "category": "network",
     "description": "Top sector leaders' moves predict laggers within 3-5 days",
     "formula": "shift(top_quartile_in_sector_5d_ret, 3)",
     "data_needed": ["sector", "return_5d"], "has_data": True},

    {"id": "index_inclusion_anticipation", "name": "Index-inclusion candidate run-up",
     "category": "network",
     "description": "Stocks ranked just outside NIFTY 50/500 cutoff often rally pre-rebalance",
     "formula": "rank by mcap, within rank 50-70 of NIFTY 500 universe",
     "data_needed": ["mcap_rank"], "has_data": False},

    # ════════════ CALENDAR / EVENT ════════════
    {"id": "monthly_expiry_pin", "name": "Expiry-day pin to max-pain",
     "category": "calendar",
     "description": "On monthly expiry, spot gravitates to max-pain strike",
     "formula": "is_expiry_day * sign(close - max_pain)",
     "data_needed": ["max_pain", "trade_date"], "has_data": False, "notes": "needs option chain"},

    {"id": "post_results_revision", "name": "Analyst revision post-results window",
     "category": "calendar",
     "description": "30 days after Q-results, analyst consensus drifts toward actuals",
     "formula": "consensus_drift_post_results",
     "data_needed": ["consensus_pat", "ann_result_5d"], "has_data": False, "notes": "paid consensus data"},

    {"id": "fy_end_window_dressing", "name": "March-end window dressing",
     "category": "calendar",
     "description": "Mid/small caps held by funds rally last week of March (window dressing)",
     "formula": "(month==3) * (date_within_last_5_business_days)",
     "data_needed": ["trade_date"], "has_data": True},

    {"id": "monsoon_agri_seasonal", "name": "Monsoon seasonality (June-Sep)",
     "category": "calendar",
     "description": "Agri-rural names rally pre-monsoon, fade if rainfall disappoints",
     "formula": "rolling(monsoon_progress) * is_agri_sector",
     "data_needed": ["sector", "monsoon_data"], "has_data": False, "notes": "IMD daily monsoon API"},

    # ════════════ MACRO-CONDITIONAL ════════════
    {"id": "usdinr_it_pharma_lift", "name": "Weak INR → IT/Pharma lift (5d)",
     "category": "macro_conditional",
     "description": "USDINR rising 1% in 5d → IT/Pharma names with USD revenue rally",
     "formula": "usdinr_5d_chg * (sector in ['NIFTY IT', 'NIFTY PHARMA'])",
     "data_needed": ["usdinr", "sector"], "has_data": True},

    {"id": "oil_omc_inverse", "name": "Oil rising → OMC margin compression",
     "category": "macro_conditional",
     "description": "Brent up 5%+ → BPCL/HPCL/IOC margin pressure",
     "formula": "brent_5d_chg * is_omc_name",
     "data_needed": ["brent_close", "sector"], "has_data": False, "notes": "needs Brent fetcher"},

    {"id": "yield_rise_bank_lift", "name": "10y yield rise → Bank NIM lift",
     "category": "macro_conditional",
     "description": "Yield rising → bank NIMs improve; banks rally",
     "formula": "yield_5d_chg * is_bank_name",
     "data_needed": ["india_10y_yield", "sector"], "has_data": False, "notes": "RBI bulletin"},

    {"id": "vix_spike_defensive_lift", "name": "VIX spike → defensive sector lift",
     "category": "macro_conditional",
     "description": "INDIA VIX up 20%+ in 3d → FMCG/Pharma outperform; high-beta lag",
     "formula": "vix_3d_chg * sector_beta_proxy",
     "data_needed": ["india_vix", "sector"], "has_data": False},

    # ════════════ FUNDAMENTAL QUALITY (broader than v3) ════════════
    {"id": "earnings_acceleration", "name": "QoQ growth acceleration",
     "category": "fundamental",
     "description": "QoQ growth this Q > QoQ growth last Q = acceleration; predicts 30-day strength",
     "formula": "qoq_pat_growth - qoq_pat_growth_prev_q",
     "data_needed": ["qoq_pat_growth"], "has_data": True},

    {"id": "margin_expansion", "name": "Operating margin expansion",
     "category": "fundamental",
     "description": "PAT growth > Revenue growth = margin lift, predicts re-rating",
     "formula": "qoq_pat_growth - qoq_revenue_growth",
     "data_needed": ["qoq_pat_growth", "qoq_revenue_growth"], "has_data": True},

    {"id": "fast_pe_compression", "name": "Fast PE compression (3 quarters)",
     "category": "fundamental",
     "description": "Falling PE despite stable earnings = price-driven discount, mean revert",
     "formula": "(pe - pe_3q_ago) / pe_3q_ago",
     "data_needed": ["pe"], "has_data": True},

    {"id": "low_pe_high_growth", "name": "Low PE × High growth (PEG proxy)",
     "category": "fundamental",
     "description": "PE/sector_PE < 0.7 AND QoQ growth > 20% = undervalued grower",
     "formula": "(pe/sector_pe < 0.7) * (qoq_pat_growth > 20)",
     "data_needed": ["pe", "sector_pe", "qoq_pat_growth"], "has_data": True},

    # ════════════ PROMOTER / INSIDER (HIGH VALUE) ════════════
    {"id": "promoter_pledge_release", "name": "Promoter pledge release",
     "category": "ownership",
     "description": "Pledge % falling 5%+ QoQ = promoter regaining control; strong positive",
     "formula": "promoter_pledge_pct - promoter_pledge_pct_prev_q",
     "data_needed": ["promoter_pledge_pct"], "has_data": False,
     "notes": "needs fetch_promoter_pledge.py — NSE shareholding pattern"},

    {"id": "promoter_buyback_announce", "name": "Buyback announcement window",
     "category": "ownership",
     "description": "Days within 60 of buyback announcement = buying pressure",
     "formula": "days_since_buyback < 60",
     "data_needed": ["ann_buyback_30d"], "has_data": True},

    {"id": "sast_5pct_acquisition", "name": "SAST 5%+ acquisition disclosure",
     "category": "ownership",
     "description": "5%+ stake change reported via SAST = institutional accumulation",
     "formula": "sast_disclosure_30d_count",
     "data_needed": ["sast_disclosures"], "has_data": False,
     "notes": "needs fetch_sast.py — NSE corporate-actions"},

    {"id": "mf_inclusion_velocity", "name": "MF holdings increase QoQ",
     "category": "ownership",
     "description": "Top 10 MFs adding the same name = institutional conviction",
     "formula": "n_mfs_holding_qoq_change",
     "data_needed": ["mf_holdings"], "has_data": False,
     "notes": "needs fetch_amfi_mf_portfolios.py — AMFI monthly"},

    # ════════════ ALT DATA / NEW SIGNALS ════════════
    {"id": "fno_ban_squeeze", "name": "F&O ban list squeeze",
     "category": "alt_market",
     "description": "Stocks in F&O ban (95% MWPL hit) often see short-squeeze rallies",
     "formula": "is_in_fno_ban_list",
     "data_needed": ["fno_ban_list"], "has_data": False,
     "notes": "needs fetch_fno_ban.py — NSE daily bulletin"},

    {"id": "circuit_to_circuit", "name": "Upper-circuit to upper-circuit streak",
     "category": "alt_market",
     "description": "3+ consecutive UC days = retail momentum; reverts after streak ends",
     "formula": "consecutive_UC_days",
     "data_needed": ["high", "prev_close"], "has_data": True},

    {"id": "auditor_resignation_flag", "name": "Auditor resignation/qualified opinion",
     "category": "alt_market",
     "description": "Auditor exits or qualifies = quality red flag, leads -10% moves",
     "formula": "auditor_change_60d",
     "data_needed": ["auditor_changes"], "has_data": False,
     "notes": "needs fetch_corporate_governance.py — BSE/NSE"},

    {"id": "credit_rating_action", "name": "CRISIL/ICRA/CARE rating action",
     "category": "alt_market",
     "description": "Upgrade = +ve catalyst, downgrade = -ve catalyst, 30-day window",
     "formula": "rating_change_30d",
     "data_needed": ["credit_rating_action"], "has_data": False,
     "notes": "CRISIL/ICRA RSS feeds"},

    {"id": "naukri_hiring_surge", "name": "Naukri job-posting surge",
     "category": "alt_market",
     "description": "30%+ jump in job postings (Naukri) = expansion mode",
     "formula": "naukri_postings_30d_z",
     "data_needed": ["naukri_postings"], "has_data": False,
     "notes": "Naukri scrape per company"},

    {"id": "app_store_rank", "name": "App Store / Play Store rank",
     "category": "alt_market",
     "description": "For D2C/fintech: app rank improvement → revenue growth proxy",
     "formula": "app_rank_30d_change",
     "data_needed": ["app_store_rank"], "has_data": False,
     "notes": "App Annie / Sensor Tower (paid) or scrape"},

    {"id": "google_trends_z", "name": "Google Trends search z-score",
     "category": "alt_market",
     "description": "Search volume z-score per company name (retail attention)",
     "formula": "(search_today - mean_30d) / std_30d",
     "data_needed": ["google_trends_7d"], "has_data": False,
     "notes": "pytrends — heavy throttle"},

    # ════════════ INTERACTION / COMBINATORIAL ════════════
    {"id": "rsi_x_volume", "name": "RSI × Volume confirmation",
     "category": "interaction",
     "description": "RSI > 70 with volume > 2× ADV = momentum confirmed (not exhaustion)",
     "formula": "(rsi_14_daily > 70) * (volume_vs_20d > 2)",
     "data_needed": ["rsi_14_daily", "volume_vs_20d"], "has_data": True},

    {"id": "low_vol_breakout", "name": "Low-vol regime breakout",
     "category": "interaction",
     "description": "Stock breaking 20-day high after compressed-vol regime = high R:R",
     "formula": "(close > rolling_max_20) * (realized_vol_20d < rolling_quantile(0.3))",
     "data_needed": ["close", "realized_vol_20d"], "has_data": True},

    {"id": "fii_dii_divergence", "name": "FII selling × DII buying divergence",
     "category": "interaction",
     "description": "When FII sells but DII buys, stocks DII favors outperform 7d",
     "formula": "(fii_net_5d < 0) * (dii_net_5d > 0)",
     "data_needed": ["fii_net_inr", "dii_net_inr"], "has_data": False},
]


def main() -> None:
    print(f"== hypothesis_agent: adding {len(NEW_HYPOTHESES)} theory-driven hypotheses ==")
    if not REGISTRY.exists():
        print(f"  registry not found at {REGISTRY} — initialize via factor_registry.py first")
        return
    reg = json.loads(REGISTRY.read_text())
    existing_ids = {h["id"] for h in reg}
    added = 0
    for h in NEW_HYPOTHESES:
        if h["id"] in existing_ids:
            continue
        # default fields
        h.setdefault("state", "PROPOSED")
        h.setdefault("lift_ic", None)
        h.setdefault("lift_top5_precision", None)
        h.setdefault("notes", "")
        reg.append(h)
        added += 1
    REGISTRY.write_text(json.dumps(reg, indent=2))
    print(f"  added {added} new hypotheses (skipped {len(NEW_HYPOTHESES)-added} dupes)")
    print(f"  total hypotheses now: {len(reg)}")
    by_cat = {}
    for h in reg:
        by_cat[h["category"]] = by_cat.get(h["category"], 0) + 1
    print("\nBy category:")
    for c, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {c:<22} {n:>3}")


if __name__ == "__main__":
    main()
