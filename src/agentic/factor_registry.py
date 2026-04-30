"""Factor / data hypothesis registry — the "what to try next" catalog.

Each entry is a hypothesis: a feature or data source that *might* improve
the 7d-forward prediction model. Categorised by data source so we know
what we already have vs what we need to fetch.

The agent loop reads this registry, picks unmet hypotheses, fetches data
or compiles features, runs feature_factory + retrain, measures OOS lift,
and writes verdicts back here.

States:
  PROPOSED   — registered, no work yet
  FETCHING   — data being pulled
  COMPILED   — feature built and added to feature parquet
  EVALUATED  — OOS lift measured (lift_ic / lift_top5_precision recorded)
  KEPT       — kept in production model (lift > threshold)
  DROPPED    — measured but did not lift; archived
  BLOCKED    — needs external resource (paid API, IP-unblocked host, etc.)

Inspired by:
  - WorldQuant 101 Alphas (Kakushadze, 2015)
  - Quantitative Trading by Ernie Chan
  - Active Portfolio Management (Grinold & Kahn) — IR decomposition
  - India-specific: Capitaline, FII/DII flow data, NSE PIT filings
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import pandas as pd

ROOT = Path("/Users/abhinavs./Documents/Zoom")
REGISTRY_PATH = ROOT / "data/derived/factor_registry.json"


class State(str, Enum):
    PROPOSED = "PROPOSED"
    FETCHING = "FETCHING"
    COMPILED = "COMPILED"
    EVALUATED = "EVALUATED"
    KEPT = "KEPT"
    DROPPED = "DROPPED"
    BLOCKED = "BLOCKED"


@dataclass
class Hypothesis:
    id: str
    name: str
    category: str
    description: str
    formula: str  # plain-English or pandas-expr
    data_needed: list[str]
    has_data: bool
    state: str = State.PROPOSED.value
    lift_ic: float | None = None
    lift_top5_precision: float | None = None
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────
# REGISTRY — 35+ hypotheses across 8 categories
# ─────────────────────────────────────────────────────────────────────────
HYPOTHESES: list[Hypothesis] = [
    # ════════════ 1. WORLDQUANT 101 ALPHAS (price/volume only) ════════════
    Hypothesis("wq_a1", "Alpha-1: rank residual reversion",
               "wq101",
               "Cross-sectional rank of (close - open)/open over 5d, inverted = alpha-1 mean revert",
               "rank(ts_argmax(SignedPower((returns < 0) ? stddev(returns,20) : close, 2), 5)) - 0.5",
               ["close", "open", "return_1d", "realized_vol_20d"], True),

    Hypothesis("wq_a3", "Alpha-3: -correlation(rank(open), rank(volume), 10)",
               "wq101", "Open-volume disagreement, mean-reverting",
               "-1 * correlation(rank(open), rank(volume), 10)",
               ["open", "volume"], True),

    Hypothesis("wq_a6", "Alpha-6: -correlation(open, volume, 10)",
               "wq101", "Raw open-volume correlation reversal",
               "-1 * correlation(open, volume, 10)",
               ["open", "volume"], True),

    Hypothesis("wq_a12", "Alpha-12: sign(delta(volume, 1)) * (-1 * delta(close, 1))",
               "wq101", "Volume-confirmed mean-reverter",
               "sign(volume - volume.shift(1)) * -1 * (close - close.shift(1))",
               ["close", "volume"], True),

    Hypothesis("wq_a23", "Alpha-23: high-extension reversal",
               "wq101", "If high > 20d high mean by 2σ, expect mean revert",
               "(sum(high,20)/20 < high) ? -1*(high - high.shift(2)) : 0",
               ["high"], True),

    Hypothesis("wq_a41", "Alpha-41: ((high*low)^0.5 - vwap) / vwap",
               "wq101", "Geometric-mean midprice vs vwap divergence",
               "(high*low)**0.5 / vwap - 1",
               ["high", "low", "vwap"], False, notes="need vwap proxy = total_value/volume"),

    Hypothesis("wq_a54", "Alpha-54: (-1 * (low - close) * open^5) / ((low - high) * close^5)",
               "wq101", "Microstructure imbalance",
               "-1*(low-close)*open**5 / ((low-high)*close**5)",
               ["open", "high", "low", "close"], True),

    Hypothesis("wq_a101", "Alpha-101: ((close - open) / ((high - low) + 0.001))",
               "wq101", "Intraday return normalized by range",
               "(close - open) / (high - low + 0.001)",
               ["open", "high", "low", "close"], True),

    # ════════════ 2. VOLATILITY REGIME ════════════
    Hypothesis("vol_regime", "Realized-vol z-score (60d)",
               "volatility", "Stock-specific vol shock detector",
               "(rv_20d - mean(rv_20d, 60)) / std(rv_20d, 60)",
               ["realized_vol_20d"], True),

    Hypothesis("vol_term", "Vol term-structure: 20d/60d ratio",
               "volatility", "If 20d vol >> 60d vol, regime breaking",
               "rv_20d / rv_60d",
               ["realized_vol_20d"], True, notes="need rv_60d compute"),

    Hypothesis("vol_of_vol", "Vol-of-vol (60d std of 20d vol)",
               "volatility", "Stocks with chaotic vol get penalized in stable regime",
               "std(rv_20d, 60)",
               ["realized_vol_20d"], True),

    # ════════════ 3. MICROSTRUCTURE / LIQUIDITY ════════════
    Hypothesis("amihud", "Amihud illiquidity ratio (20d)",
               "microstructure", "|return| / dollar_volume averaged over 20d",
               "mean(|return_1d| / (close*volume), 20)",
               ["return_1d", "close", "volume"], True),

    Hypothesis("kyle_lambda", "Kyle's lambda (proxy)",
               "microstructure", "Price impact per unit volume",
               "regression slope of |return_1d| on volume_change over 20d",
               ["return_1d", "volume"], True),

    Hypothesis("turnover_skew", "20d turnover skewness",
               "microstructure", "Skewed turnover = informed trading",
               "skew(volume/avg_vol_20d, 20)",
               ["volume", "avg_vol_20d"], True),

    # ════════════ 4. CROSS-SECTIONAL / RELATIVE STRENGTH ════════════
    Hypothesis("rs_sector", "Relative-strength rank within sector (20d)",
               "cross_sectional", "Stock's 20d return percentile within sector",
               "rank(return_20d) over sector group",
               ["return_20d", "sector"], True),

    Hypothesis("residual_momentum", "Sector-residual momentum (20d)",
               "cross_sectional", "Return - sector_return — orthogonalized momentum",
               "return_20d - sector_20d_ret",
               ["return_20d", "sector_20d_ret"], True),

    Hypothesis("breadth_thrust", "Sector breadth thrust",
               "cross_sectional", "% of sector members above 20-DMA",
               "mean(close > sma_20) over sector group",
               ["close", "sma_20", "sector"], True),

    # ════════════ 5. CALENDAR / SEASONALITY ════════════
    Hypothesis("month_end", "Month-end effect dummy",
               "calendar", "Last 3 trading days of month often see retail flow",
               "trade_date.day >= last_3_business_days_of_month",
               ["trade_date"], True),

    Hypothesis("expiry_week", "Monthly F&O expiry week dummy",
               "calendar", "Last week of month sees position-squaring",
               "is_last_thursday_or_within_4d_of_it",
               ["trade_date"], True),

    Hypothesis("results_week", "Earnings-results-week proximity",
               "calendar", "Days until next quarterly result announcement",
               "min(announcement_date - today) within 30d",
               ["ann_30d_count"], True),

    # ════════════ 6. INDIA-SPECIFIC FLOW + MACRO (NEW DATA NEEDED) ════════════
    Hypothesis("fii_dii_flow_5d", "FII+DII net flow 5d cumulative",
               "macro_flow", "Daily net buy-sell from NSE FIIDII bulletin",
               "sum(fii_net + dii_net) over 5d",
               ["fii_net_inr", "dii_net_inr"], False,
               notes="need fetch_fii_dii.py — pulls https://www.nseindia.com/api/fiidiiTradeReact"),

    Hypothesis("usdinr_5d", "USDINR change 5d",
               "macro_flow", "Currency pressure on IT/Pharma exporters and importers",
               "(usdinr_today - usdinr_5d_ago) / usdinr_5d_ago",
               ["usdinr_close"], False,
               notes="need fetch_forex.py — RBI reference rates or yfinance INR=X"),

    Hypothesis("brent_5d", "Brent crude 5d return",
               "macro_flow", "Oil price = direct India CPI driver, sectorally OMC/Aviation/Refineries",
               "(brent_today - brent_5d_ago) / brent_5d_ago",
               ["brent_close"], False, notes="yfinance BZ=F or RBI commodity"),

    Hypothesis("gsec_10y", "10y G-sec yield change 5d",
               "macro_flow", "Risk-free rate + bank NIMs",
               "(yield_10y - yield_10y_5d_ago)",
               ["india_10y_yield"], False, notes="RBI weekly bulletin or CCIL"),

    Hypothesis("india_vix", "India VIX level + 5d change",
               "macro_flow", "Risk-off detector",
               "vix_today + (vix - vix_5d_ago) / vix",
               ["india_vix"], False, notes="NSE VIX endpoint, need fetch_vix.py"),

    Hypothesis("nifty_pe_z", "NIFTY 50 PE z-score (1y)",
               "macro_flow", "Index-level valuation regime",
               "(nifty_pe - mean(nifty_pe, 252)) / std(nifty_pe, 252)",
               ["nifty_pe"], False, notes="NSE indices PE history"),

    # ════════════ 7. ALT DATA / TEXT ════════════
    Hypothesis("google_trends", "Google Trends per-symbol search volume",
               "alt_text", "Retail interest leading indicator",
               "google_trends_score(symbol, last 7d)",
               ["google_trends_7d"], False,
               notes="need fetch_google_trends.py — pytrends library, but throttles hard"),

    Hypothesis("wiki_pageviews", "Wikipedia daily page-view ratio",
               "alt_text", "Retail attention proxy (works for famous names)",
               "wiki_views_today / mean(wiki_views, 30)",
               ["wiki_pageviews_7d"], False,
               notes="need fetch_wiki.py — Wikimedia REST API, free, fast"),

    Hypothesis("news_velocity", "News count acceleration",
               "alt_text", "News count rising = breakout candidate",
               "news_count_3d / news_count_30d_avg",
               ["news_count_5d"], True),

    Hypothesis("sentiment_dispersion", "News sentiment dispersion (std)",
               "alt_text", "Disagreement = volatility ahead",
               "std(news_sentiment_per_article) over 5d",
               ["news_sentiment_5d"], True, notes="need per-article sentiment retained"),

    # ════════════ 8. CROSS-ASSET / DERIVATIVES ════════════
    Hypothesis("pcr_change", "PCR (put-call ratio) 5d change",
               "derivatives", "Sentiment from options market",
               "(pcr_oi_today - pcr_oi_5d_ago)",
               ["pcr_oi"], False,
               notes="BLOCKED — NSE option-chain endpoint blocks our IP"),

    Hypothesis("iv_skew_change", "Vol skew steepening",
               "derivatives", "Put IV rising vs call IV = bearish positioning",
               "iv_25d_put - iv_25d_call",
               ["iv_skew"], False, notes="BLOCKED — needs option-chain"),

    Hypothesis("max_pain_dist", "Distance to max-pain strike",
               "derivatives", "Spot tends to gravitate toward max-pain at expiry",
               "(close - max_pain) / close",
               ["max_pain"], False, notes="BLOCKED — needs option-chain"),

    Hypothesis("oi_buildup", "Futures OI build-up direction",
               "derivatives", "Long buildup vs short buildup",
               "delta_OI * sign(price_change)",
               ["futures_oi"], False, notes="need fetch_futures_oi.py — NSE F&O OI history"),

    # ════════════ 9. CORPORATE / FUNDAMENTAL ════════════
    Hypothesis("earnings_surprise", "PAT QoQ vs consensus",
               "fundamental", "Beat/miss vs analyst consensus",
               "actual_pat / consensus_pat - 1",
               ["consensus_pat"], False, notes="need analyst consensus — Trendlyne / Bloomberg"),

    Hypothesis("promoter_pledge", "Promoter pledge change 60d",
               "fundamental", "Pledge release = promoters confident; pledge increase = bad",
               "pledged_pct - pledged_pct_60d_ago",
               ["promoter_pledge_pct"], False,
               notes="need fetch_pledge.py — NSE promoter shareholding pattern"),

    Hypothesis("promoter_holding_chg", "Promoter holding QoQ change",
               "fundamental", "Promoter buying = strong signal",
               "promoter_holding - promoter_holding_q-1",
               ["promoter_holding_pct"], False, notes="quarterly NSE shareholding pattern"),

    Hypothesis("rating_change", "Credit-rating action 90d",
               "fundamental", "Upgrade/downgrade = catalyst",
               "rating_action_dummy",
               ["credit_rating_action"], False, notes="CRISIL/ICRA/CARE feeds"),
]


def save_registry() -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump([asdict(h) for h in HYPOTHESES], f, indent=2)


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        save_registry()
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def summary() -> None:
    rows = load_registry()
    df = pd.DataFrame(rows)
    print(f"Total hypotheses: {len(df)}")
    print(f"By state:")
    print(df["state"].value_counts().to_string())
    print(f"\nBy category:")
    print(df["category"].value_counts().to_string())
    print(f"\nData availability:")
    print(f"  has data already       : {df['has_data'].sum()}")
    print(f"  needs new fetcher      : {(~df['has_data']).sum()}")
    blocked = df[df["notes"].astype(str).str.contains("BLOCKED", na=False)]
    print(f"  IP-blocked (option chain): {len(blocked)}")


if __name__ == "__main__":
    save_registry()
    summary()
