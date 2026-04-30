"""Plain-English description for every feature the model uses.

Used by build_dashboard.py for hover tooltips. Layman-friendly language only —
NO jargon. If you can't explain a feature to a smart non-quant in 1-2 sentences,
the description is wrong.
"""
from __future__ import annotations

DESCRIPTIONS: dict[str, str] = {
    # ━━━━━━━━━━━━━━━━ RAW PRICE / TECHNICAL ━━━━━━━━━━━━━━━━
    "close": "Today's closing price of the stock.",
    "open": "Today's opening price.",
    "high": "Today's highest price during trading hours.",
    "low": "Today's lowest price.",
    "return_1d": "How much the stock moved today vs yesterday, as a percentage. +5% means it gained 5%.",
    "return_20d": "How much the stock moved over the last month (~20 trading days). Bigger = stronger trend.",
    "sma_20": "Average of the last 20 days' closing prices. Smooths out daily noise to reveal the short-term trend.",
    "sma_50": "Average of the last 50 days' closing prices. Medium-term trend indicator.",
    "sma_200": "Average of the last 200 days' closing prices. Long-term trend — above this is bullish, below is bearish.",
    "above_50dma": "1 if today's price is above the 50-day average, 0 if below. Quick bullish/bearish signal.",
    "above_200dma": "1 if today's price is above the 200-day average, 0 if below. Long-term momentum flag.",
    "dist_sma20": "How far above or below the 20-day average we are, as a %. +10% = 10% above the average.",
    "dist_sma50": "Distance from the 50-day average, as a %. Tells us if we're stretched.",
    "dist_sma200": "Distance from the 200-day average, as a %. Big number = far from long-term trend.",
    "rsi_14_daily": "Daily Relative Strength Index (0-100). Above 70 = overbought, below 30 = oversold. RSI 80 means buyers have been aggressive.",
    "rsi_14_weekly": "Same as RSI but on weekly bars — gives a calmer view of momentum.",
    "rsi_14_monthly": "Same as RSI on monthly bars — long-term momentum.",
    "volume_vs_20d": "Today's traded volume divided by the 20-day average. 2x = double normal volume = something happening.",
    "traded_value_vs_20d": "Today's rupee-value traded vs 20-day average. Captures interest in expensive stocks better than volume alone.",
    "delivery_pct": "% of today's volume that was delivered (kept in demat) vs intraday-traded. High = real conviction; low = day traders.",
    "realized_vol_20d": "How jumpy the stock has been recently — standard deviation of daily returns over 20 days. Higher = riskier.",
    "adv_20d_cr": "Average daily traded value over 20 days, in crores. Liquidity measure — how easy it is to buy/sell without moving the price.",
    "avg_traded_value_20d": "Same as adv_20d_cr but in raw rupees, not crores.",

    # ━━━━━━━━━━━━━━━━ MARKET / SECTOR / MACRO ━━━━━━━━━━━━━━━━
    "market_1d_ret": "How the overall market moved today (median of all liquid stocks).",
    "market_5d_ret": "How the overall market moved over the last 5 days.",
    "market_20d_ret": "How the overall market moved over the last month.",
    "market_breadth_50dma": "% of liquid stocks trading above their 50-day average. Above 65% = broad bull rally; below 40% = broad weakness.",
    "market_breadth_200dma": "% of stocks above their 200-day average. Long-term bull/bear indicator.",
    "rel_strength_20d": "Stock's 20-day return MINUS its sector's 20-day return. Positive = stock outperforming peers.",
    "sector_5d_ret": "How the stock's sector moved over the last 5 days. Sector momentum.",
    "sector_20d_ret": "How the stock's sector moved over the last month.",
    "sector_60d_ret": "How the stock's sector moved over the last 3 months. Medium-term sector trend.",
    "sector": "Which industry classification the stock belongs to (NIFTY IT, NIFTY BANK, etc.).",

    # ━━━━━━━━━━━━━━━━ MACRO OVERLAYS (BUILT TODAY 2026-04-29) ━━━━━━━━━━━━━━━━
    "usdinr": "Today's USD/INR exchange rate. INR weakening (USDINR rising) = good for IT/Pharma exporters, bad for oil importers.",
    "usdinr_5d_chg": "How much USD/INR moved in the last 5 days. Captures short-term currency-pressure.",
    "usdinr_20d_chg": "How much USD/INR moved in the last 20 days. Captures the FX trend.",
    "eurinr": "EUR/INR rate — affects companies with Europe revenue.",
    "gbpinr": "GBP/INR rate — UK exposure.",
    "jpyinr": "JPY/INR rate — Japan exposure.",

    # ━━━━━━━━━━━━━━━━ VOLATILITY REGIME ━━━━━━━━━━━━━━━━
    "rv_60d": "Realized volatility over 60 days. Calmer markets favor different strategies than chaotic ones.",
    "vol_z_60d": "How unusual today's 20-day volatility is vs its own 60-day history (z-score). Big positive = vol spike.",
    "vol_term_20_60": "Ratio of short-term to medium-term volatility. >1 = vol breaking higher recently.",
    "vol_of_vol_60d": "How much the volatility itself swings around. Stocks with chaotic vol behave differently than steady-vol ones.",
    "vol_max_63d": "Highest realized vol seen in the last 3 months. Captures past vol shocks.",

    # ━━━━━━━━━━━━━━━━ MICROSTRUCTURE ━━━━━━━━━━━━━━━━
    "amihud_20d": "Illiquidity measure — average daily |return| ÷ rupee-volume. Higher = thinner trading = more slippage cost.",
    "turnover_skew_20d": "Skewness of recent volume. Indicates whether some big players were trading vs steady retail flow.",

    # ━━━━━━━━━━━━━━━━ WORLDQUANT-101 ALPHAS ━━━━━━━━━━━━━━━━
    "alpha_volume_signed_revert": "WorldQuant-style mean-revert: if volume rose AND price fell, expect bounce; if volume rose AND price rose, expect cool-off.",
    "alpha_intraday_norm_range": "(close - open) ÷ (high - low). Where in today's trading range we closed — captures intraday strength.",
    "alpha_high_extension_revert": "Stock breaking above its 20-day average high tends to mean-revert. Captures climax-of-rally signal.",
    "alpha_geom_mid_vs_vwap": "Comparison of (high × low)^0.5 vs the volume-weighted average price. Microstructure direction signal.",
    "alpha_open_volume_corr_10": "Negative correlation between opening price and volume over 10 days. WorldQuant-101 reversal alpha.",

    # ━━━━━━━━━━━━━━━━ CATALYSTS (CORPORATE ANNOUNCEMENTS) ━━━━━━━━━━━━━━━━
    "ann_5d_count": "Total number of corporate announcements in the last 5 days. More activity = more news flow.",
    "ann_30d_count": "Total announcements in the last 30 days.",
    "ann_order_5d": "Order-win announcements in the last 5 days. Big positive for capital-goods/EPC companies.",
    "ann_order_30d": "Order wins in the last 30 days.",
    "ann_result_5d": "Quarterly result announcements in the last 5 days. Recently-reported earnings.",
    "ann_capex_30d": "Capital-expenditure / expansion announcements in 30 days. Bullish growth signal.",
    "ann_fundraise_30d": "Capital-raising (rights issue, QIP) announcements. Can be dilutive or bullish depending on use.",
    "ann_buyback_30d": "Share buyback announcements. Strong positive — promoter confidence + share supply reduction.",
    "ann_ma_30d": "M&A announcements (mergers/acquisitions). Often catalyst for re-rating.",
    "ann_regulatory_30d": "Regulatory action announcements (SEBI, NCLT, etc.). Usually negative.",
    "ann_dividend_30d": "Dividend announcements. Income-investor signal.",
    "ann_bonussplit_30d": "Bonus issues or stock splits. Often retail-momentum trigger.",
    "ann_rating_30d": "Credit-rating announcements (CRISIL, ICRA, CARE).",
    "ann_guidance_30d": "Forward-looking guidance from management. Big information event.",
    "catalyst_score_5d": "Aggregated 'how much catalyst momentum' score over 5 days.",
    "catalyst_score_30d": "Same over 30 days.",

    # ━━━━━━━━━━━━━━━━ INSIDER / OWNERSHIP ━━━━━━━━━━━━━━━━
    "insider_net_60d_inr": "Net rupee-value of insider buying minus selling over 60 days. Positive = insiders accumulating.",
    "insider_buy_60d_inr": "Pure insider BUY value over 60 days. Strong bullish signal when promoters buy their own stock.",
    "insider_stake_delta_60d": "Change in promoter/insider stake % over 60 days.",
    "block_buy_5d_inr": "Block-deal buying value over 5 days. Captures institutional accumulation visible in NSE block-deals data.",
    "block_sell_5d_inr": "Block-deal selling value over 5 days.",
    "block_net_5d_inr": "Net block-deal flow over 5 days (buys - sells).",
    "block_buy_30d_inr": "Block-deal buying over 30 days.",
    "block_sell_30d_inr": "Block-deal selling over 30 days.",
    "block_net_30d_inr": "Net block flow over 30 days.",
    "distinct_buyers_30d": "How many distinct institutional buyers showed up in block deals (30d). More = broader institutional interest.",

    # ━━━━━━━━━━━━━━━━ NEWS / SENTIMENT / SOCIAL ━━━━━━━━━━━━━━━━
    "news_count_5d": "Number of news articles mentioning the stock in the last 5 days (RSS + per-symbol Google News).",
    "news_count_30d": "News articles mentioning the stock in 30 days.",
    "news_sentiment_5d": "Average sentiment score of news mentions (-1 = very negative, +1 = very positive). Uses our finance-tuned lexicon.",
    "news_sentiment_30d": "Same over 30 days.",
    "reddit_mentions_5d": "Number of Reddit posts mentioning the stock in 5 days. Captures retail buzz.",
    "reddit_sentiment_5d": "Average sentiment of those Reddit posts.",
    "youtube_mentions_5d": "Number of YouTube videos mentioning the stock in 5 days (Indian finance channels).",
    "youtube_sentiment_5d": "Average sentiment of those YouTube videos.",

    # ━━━━━━━━━━━━━━━━ ALT / WIKIPEDIA ━━━━━━━━━━━━━━━━
    "wiki_views": "Daily Wikipedia page views for the company. Proxy for retail public attention.",
    "wiki_views_z": "How unusual today's Wikipedia interest is vs the 30-day baseline (z-score). Big spike = something is going on retail-side.",
    "wiki_views_7d_mean": "7-day average Wikipedia views.",
    "wiki_views_30d_mean": "30-day average Wikipedia views.",

    # ━━━━━━━━━━━━━━━━ FUNDAMENTALS ━━━━━━━━━━━━━━━━
    "pe": "Price-to-Earnings ratio. How much you're paying for ₹1 of annual earnings. Lower = cheaper (loosely).",
    "sector_pe": "Average PE for the stock's sector. The peer benchmark.",
    "pe_vs_sector_ratio": "Stock PE ÷ Sector PE. <1 = trading at discount to sector; >1 = premium.",
    "week52_high": "Highest closing price in the last 52 weeks.",
    "week52_low": "Lowest closing price in the last 52 weeks.",
    "dist_from_52w_high_pct": "How far below the 52-week high we are, as %. -5% means we're 5% off the highs.",
    "dist_from_52w_low_pct": "How far above the 52-week low we are, as %.",
    "last_q_revenue": "Most recent quarterly revenue (₹ crore).",
    "last_q_pat": "Most recent quarterly profit-after-tax (₹ crore).",
    "qoq_revenue_growth": "Quarter-over-quarter revenue growth %. +20% means revenue grew 20% vs prior quarter.",
    "qoq_pat_growth": "Quarter-over-quarter PAT growth %.",

    # ━━━━━━━━━━━━━━━━ MULTI-HORIZON / MODEL OUTPUTS ━━━━━━━━━━━━━━━━
    "score_lgb": "Raw LightGBM model probability of the stock hitting +5% high within 7 days.",
    "score_xgb": "Raw XGBoost model probability of +5%/7d.",
    "score_ens": "Ensemble of LGB + XGB (50-50 blend).",
    "score_calibrated": "Isotonic-calibrated ensemble probability. 0.65 = 65% real-world chance the stock hits +5% within 7 days.",
    "score_h1_cal": "Multi-horizon model: probability of a 1-day move signal.",
    "score_h7_cal": "Multi-horizon model: probability of a 7-day move signal.",
    "score_h21_cal": "Multi-horizon model: probability of a 21-day move signal.",
    "consensus": "Geometric mean of 1d/7d/21d calibrated scores. High = all horizons agree.",
    "triangulated": "True if all 3 horizons (1d, 7d, 21d) agree the stock is in their top quartile. Strong confirmation signal.",
}


def get(feature: str) -> str:
    """Return plain-English description for a feature, or a fallback."""
    return DESCRIPTIONS.get(feature, "Engineered feature — see source script for details.")
