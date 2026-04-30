# Architecture

A 5-layer agentic trading system for the Indian (NSE) equity universe.
Built to produce honest, calibrated 7-day forward predictions with a
hard discipline gate that refuses to trade on low-conviction days.

## High-level topology — 5 layers

```mermaid
flowchart TD
    LA[macOS LaunchAgent: daily 18:00 IST]
    CR[Cron weekly: Sun 19:00]
    PR[paper_trading_recorder: live OOS calibration]

    L1[Layer 1: DATA<br>14 fetchers - prices, catalysts, fundamentals,<br>news, FX, FII/DII, Wikipedia]
    L2[Layer 2: FEATURE<br>30+ engineered features - WQ alphas,<br>volatility regime, sector overlay, sentiment]
    L3[Layer 3: MODEL<br>4 ensembles - long, short, multi-horizon,<br>sector-weak overlay]
    L4[Layer 4: DISCIPLINE<br>completeness audit + 8-gate filter cascade<br>refuses to trade if any gate fails]
    L5[Layer 5: OUTPUT<br>daily_pro_brief, status dashboard,<br>actionable_today.csv, inspect_symbol]

    LA --> L1
    CR --> AL[agent_loop.py: hypothesis cycle]
    AL --> L2
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    L5 --> PR
    PR --> L3

    style L1 fill:#dbeafe,stroke:#1e40af,color:#000
    style L2 fill:#d1fae5,stroke:#065f46,color:#000
    style L3 fill:#ede9fe,stroke:#6d28d9,color:#000
    style L4 fill:#fef3c7,stroke:#b45309,color:#000
    style L5 fill:#fce7f3,stroke:#a21caf,color:#000
    style LA fill:#1f2937,color:#fff
    style CR fill:#1f2937,color:#fff
    style AL fill:#1f2937,color:#fff
    style PR fill:#1f2937,color:#fff
```

## Agent loop (weekly hypothesis cycle)

```mermaid
flowchart LR
    REG[(factor_registry.json: 75 hypotheses)]
    P{state PROPOSED?}
    HD{has_data?}
    CO[compile via feature_factory]
    FM[FETCHER_MAP lookup]
    RUN[run fetcher script]
    BL[state = BLOCKED]
    EV[factor_evaluator.py: IC + IR gate]
    IC{IC >= 0.02 AND IR >= 0.5?}
    ICP[state = IC_PASSED: awaits portfolio A/B]
    DR[state = DROP]
    AB[backtest_10yr_with_factors: portfolio A/B]
    LIFT{lift_pp >= 0.30?}
    KEEP[state = KEEP: ship to production]
    DAB[state = DROP_AB_FAIL]

    REG --> P
    P --> HD
    HD -- yes --> CO
    HD -- no --> FM
    FM -- exists --> RUN
    FM -- missing --> BL
    RUN --> CO
    CO --> EV
    EV --> IC
    IC -- yes --> ICP
    IC -- no --> DR
    ICP --> AB
    AB --> LIFT
    LIFT -- yes --> KEEP
    LIFT -- no --> DAB
    KEEP --> REG
    DAB --> REG
    DR --> REG
    BL --> REG

    style KEEP fill:#10b981,color:#fff
    style DAB fill:#ef4444,color:#fff
    style DR fill:#ef4444,color:#fff
    style BL fill:#6b7280,color:#fff
    style ICP fill:#fbbf24,color:#000
```

## Discipline cascade (the trade gate)

```mermaid
flowchart TD
    U[Liquid universe: 2479 EQ names ADV >= 0.1cr]
    G2[Has 5 core features<br>close, sma_50, rsi, ret_20d, vol]
    G3[Has model score]
    G4[Macro overlay<br>RISK_OFF: floor 0.75<br>NEUTRAL: floor 0.65]
    G5[RSI sanity: 20 to 90]
    G6[Liquidity sizing<br>ADV >= 5cr: 8 percent<br>else 4 percent]
    G7[Sector cap 25 percent]
    G8[Top-N selection]
    NT[NO TRADE: park in LIQUIDPLUS]
    GO[Actionable list]

    U --> G2 --> G3 --> G4 --> G5 --> G6 --> G7 --> G8
    G8 -- n equals 0 --> NT
    G8 -- n greater than 0 --> GO

    style U fill:#dbeafe,stroke:#1e40af,color:#000
    style NT fill:#ef4444,color:#fff
    style GO fill:#10b981,color:#fff
```

## File index (single source of truth)

| Layer | Files | Purpose |
|---|---|---|
| **Control** | `daily_pipeline.sh` · `com.zoom.daily-pipeline.plist` · `agent_loop.py` | Orchestration |
| **Data** | `refresh_prices.py` · `refresh_announcements.py` · `catalyst_tagger.py` · `build_catalyst_features.py` · `fetch_block_deals.py` · `fetch_news_rss.py` · `fetch_news_per_symbol.py` · `fetch_reddit.py` · `fetch_youtube.py` · `fetch_options_chain.py` · `fetch_fundamentals.py` · `fetch_forex_macro.py` · `fetch_fii_dii.py` · `fetch_wiki_pageviews.py` · `score_sentiment.py` | 15 ingesters |
| **Feature** | `feature_factory.py` · `factor_registry.py` · `factor_evaluator.py` · `compute_feature_importance.py` | Compile + evaluate alphas |
| **Model** | `run_v3_with_catalysts.py` · `run_short_side.py` · `run_multi_horizon.py` · `sector_weak_shorts.py` · `portfolio_sizer.py` | 4 ensembles + sizer |
| **Discipline** | `data_completeness.py` · `filter_cascade.py` · `paper_trading_recorder.py` | Gate enforcement + live calibration |
| **Output** | `generate_pro_brief.py` · `generate_daily_brief.py` · `inspect_symbol.py` · `build_workflow_diagram.py` · `build_status_dashboard.py` · `build_dashboard.py` · `build_html_viewer.py` | Reports + dashboards |
| **Backtest** | `backtest_10yr.py` · `backtest_10yr_with_factors.py` | Walk-forward validation |

Every file lives in `src/agentic/`. Total: ~28 Python scripts + 1 bash orchestrator + 1 plist.

## Honest performance — 10-year walk-forward

| Year | Mean 7d | Days >= +5% | Days < 0 |
|---:|---:|---:|---:|
| 2017 | +1.71% | 32% | 43% |
| 2018 | -0.22% | 21% | 55% |
| 2019 | -0.66% | 26% | 58% |
| 2020 | +3.86% | 45% | 36% |
| 2021 | +5.77% | 49% | 30% |
| 2022 | +1.58% | 29% | 43% |
| 2023 | +6.83% | 42% | 30% |
| 2024 | +0.13% | 29% | 48% |
| 2025 | +0.20% | 22% | 52% |
| **2016-2025** | **+2.18%** | **33%** | **44%** |

Realistic annualised ROI: **30-50% unlevered** (theoretical compound × 30% real-world capture).

## What this system is NOT

- Not a magic 4,000% machine. The original framing was wrong; honest forward expectation is 30-50% ann.
- Not survivable in 2018-style bear regimes without the discipline gate.
- Not validated on production v3's catalyst+sentiment lift — those need rigorous A/B (queued).
- Not real-time intraday — bhavcopy lands at ~5pm IST; brief runs at 18:00.
- Not for forced trades. **No-trade days exist and are mandatory.**
