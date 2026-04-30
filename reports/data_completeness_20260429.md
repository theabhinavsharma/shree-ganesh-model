# Data Completeness Audit — 2026-04-29

**Liquid universe:** 2,137 symbols (EQ series, ADV ≥ ₹0.1cr/day)

## Group summary

| Group | Params tracked | Avg coverage | Min coverage |
|---|---|---|---|
| PRICE_TECHNICAL | 16 present, 0 absent | 98.2% | 85.3% ✓ |
| FUNDAMENTAL | 11 present, 0 absent | 82.6% | 71.4% |
| CATALYST | 12 present, 0 absent | 100.0% | 100.0% ✓ |
| INSIDER_PIT | 3 present, 0 absent | 100.0% | 100.0% ✓ |
| BLOCK_BULK | 7 present, 0 absent | 0.7% | 0.7% ⚠️ |
| OPTIONS_FNO | 0 present, 6 absent | 0.0% | 0.0% ⚠️ |
| SECTOR | 4 present, 0 absent | 100.0% | 100.0% ✓ |
| MARKET_MACRO | 6 present, 0 absent | 100.0% | 100.0% ✓ |
| NEWS_SOCIAL | 8 present, 0 absent | 100.0% | 100.0% ✓ |
| MACRO_SENT | 6 present, 0 absent | 100.0% | 100.0% ✓ |
| MODEL_OUTPUTS | 8 present, 0 absent | 3.2% | 2.3% ⚠️ |

## Per-param coverage

### PRICE_TECHNICAL

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `close` | ✓ | 100.0% | 2,137 | 2,137 |
| `open` | ✓ | 100.0% | 2,137 | 2,137 |
| `high` | ✓ | 100.0% | 2,137 | 2,137 |
| `low` | ✓ | 100.0% | 2,137 | 2,137 |
| `sma_20` | ✓ | 100.0% | 2,137 | 2,137 |
| `sma_50` | ✓ | 98.6% | 2,106 | 2,137 |
| `sma_200` | ✓ | 89.8% | 1,918 | 2,137 |
| `rsi_14_daily` | ✓ | 100.0% | 2,137 | 2,137 |
| `rsi_14_weekly` | ✓ | 97.9% | 2,092 | 2,137 |
| `rsi_14_monthly` | ✓ | 85.3% | 1,823 | 2,137 |
| `return_1d` | ✓ | 100.0% | 2,137 | 2,137 |
| `return_20d` | ✓ | 100.0% | 2,137 | 2,137 |
| `volume_vs_20d` | ✓ | 100.0% | 2,137 | 2,137 |
| `traded_value_vs_20d` | ✓ | 100.0% | 2,137 | 2,137 |
| `delivery_pct` | ✓ | 100.0% | 2,137 | 2,137 |
| `avg_traded_value_20d` | ✓ | 100.0% | 2,137 | 2,137 |

### FUNDAMENTAL

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `pe` | ✓ | 74.1% | 1,584 | 2,137 |
| `sector_pe` | ✓ | 74.1% | 1,584 | 2,137 |
| `pe_vs_sector_ratio` | ✓ | 72.1% | 1,541 | 2,137 |
| `week52_high` | ✓ | 99.8% | 2,133 | 2,137 |
| `week52_low` | ✓ | 99.8% | 2,133 | 2,137 |
| `dist_from_52w_high_pct` | ✓ | 99.8% | 2,132 | 2,137 |
| `dist_from_52w_low_pct` | ✓ | 99.8% | 2,132 | 2,137 |
| `last_q_revenue` | ✓ | 73.1% | 1,563 | 2,137 |
| `last_q_pat` | ✓ | 73.1% | 1,563 | 2,137 |
| `qoq_revenue_growth` | ✓ | 71.4% | 1,525 | 2,137 |
| `qoq_pat_growth` | ✓ | 71.7% | 1,532 | 2,137 |

### CATALYST

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `ann_5d_count` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_30d_count` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_order_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_order_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_result_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_capex_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_fundraise_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_buyback_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_ma_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `ann_regulatory_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `catalyst_score_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `catalyst_score_30d` | ✓ | 100.0% | 2,137 | 2,137 |

### INSIDER_PIT

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `insider_net_60d_inr` | ✓ | 100.0% | 2,137 | 2,137 |
| `insider_buy_60d_inr` | ✓ | 100.0% | 2,137 | 2,137 |
| `insider_stake_delta_60d` | ✓ | 100.0% | 2,137 | 2,137 |

### BLOCK_BULK

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `block_buy_5d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `block_sell_5d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `block_net_5d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `block_buy_30d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `block_sell_30d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `block_net_30d_inr` | ✓ | 0.7% | 15 | 2,137 |
| `distinct_buyers_30d` | ✓ | 0.7% | 15 | 2,137 |

### OPTIONS_FNO

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `atm_iv` | ❌ | 0.0% | 0 | 2,137 |
| `iv_skew` | ❌ | 0.0% | 0 | 2,137 |
| `pcr_oi` | ❌ | 0.0% | 0 | 2,137 |
| `pcr_volume` | ❌ | 0.0% | 0 | 2,137 |
| `max_pain` | ❌ | 0.0% | 0 | 2,137 |
| `max_pain_distance_pct` | ❌ | 0.0% | 0 | 2,137 |

### SECTOR

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `sector` | ✓ | 100.0% | 2,137 | 2,137 |
| `sector_5d_ret` | ✓ | 100.0% | 2,137 | 2,137 |
| `sector_20d_ret` | ✓ | 100.0% | 2,137 | 2,137 |
| `sector_60d_ret` | ✓ | 100.0% | 2,137 | 2,137 |

### MARKET_MACRO

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `market_1d_ret` | ✓ | 100.0% | 2,137 | 2,137 |
| `market_5d_ret` | ✓ | 100.0% | 2,137 | 2,137 |
| `market_20d_ret` | ✓ | 100.0% | 2,137 | 2,137 |
| `market_breadth_50dma` | ✓ | 100.0% | 2,137 | 2,137 |
| `market_breadth_200dma` | ✓ | 100.0% | 2,137 | 2,137 |
| `rel_strength_20d` | ✓ | 100.0% | 2,137 | 2,137 |

### NEWS_SOCIAL

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `news_count_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `news_sentiment_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `news_count_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `news_sentiment_30d` | ✓ | 100.0% | 2,137 | 2,137 |
| `reddit_mentions_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `reddit_sentiment_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `youtube_mentions_5d` | ✓ | 100.0% | 2,137 | 2,137 |
| `youtube_sentiment_5d` | ✓ | 100.0% | 2,137 | 2,137 |

### MACRO_SENT

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `global_macro_sent` | ✓ | 100.0% | 2,137 | 2,137 |
| `domestic_macro_sent` | ✓ | 100.0% | 2,137 | 2,137 |
| `rate_hawkish_score` | ✓ | 100.0% | 2,137 | 2,137 |
| `rate_dovish_score` | ✓ | 100.0% | 2,137 | 2,137 |
| `oil_sentiment` | ✓ | 100.0% | 2,137 | 2,137 |
| `usdinr_sentiment` | ✓ | 100.0% | 2,137 | 2,137 |

### MODEL_OUTPUTS

| Param | Present? | Coverage | n with data | n total |
|---|---|---|---|---|
| `score_ens` | ✓ | 4.7% | 100 | 2,137 |
| `score_calibrated` | ✓ | 4.7% | 100 | 2,137 |
| `short_score_calibrated` | ✓ | 4.7% | 100 | 2,137 |
| `score_h1_cal` | ✓ | 2.3% | 50 | 2,137 |
| `score_h7_cal` | ✓ | 2.3% | 50 | 2,137 |
| `score_h21_cal` | ✓ | 2.3% | 50 | 2,137 |
| `consensus` | ✓ | 2.3% | 50 | 2,137 |
| `triangulated` | ✓ | 2.3% | 50 | 2,137 |

## Action items (gaps)

- **BLOCK_BULK.block_buy_5d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.block_sell_5d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.block_net_5d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.block_buy_30d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.block_sell_30d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.block_net_30d_inr**: only 1% covered — backfill needed
- **BLOCK_BULK.distinct_buyers_30d**: only 1% covered — backfill needed
- **OPTIONS_FNO.atm_iv**: column absent — fetcher missing or not yet wired
- **OPTIONS_FNO.iv_skew**: column absent — fetcher missing or not yet wired
- **OPTIONS_FNO.pcr_oi**: column absent — fetcher missing or not yet wired
- **OPTIONS_FNO.pcr_volume**: column absent — fetcher missing or not yet wired
- **OPTIONS_FNO.max_pain**: column absent — fetcher missing or not yet wired
- **OPTIONS_FNO.max_pain_distance_pct**: column absent — fetcher missing or not yet wired
- **MODEL_OUTPUTS.score_ens**: only 5% covered — backfill needed
- **MODEL_OUTPUTS.score_calibrated**: only 5% covered — backfill needed
- **MODEL_OUTPUTS.short_score_calibrated**: only 5% covered — backfill needed
- **MODEL_OUTPUTS.score_h1_cal**: only 2% covered — backfill needed
- **MODEL_OUTPUTS.score_h7_cal**: only 2% covered — backfill needed
- **MODEL_OUTPUTS.score_h21_cal**: only 2% covered — backfill needed
- **MODEL_OUTPUTS.consensus**: only 2% covered — backfill needed
- **MODEL_OUTPUTS.triangulated**: only 2% covered — backfill needed
