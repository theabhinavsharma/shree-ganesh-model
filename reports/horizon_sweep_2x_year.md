# Horizon sweep — which bucket doubles capital within a year? (EXP-2026-09-24-horizon-sweep-2x-year)

_generated 2026-09-23 21:27 · weekly entries 2018-01..(last with a complete outcome) · walk-forward LGBM per bucket, top-8/week, next-open entry, sell at +X% or day-H close, 0.30% RT, no stop, H/5-tranche ladder, realized-only equity_

| bucket | era | cohorts | hit % | cohort net % | needed/cycle for 2x/yr % | 12m windows | **P(≥2x in 12m) %** | P(≥1.5x) % | median 12m × | 10th pct 12m × | baseline P(2x) % | baseline median × |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15d/5% | disc | 254 | 67.8 | -0.19 | 4.2 | 254 | **0.0** | 0.0 | 1.05 | 0.75 | 0.0 | 0.96 |
| 15d/5% | conf | 190 | 72.2 | 0.45 | 4.2 | 138 | **0.0** | 0.0 | 1.04 | 0.96 | 0.0 | 1.01 |
| 30d/10% | disc | 254 | 59.3 | 0.17 | 8.6 | 254 | **0.0** | 0.0 | 1.05 | 0.76 | 0.0 | 0.99 |
| 30d/10% | conf | 187 | 63.2 | 1.43 | 8.6 | 135 | **0.0** | 0.0 | 1.07 | 0.93 | 0.0 | 1.01 |
| 60d/20% | disc | 254 | 44.6 | 0.32 | 17.9 | 254 | **0.0** | 8.7 | 1.1 | 0.66 | 0.0 | 1.03 |
| 60d/20% | conf | 181 | 52.8 | 4.13 | 17.9 | 129 | **0.0** | 0.0 | 1.11 | 0.95 | 0.0 | 1.03 |
| 126d/40% | disc | 254 | 32.4 | 1.32 | 41.4 | 254 | **0.0** | 14.6 | 1.07 | 0.67 | 0.0 | 1.02 |
| 126d/40% | conf | 167 | 39.8 | 8.21 | 41.4 | 115 | **0.0** | 19.1 | 1.26 | 0.99 | 0.0 | 1.24 |
| 252d/100% | disc | 254 | 20.3 | 15.16 | 100.0 | 254 | **0.0** | 14.6 | 1.06 | 0.83 | 0.0 | 1.06 |
| 252d/100% | conf | 142 | 25.4 | 24.78 | 100.0 | 90 | **0.0** | 42.2 | 1.45 | 1.28 | 0.0 | 1.34 |

**Ranking (registered rule: P(2x/12m) in the weaker era, tie → higher 10th pct):** 252d/100% > 30d/10% > 15d/5% > 126d/40% > 60d/20%

## Caveats

- Realized-only equity: open positions are not marked to market, so drawdowns are understated.
- Gap-ups through the target are filled at the target (conservative); no stop-loss in any bucket.
- Price-only model: the engines' 19 tape features. 252d windows starting after ~Sep 2024 can't close yet, so conf-era long buckets have fewer windows.
- 5 buckets tested; all reported.
