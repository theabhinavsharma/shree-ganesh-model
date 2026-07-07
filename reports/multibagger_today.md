# Multibagger picks — today (2026-05-01)

## 🔴 REGIME GATE: WAIT

_market_20d=+8.70% (NOT ≤-2%); breadth_50=80% (OUT of [50,75])_

Gate v1 backtest (2024): ALL-IN 41% success → GATED 64% success (+23pp). Regime gate identifies a meaningful subset of weeks when the strategy works.
**TODAY: WAIT.** Even though names below pass the score bar, the regime doesn't match the historical success pattern. Deploy when market_20d ≤ -2% AND breadth_50 is 50-75%.

## Names that would clear the score bar (regardless of regime)

Score thresholds verified historically on 2024-2025 OOS:

- **100% in 180d** → score ≥ 0.86 (9,933 OOS, 90% hit rate)
- **100% in 252d** → score ≥ 0.84 (13,950 OOS, 90% hit rate)
- **100% in 378d** → score ≥ 0.77 (8,304 OOS, 90% hit rate)

**Caveat:** Real prospective hit rate ~40% basket-level (not 90%). The 90% claim is in-sample artifact.
Use these as candidates ONLY when the regime gate flips to DEPLOY.

## +100% in 180d (score ≥ 0.86)

⚠️ No name today clears the 0.86 threshold for this target.

## +100% in 252d (score ≥ 0.84)

⚠️ No name today clears the 0.84 threshold for this target.

## +100% in 378d (score ≥ 0.77)

⚠️ No name today clears the 0.77 threshold for this target.

## Union: any name clearing the 100%-double bar at any of (180/252/378d)

⚠️ No name today clears the 100%-double bar at any horizon.

## How to act on this list

1. **Universe:** the top names with high `best_score_100pct`
2. **Sizing:** spread 5-8% per name across 5-10 names = 25-80% of capital
3. **Hold:** 6-12 months minimum; do NOT trade on weekly noise
4. **Stop:** -25% from entry (these are long-horizon bets; tight stops kill the strategy)
5. **Re-evaluate:** rerun this script weekly; rebalance if names drop below 0.70 score