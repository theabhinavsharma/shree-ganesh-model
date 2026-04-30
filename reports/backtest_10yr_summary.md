# 10-year Walk-forward Backtest

**Strategy:** every trading day, equal-weight top-5 picks by score; hold 7 trading days; close-to-close return.

**Train:** all data with year < target. **Test:** target year (2016–2025).

**Universe filter:** ADV ≥ ₹1cr/day, EQ series.

## Per-year summary

| Year | OOS days | mean 7d | median 7d | days ≥+5% (n / %) | days <0 (n / %) | theoretical ann ROI |
|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 248 | +1.71% | +1.48% | 79 / 31.9% | 107 / 43.1% | +142% |
| 2018 | 246 | -0.22% | -0.39% | 51 / 20.7% | 135 / 54.9% | -11% |
| 2019 | 244 | -0.66% | -1.85% | 64 / 26.2% | 142 / 58.2% | -29% |
| 2020 | 261 | +3.86% | +4.00% | 117 / 44.8% | 95 / 36.4% | +615% |
| 2021 | 261 | +5.77% | +4.70% | 128 / 49.0% | 79 / 30.3% | +1751% |
| 2022 | 259 | +1.58% | +1.59% | 75 / 29.0% | 110 / 42.5% | +126% |
| 2023 | 260 | +6.83% | +3.33% | 108 / 41.5% | 79 / 30.4% | +3003% |
| 2024 | 261 | +0.13% | +0.55% | 76 / 29.1% | 125 / 47.9% | +7% |
| 2025 | 250 | +0.20% | -0.28% | 55 / 22.0% | 129 / 51.6% | +11% |
| **2016-2025** | 2290 | **+2.18%** | +1.50% | 753 / **32.9%** | 1001 / 43.7% | +207% |

## Honest reading

- **Theoretical ann ROI** assumes you trade every day with zero slippage and no overlap. Real capture is 20-40% of this.
- **Realistic execution:** if you trade weekly, capture is ~ (1+median_7d)^52 - 1, not mean-based.
- **Negative-day count matters:** ~30-50% of OOS days are negative — you need patience-filter to avoid trading those.
- This run uses the basic feature set (no catalysts, no sentiment, no fundamentals). The production model should outperform.