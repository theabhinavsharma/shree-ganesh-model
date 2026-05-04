# Event-driven, fixed-capital, signal-gated backtest

Mirrors the user's actual behavior:
- Score every stock every day (OOF, walk-forward, isotonic-calibrated).
- Trade ONLY when ≥1 name fires at the calibrated bar (`score_cal ≥ GATE`).
- Sit in cash @ 7% ann on no-fire days.
- Capital ROTATES: when a position exits (target / SL / time-out), the freed cash refills slots.
- Fixed starting capital ₹100; equity curve compounds through time.

Universe: 9-year walk-forward, 2017-01-02 → 2025-12-31 (2290 trading days)
Cash yield: 7% ann (LIQUIDPLUS proxy)

## Configurations

| Config | GATE | Slots | Hold | Target | SL | n_trades | Hit % | Avg/trade | Deployed % | **CAGR** | Max DD | Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single-name 0.95 | 0.95 | 1 | 7d | +5% | -3% | 71 | 46% | +0.72% | 5% | **+12.2%** | -17.7% | 1.11 |
| Top-3 basket 0.95 | 0.95 | 3 | 7d | +5% | -3% | 182 | 47% | +0.77% | 4% | **+12.1%** | -11.9% | 1.56 |
| Top-5 basket 0.95 | 0.95 | 5 | 7d | +5% | -3% | 283 | 47% | +0.78% | 4% | **+11.8%** | -13.5% | 1.63 |
| Top-5 basket 0.95 noSL | 0.95 | 5 | 7d | +5% | none | 150 | 73% | +1.28% | 5% | **+10.9%** | -17.2% | 1.12 |
| Single-name 0.85 | 0.85 | 1 | 7d | +5% | -3% | 136 | 51% | +1.12% | 24% | **+23.1%** | -17.7% | 1.39 |
| Top-5 basket 0.85 | 0.85 | 5 | 7d | +5% | -3% | 571 | 55% | +1.39% | 18% | **+26.4%** | -13.5% | 2.54 |
| Top-5 basket 0.80 | 0.80 | 5 | 7d | +5% | -3% | 758 | 57% | +1.58% | 20% | **+38.8%** | -13.5% | 3.08 |
| Single-name 0.95 wider | 0.95 | 1 | 10d | +7% | -4% | 51 | 41% | +0.35% | 6% | **+7.9%** | -29.1% | 0.69 |
| Top-10 basket 0.80 | 0.80 | 10 | 7d | +5% | -3% | 1246 | 57% | +1.52% | 15% | **+31.3%** | -9.6% | 3.21 |
| Top-10 basket 0.75 | 0.75 | 10 | 7d | +5% | -3% | 2662 | 53% | +1.22% | 30% | **+51.8%** | -19.5% | 3.63 |
| Top-10 basket 0.70 | 0.70 | 10 | 7d | +5% | -3% | 3591 | 56% | +1.44% | 43% | **+93.5%** | -14.1% | 5.03 |
| Top-10 basket 0.65 | 0.65 | 10 | 7d | +5% | -3% | 5174 | 56% | +1.50% | 60% | **+154.8%** | -20.1% | 6.18 |
| Top-20 basket 0.65 | 0.65 | 20 | 7d | +5% | -3% | 9235 | 54% | +1.30% | 50% | **+107.6%** | -18.8% | 5.69 |
| Top-5 basket 0.80 hold15 | 0.80 | 5 | 15d | +10% | -5% | 487 | 52% | +2.79% | 25% | **+42.4%** | -14.2% | 2.69 |
| Top-5 basket 0.80 hold30 | 0.80 | 5 | 30d | +20% | -7% | 285 | 43% | +3.98% | 34% | **+37.2%** | -16.5% | 1.97 |
| Top-5 basket 0.80 noSL | 0.80 | 5 | 7d | +5% | none | 431 | 79% | +1.22% | 27% | **+17.7%** | -36.1% | 1.02 |
| Top-5 basket 0.70 noSL | 0.70 | 5 | 7d | +5% | none | 1077 | 77% | +1.29% | 61% | **+38.6%** | -49.5% | 1.51 |
| Top-10 basket 0.70 noSL | 0.70 | 10 | 7d | +5% | none | 2037 | 76% | +1.24% | 54% | **+36.4%** | -35.3% | 1.74 |
| Top-5 basket 0.80 widerT | 0.80 | 5 | 10d | +8% | -4% | 627 | 54% | +2.39% | 21% | **+48.5%** | -18.2% | 3.00 |
| Single-name 0.70 | 0.70 | 1 | 7d | +5% | -3% | 330 | 60% | +1.77% | 63% | **+90.2%** | -32.3% | 2.63 |

## Per-year returns (0.95-bar configs)

| Year | Single 0.95 | Top-3 0.95 | Top-5 0.95 | Top-5 0.85 |
|---|---:|---:|---:|---:|
| 2017 | +19.9% | +11.8% | +9.8% | +9.8% |
| 2018 | +37.9% | +48.1% | +45.7% | +45.7% |
| 2019 | +6.7% | +6.7% | +6.7% | +6.7% |
| 2020 | +7.2% | +7.2% | +7.2% | +6.6% |
| 2021 | +7.2% | +7.2% | +7.2% | +7.2% |
| 2022 | +7.2% | +7.2% | +7.2% | +54.3% |
| 2023 | +4.0% | +7.9% | +7.6% | +22.3% |
| 2024 | +16.6% | +11.2% | +12.6% | +114.5% |
| 2025 | +6.9% | +6.9% | +6.9% | +3.5% |

## Reading this

- **Avg/trade** is the per-position close-to-close return when in a position.
- **Deployed %** is the fraction of trading days the portfolio is invested (vs in cash).
- **CAGR** is the compounded equity-curve growth — the honest number that matches what fixed capital would actually do.
- **Max DD** is the deepest peak-to-trough drawdown of the equity curve, NOT a single-trade SL.

### Why this differs from the prior 9-year backtest

The earlier number forced top-5 baskets EVERY day (2,290 days × 5 names = 11,450 trades). Most days the model wasn't above 0.95 — those were forced trades on noise. By gating to fire-only, we cut to ~3,000 high-conviction trades over 9 years and let cash earn 7% on the other ~95% of days.
